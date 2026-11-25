"""Event-driven Mode-2-faithful transport (R1–R3 semantics, no single delay constant).

Sequence for a remote stage:
  COMMIT_REQ → ACK → input fragments → helper CPU → output fragments → readiness

Uses TB size 1180 B, reservation/opportunity period, contact gating, and stranding.
Does not simulate full NR PHY/SPS; advances on MAC opportunities + contact like the
verified WorkflowMode2Adapter + control protocol path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

TB_BYTES = 1180
# Edge payload sizes (diamond-4), bytes
INPUT_BYTES = {"A": 1500, "B": 900, "C": 900, "D": 600}
OUTPUT_BYTES = {"A": 900, "B": 600, "C": 600, "D": 300}


@dataclass
class Frag:
    kind: str  # CTRL_REQ CTRL_ACK DATA_IN DATA_OUT
    wf: int
    sid: str
    src: int
    dst: int
    seq: int
    nfrag: int
    bytes: int
    ready_t: float = 0.0  # earliest send time


@dataclass
class RemoteFlow:
    wf: int
    sid: str
    helper: int
    origin: int = 0
    phase: str = "REQ"  # REQ ACK_WAIT IN_TX CPU OUT_TX DONE FAIL
    req_sent: bool = False
    ack: bool = False
    in_done: int = 0
    in_need: int = 1
    out_done: int = 0
    out_need: int = 1
    cpu_enqueued: bool = False
    cycles: float = 0.0


@dataclass
class Mode2Radio:
    """Per-vehicle TX opportunity clock + pending fragments."""

    alloc_period_s: float = 0.100
    ctrl_slots: float = 1.0  # ctrl consumes one opportunity
    next_opp: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    tx_q: List[List[Frag]] = field(default_factory=lambda: [[], [], []])
    flows: List[RemoteFlow] = field(default_factory=list)
    # delivery events: (t_deliver, kind, wf, sid, helper)
    deliver_events: List[Tuple[float, str, int, str, int]] = field(default_factory=list)

    def reset(self, n_veh: int = 3):
        self.next_opp = [0.0] * n_veh
        self.tx_q = [[] for _ in range(n_veh)]
        self.flows = []
        self.deliver_events = []

    def start_remote(self, t: float, wf: int, sid: str, helper: int, cycles: float, origin: int = 0):
        inn = max(1, (INPUT_BYTES.get(sid, 900) + TB_BYTES - 1) // TB_BYTES)
        outn = max(1, (OUTPUT_BYTES.get(sid, 600) + TB_BYTES - 1) // TB_BYTES)
        flow = RemoteFlow(
            wf=wf, sid=sid, helper=helper, origin=origin,
            phase="REQ", in_need=inn, out_need=outn, cycles=cycles,
        )
        self.flows.append(flow)
        # enqueue COMMIT_REQ from origin → helper
        self.tx_q[origin].append(
            Frag("CTRL_REQ", wf, sid, origin, helper, 1, 1, 200, t)
        )

    def next_event_time(self, t: float) -> Optional[float]:
        times = []
        for ev in self.deliver_events:
            if ev[0] > t + 1e-15:
                times.append(ev[0])
        for v, q in enumerate(self.tx_q):
            if q:
                times.append(max(t, self.next_opp[v], q[0].ready_t))
        return min(times) if times else None

    def advance(
        self,
        t: float,
        next_t: float,
        contact,  # bool matrix
        rng,
    ) -> List[Tuple[str, int, str, int]]:
        """Advance radio to next_t. Returns list of (event, wf, sid, helper)."""
        completed: List[Tuple[str, int, str, int]] = []
        # process deliveries due
        keep = []
        for ev in self.deliver_events:
            td, kind, wf, sid, helper = ev
            if td <= next_t + 1e-12:
                completed.extend(self._on_deliver(td, kind, wf, sid, helper, contact, rng))
            else:
                keep.append(ev)
        self.deliver_events = keep

        # attempt TX opportunities in [t, next_t]
        for v in range(len(self.tx_q)):
            while self.tx_q[v]:
                frag = self.tx_q[v][0]
                opp = max(self.next_opp[v], frag.ready_t, t)
                if opp > next_t + 1e-12:
                    break
                # contact required for non-local
                if frag.src != frag.dst and not contact[frag.src, frag.dst]:
                    # cannot send — drop frag / fail flow if contact gone
                    self.tx_q[v].pop(0)
                    self._fail_flow(frag.wf, frag.sid, completed)
                    self.next_opp[v] = opp + self.alloc_period_s
                    continue
                # send at opp; delivery after one allocation period (Mode-2 reservation-ish)
                self.tx_q[v].pop(0)
                self.next_opp[v] = opp + self.alloc_period_s
                delay = self.alloc_period_s  # air + decode within period
                self.deliver_events.append(
                    (opp + delay, frag.kind, frag.wf, frag.sid, frag.dst if frag.kind.startswith("CTRL") or frag.kind.startswith("DATA") else frag.dst)
                )
                # fix helper id for flow matching: store helper as flow.helper in deliver
                # Overwrite last event helper field to flow helper for DATA/CTRL targeting
                td, kind, wf, sid, _ = self.deliver_events[-1]
                helper = self._flow_helper(wf, sid)
                self.deliver_events[-1] = (td, kind, wf, sid, helper if helper is not None else frag.dst)
        return completed

    def _flow_helper(self, wf: int, sid: str) -> Optional[int]:
        for f in self.flows:
            if f.wf == wf and f.sid == sid and f.phase != "DONE" and f.phase != "FAIL":
                return f.helper
        return None

    def _fail_flow(self, wf: int, sid: str, completed: List):
        for f in self.flows:
            if f.wf == wf and f.sid == sid and f.phase not in ("DONE", "FAIL"):
                f.phase = "FAIL"
                completed.append(("FAIL", wf, sid, f.helper))

    def _on_deliver(self, t, kind, wf, sid, helper, contact, rng) -> List[Tuple[str, int, str, int]]:
        out: List[Tuple[str, int, str, int]] = []
        flow = None
        for f in self.flows:
            if f.wf == wf and f.sid == sid and f.phase not in ("DONE", "FAIL"):
                flow = f
                break
        if flow is None:
            return out
        if kind == "CTRL_REQ":
            # helper replies ACK if contact
            if contact[flow.helper, flow.origin]:
                self.tx_q[flow.helper].append(
                    Frag("CTRL_ACK", wf, sid, flow.helper, flow.origin, 1, 1, 100, t)
                )
            else:
                flow.phase = "FAIL"
                out.append(("FAIL", wf, sid, flow.helper))
            return out
        if kind == "CTRL_ACK":
            flow.ack = True
            flow.phase = "IN_TX"
            # enqueue input fragments origin → helper
            for i in range(flow.in_need):
                self.tx_q[flow.origin].append(
                    Frag("DATA_IN", wf, sid, flow.origin, flow.helper, i + 1, flow.in_need, TB_BYTES, t)
                )
            return out
        if kind == "DATA_IN":
            flow.in_done += 1
            if flow.in_done >= flow.in_need:
                flow.phase = "CPU"
                out.append(("INPUT_READY", wf, sid, flow.helper))
            return out
        if kind == "DATA_OUT":
            flow.out_done += 1
            if flow.out_done >= flow.out_need:
                flow.phase = "DONE"
                out.append(("OUTPUT_READY", wf, sid, flow.helper))
            return out
        return out

    def on_cpu_finished(self, t: float, wf: int, sid: str):
        for f in self.flows:
            if f.wf == wf and f.sid == sid and f.phase == "CPU":
                f.phase = "OUT_TX"
                for i in range(f.out_need):
                    self.tx_q[f.helper].append(
                        Frag("DATA_OUT", wf, sid, f.helper, f.origin, i + 1, f.out_need, TB_BYTES, t)
                    )
                return
