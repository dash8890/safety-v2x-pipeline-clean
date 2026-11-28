"""Diamond-4 workflow MDP for deadline-aware joint offloading (R4 + corrections).

R4 reference: privileged contact timing + abstract delay (frozen checkpoint).
Correction A: causal/local contact features only (no schedule leakage).
Correction B: radio_mode=mode2_event — COMMIT→ACK→input→CPU→output (Mode-2-faithful).

Workload/objective/actions unchanged from R4. Do not overwrite ddqn_best.pt.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mode2_radio import Mode2Radio

# Vehicle ids: 0=origin, 1=helper2, 2=helper3
ORIGIN, H2, H3 = 0, 1, 2
N_VEH = 3

# Actions
A_LOCAL, A_H2, A_H3, A_HOLD, A_DROP = 0, 1, 2, 3, 4
N_ACTIONS = 5

STAGE_CYCLES = {"A": 9e6, "B": 12e6, "C": 12e6, "D": 3e6}
EDGE_BYTES = {
    "root_A": 1500, "A_B": 900, "A_C": 900, "B_D": 600, "C_D": 600, "D_out": 300
}
PRED = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}
SUCC = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}


@dataclass
class Stage:
    wf: int
    sid: str
    cycles: float
    status: str = "WAITING"  # WAITING ELIGIBLE RUNNING DONE
    place: int = -1
    remain: float = 0.0
    finish_t: float = np.nan
    blocked_until: float = 0.0
    ban_helpers: tuple = ()
    defer_count: int = 0


@dataclass
class Workflow:
    wf: int
    release: float
    deadline: float
    value: float = 1.0
    status: str = "ACTIVE"  # ACTIVE SUCCESS EXPIRED
    stages: Dict[str, Stage] = field(default_factory=dict)


@dataclass
class Job:
    veh: int
    wf: int
    sid: str
    remain: float
    t_done: float = np.nan


class WorkflowMDP:
    """Discrete-event abstract simulator with RL decision points."""

    def __init__(self, seed: int = 0, cfg: Optional[dict] = None):
        self.cfg = {
            # Scarce capacity so priority/value selection matters (VEC prioritized offloading)
            "cpu_hz": np.array([3.0e8, 3.5e9, 1.05e9], dtype=np.float64),
            "link_delay_s": 0.024,
            "ctrl_delay_s": 0.006,
            "deadline_s": 0.36,
            "arrivals": np.arange(0.04, 0.04 + 20 * 0.016, 0.016),
            "horizon_s": 1.45,
            "dt_hold_s": 0.010,
            "bg_cycles_origin": 14e6,
            "contact_exit": 0.10,
            "contact_return": 0.62,
            "max_steps": 550,
            "link_jitter": True,
            # safety-critical vs best-effort
            "value_high": 5.0,
            "value_low": 1.0,
            "p_high_value": 0.30,
            # observation: "privileged" (R4 ref) | "causal" (correction A)
            "obs_mode": "causal",
            # radio: "abstract" (R4) | "mode2_event" (correction B)
            "radio_mode": "abstract",
            "advertisement_period_s": 0.100,
            "advertisement_ttl_s": 0.300,
            # Per-opportunity period for event-driven Mode-2 path (TB/ctrl).
            # Keeps multi-step COMMIT→ACK→frags→CPU→frags; not a single delay constant.
            # 20 ms ≈ aggregated NR-V2X opportunity under R4 deadlines (100 ms SPS too harsh alone).
            "mode2_alloc_period_s": 0.020,
            # reward_mode: standard | utility_focus
            "reward_mode": "standard",
            # if True: early arrivals low-biased, late arrivals high-biased (VG threshold fails)
            "phased_values": False,
            "phase_split_s": 0.20,
            "p_high_early": 0.12,
            "p_high_late": 0.55,
        }
        if cfg:
            self.cfg.update(cfg)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        self.t = 0.0
        self.step_i = 0
        self.contact = np.ones((N_VEH, N_VEH), dtype=bool)
        np.fill_diagonal(self.contact, True)
        self.queues: List[List[Job]] = [[] for _ in range(N_VEH)]
        self.busy_remain = np.zeros(N_VEH)
        self.busy_job: List[Optional[Job]] = [None] * N_VEH
        self.wfs: Dict[int, Workflow] = {}
        self.next_wf = 1
        self.pending_arrivals = list(self.cfg["arrivals"])
        self.tx_events: List[Tuple[float, int, str, int]] = []  # t, wf, sid, place
        self.n_success = 0
        self.n_expired = 0
        self.n_remote = 0
        self.total_value = 0.0
        self.pending_decision: Optional[Tuple[int, str]] = None
        self.done = False
        # Causal contact / advertisement state (ego-observable only)
        self.last_contact_change_t = {H2: 0.0, H3: 0.0}
        self.last_adv_rx_t = {H2: -np.inf, H3: -np.inf}
        self._next_adv_t = 0.0
        self.radio = Mode2Radio(alloc_period_s=float(self.cfg["mode2_alloc_period_s"]))
        self.radio.reset(N_VEH)
        # seed background on origin
        if self.cfg["bg_cycles_origin"] > 0:
            self._enqueue(ORIGIN, Job(ORIGIN, -1, "BG", self.cfg["bg_cycles_origin"]))
        self._ingest_arrivals()
        self._update_contact()
        self._refresh_eligible()
        obs, _ = self._ensure_decision_or_advance()
        return obs

    # ----- public API -----
    def step(self, action: int):
        if self.done:
            return self._obs_idle(), 0.0, True, self._info()
        assert self.pending_decision is not None
        wf_id, sid = self.pending_decision
        reward = 0.0
        reward += self._apply_action(wf_id, sid, int(action))
        self.pending_decision = None
        self.step_i += 1
        obs, term_r = self._ensure_decision_or_advance()
        reward += term_r
        if self.step_i >= self.cfg["max_steps"] or self.t >= self.cfg["horizon_s"]:
            self._expire_all_remaining()
            self.done = True
        if self.done:
            # small shaping: fraction completed
            n = max(len(self.cfg["arrivals"]), 1)
            reward += 0.1 * (self.n_success / n)
        return obs, float(reward), self.done, self._info()

    def run_policy(self, policy_fn, seed: Optional[int] = None) -> dict:
        obs = self.reset(seed)
        done = False
        while not done:
            mask = self.action_mask()
            a = policy_fn(obs, mask, self)
            obs, r, done, info = self.step(a)
        return info

    def action_mask(self) -> np.ndarray:
        mask = np.ones(N_ACTIONS, dtype=bool)
        if self.pending_decision is None:
            return mask
        wf_id, sid = self.pending_decision
        # A and D: force local (DAG anchors); HOLD allowed only for B/C
        wf = self.wfs[wf_id]
        st = wf.stages[sid]
        if sid in ("A", "D"):
            mask[:] = False
            mask[A_LOCAL] = True
            mask[A_DROP] = True  # admission reject allowed at anchor
            return mask
        ban = set(st.ban_helpers)
        if (not self.contact[ORIGIN, H2]) or (H2 in ban):
            mask[A_H2] = False
        if (not self.contact[ORIGIN, H3]) or (H3 in ban):
            mask[A_H3] = False
        return mask

    # ----- internals -----
    def _info(self) -> dict:
        n_wf = len(self.cfg["arrivals"])
        return {
            "n_success": self.n_success,
            "n_expired": self.n_expired,
            "n_workflows": n_wf,
            "n_remote": self.n_remote,
            "total_value": self.total_value,
            "t": self.t,
            "seed": self.seed,
        }

    def _obs_idle(self) -> np.ndarray:
        return np.zeros(self.obs_dim, dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        return 15

    def _obs(self) -> np.ndarray:
        qload = np.array([self._queued_cycles(v) for v in range(N_VEH)], dtype=np.float64)
        qnorm = qload / (self.cfg["cpu_hz"] * self.cfg["deadline_s"] + 1e-9)
        c2 = 1.0 if self.contact[ORIGIN, H2] else 0.0
        c3 = 1.0 if self.contact[ORIGIN, H3] else 0.0
        if self.pending_decision is None:
            slack = 0.0
            stype = 0.0
            cycles_n = 0.0
            val_n = 0.0
            n_elig = 0.0
        else:
            wf_id, sid = self.pending_decision
            wf = self.wfs[wf_id]
            slack = (wf.deadline - self.t) / self.cfg["deadline_s"]
            stype = {"A": 0.0, "B": 0.33, "C": 0.66, "D": 1.0}[sid]
            cycles_n = STAGE_CYCLES[sid] / 12e6
            val_n = wf.value / self.cfg["value_high"]
            n_elig = sum(
                1
                for w in self.wfs.values()
                if w.status == "ACTIVE"
                for st in w.stages.values()
                if st.status == "ELIGIBLE"
            ) / 8.0
        n_active = sum(1 for w in self.wfs.values() if w.status == "ACTIVE") / max(
            len(self.cfg["arrivals"]), 1
        )
        if self.cfg.get("obs_mode", "causal") == "privileged":
            # R4 reference only — exact future schedule (not deployable)
            te = self.cfg["contact_exit"]
            tr = self.cfg["contact_return"]
            f10 = max(0.0, te - self.t) / self.cfg["horizon_s"]
            f11 = max(0.0, tr - self.t) / self.cfg["horizon_s"]
        else:
            # Causal: contact age if up; advertisement age from last RX (else TTL)
            f10 = self._contact_age_norm(H2)
            f11 = self._advertisement_age_norm(H2)
        vec = np.array(
            [
                qnorm[0],
                qnorm[1],
                qnorm[2],
                c2,
                c3,
                float(np.clip(slack, -1, 2)),
                stype,
                cycles_n,
                n_active,
                self.t / self.cfg["horizon_s"],
                f10,
                f11,
                self.n_remote / max(len(self.cfg["arrivals"]) * 2, 1),
                self.n_success / max(len(self.cfg["arrivals"]), 1),
                val_n,
                float(np.clip(n_elig, 0, 1)),
            ],
            dtype=np.float32,
        )
        return vec

    def _contact_age_norm(self, helper: int) -> float:
        if not self.contact[ORIGIN, helper]:
            return 0.0
        age = self.t - self.last_contact_change_t[helper]
        return float(np.clip(age / self.cfg["horizon_s"], 0.0, 1.0))

    def _advertisement_age_norm(self, helper: int) -> float:
        ttl = float(self.cfg["advertisement_ttl_s"])
        last = self.last_adv_rx_t[helper]
        if not np.isfinite(last) or last < 0:
            return 1.0  # no advertisement received → aged out
        age = self.t - last
        return float(np.clip(age / max(ttl, 1e-9), 0.0, 1.0))

    def contact_age(self, helper: int) -> float:
        """Seconds since last contact edge change for helper (0 if currently down)."""
        if not self.contact[ORIGIN, helper]:
            return 0.0
        return max(0.0, self.t - self.last_contact_change_t[helper])

    def _queued_cycles(self, v: int) -> float:
        s = self.busy_remain[v]
        for j in self.queues[v]:
            s += j.remain
        return s

    def _enqueue(self, v: int, job: Job):
        if self.busy_job[v] is None:
            self.busy_job[v] = job
            self.busy_remain[v] = job.remain
        else:
            self.queues[v].append(job)

    def _update_contact(self):
        # Private schedule drives ground truth only — never exposed in causal obs/baselines
        up = not (self.cfg["contact_exit"] <= self.t < self.cfg["contact_return"])
        was = bool(self.contact[ORIGIN, H2])
        self.contact[ORIGIN, H2] = up
        self.contact[H2, ORIGIN] = up
        self.contact[ORIGIN, H3] = True
        self.contact[H3, ORIGIN] = True
        self.contact[H2, H3] = True
        self.contact[H3, H2] = True
        if was != up:
            self.last_contact_change_t[H2] = self.t
        # Ego receives helper advertisements only while in contact (causal)
        self._poll_advertisements()
        if was and not up:
            self._strand_helper(H2)
            if self.cfg.get("radio_mode") == "mode2_event":
                self._strand_mode2_helper(H2)

    def _poll_advertisements(self):
        period = float(self.cfg["advertisement_period_s"])
        while self._next_adv_t <= self.t + 1e-12:
            for h in (H2, H3):
                if self.contact[ORIGIN, h]:
                    self.last_adv_rx_t[h] = self._next_adv_t
            self._next_adv_t += period

    def _strand_mode2_helper(self, helper: int):
        for f in self.radio.flows:
            if f.helper == helper and f.phase not in ("DONE", "FAIL"):
                f.phase = "FAIL"
                wf = self.wfs.get(f.wf)
                if wf and wf.status == "ACTIVE":
                    st = wf.stages.get(f.sid)
                    if st and st.status == "RUNNING":
                        st.status = "WAITING"
                        st.place = -1
                        st.ban_helpers = tuple(set(st.ban_helpers) | {helper})
                        st.blocked_until = max(st.blocked_until, self.cfg["contact_return"])
        # drop TX toward/from helper
        for v in range(N_VEH):
            self.radio.tx_q[v] = [
                fr for fr in self.radio.tx_q[v]
                if fr.dst != helper and fr.src != helper
            ]

    def _strand_helper(self, v: int):
        """Abort remote jobs on a helper that left contact (re-eligible if wf alive)."""
        jobs = []
        if self.busy_job[v] is not None:
            jobs.append(self.busy_job[v])
            self.busy_job[v] = None
            self.busy_remain[v] = 0.0
        jobs.extend(self.queues[v])
        self.queues[v] = []
        # drop pending TX to this helper
        keep = []
        for te, wf_id, sid, place in self.tx_events:
            if place == v and not str(sid).startswith("HOLD_"):
                wf = self.wfs.get(wf_id)
                if wf and wf.status == "ACTIVE":
                    st = wf.stages.get(sid)
                    if st and st.status == "RUNNING":
                        st.status = "WAITING"
                        st.place = -1
                continue
            keep.append((te, wf_id, sid, place))
        self.tx_events = keep
        for j in jobs:
            if j.wf < 0:
                continue
            wf = self.wfs.get(j.wf)
            if wf is None or wf.status != "ACTIVE":
                continue
            st = wf.stages.get(j.sid)
            if st is None:
                continue
            st.status = "WAITING"
            st.place = -1
            st.remain = st.cycles
            st.ban_helpers = tuple(set(st.ban_helpers) | {v})
            # cannot retry until helper could return (mobility recovery lag)
            st.blocked_until = max(st.blocked_until, self.cfg["contact_return"])
        self._refresh_eligible()

    def _ingest_arrivals(self):
        while self.pending_arrivals and self.pending_arrivals[0] <= self.t + 1e-12:
            rel = self.pending_arrivals.pop(0)
            self._spawn_workflow(rel)

    def _spawn_workflow(self, release: float):
        wid = self.next_wf
        self.next_wf += 1
        if self.cfg.get("phased_values", False):
            p = (
                self.cfg["p_high_early"]
                if release < self.cfg["phase_split_s"]
                else self.cfg["p_high_late"]
            )
        else:
            p = self.cfg["p_high_value"]
        high = self.rng.random() < p
        value = self.cfg["value_high"] if high else self.cfg["value_low"]
        stages = {
            s: Stage(wid, s, STAGE_CYCLES[s], "WAITING", -1, STAGE_CYCLES[s])
            for s in ("A", "B", "C", "D")
        }
        self.wfs[wid] = Workflow(
            wid, release, release + self.cfg["deadline_s"], value, "ACTIVE", stages
        )

    def _any_high_pending(self) -> bool:
        vh = self.cfg["value_high"]
        for w in self.wfs.values():
            if w.status == "ACTIVE" and w.value >= vh - 1e-9:
                return True
        return False

    def _success_reward(self, wf: Workflow) -> float:
        r = float(wf.value)
        if self.cfg.get("reward_mode") != "utility_focus":
            return r
        # Emphasize high-value; penalize low-value completions that crowd highs
        if wf.value >= self.cfg["value_high"] - 1e-9:
            r += 6.0
        elif self._any_high_pending():
            r -= 4.0
        return r

    def _refresh_eligible(self):
        for wf in self.wfs.values():
            if wf.status != "ACTIVE":
                continue
            for sid, st in wf.stages.items():
                if st.status != "WAITING":
                    continue
                if self.t < st.blocked_until - 1e-12:
                    continue
                preds = PRED[sid]
                if all(wf.stages[p].status == "DONE" for p in preds):
                    st.status = "ELIGIBLE"

    def _next_eligible(self) -> Optional[Tuple[int, str]]:
        # Soft-EDF with defer_count so HOLD can skip to another eligible stage
        order = {"A": 0, "B": 1, "C": 2, "D": 3}
        best = None
        for wf in self.wfs.values():
            if wf.status != "ACTIVE":
                continue
            for sid, st in wf.stages.items():
                if st.status != "ELIGIBLE":
                    continue
                key = (st.defer_count, wf.deadline, order[sid], wf.wf)
                if best is None or key < best[0]:
                    best = (key, wf.wf, sid)
        if best is None:
            return None
        return best[1], best[2]

    def _apply_action(self, wf_id: int, sid: str, action: int) -> float:
        mask = self.action_mask()
        if action < 0 or action >= N_ACTIONS or not mask[action]:
            # illegal → local fallback + penalty
            action = A_LOCAL if mask[A_LOCAL] else int(np.argmax(mask))
            penalty = -0.05
        else:
            penalty = 0.0
        wf = self.wfs[wf_id]
        st = wf.stages[sid]
        if action == A_DROP:
            # Admission reject / abort workflow (frees scarce capacity)
            wf.status = "EXPIRED"
            self.n_expired += 1
            for s in wf.stages.values():
                if s.status != "DONE":
                    s.status = "DONE"
            if self.cfg.get("reward_mode") == "utility_focus":
                if wf.value >= self.cfg["value_high"] - 1e-9:
                    return penalty - 4.0
                # Earlier admission control (PDP teacher gap vs ValueGreedy):
                # 1) drop low while high already pending
                # 2) drop low in the early phase to reserve capacity for the late high burst
                if self._any_high_pending():
                    return penalty + 1.5
                if self.cfg.get("phased_values") and self.t < float(self.cfg.get("phase_split_s", 0)):
                    return penalty + 1.2
                return penalty + 0.2
            return penalty - 0.05 * (wf.value / self.cfg["value_high"])
        if action == A_HOLD:
            # Deprioritize this eligible stage so another can be chosen (value-aware skip)
            st.defer_count += 1
            if st.defer_count > 6:
                action = A_LOCAL
            else:
                return penalty - 0.002
        place = {A_LOCAL: ORIGIN, A_H2: H2, A_H3: H3}[action]
        if place != ORIGIN and not self.contact[ORIGIN, place]:
            place = ORIGIN
            penalty -= 0.1
        st.status = "RUNNING"
        st.place = place
        if place != ORIGIN:
            self.n_remote += 1
            if self.cfg.get("radio_mode") == "mode2_event":
                # COMMIT_REQ → ACK → input → CPU → output (no compute before input delivery)
                self.radio.start_remote(self.t, wf_id, sid, place, st.cycles, ORIGIN)
            else:
                jitter = float(self.rng.uniform(0.85, 1.35)) if self.cfg.get("link_jitter", False) else 1.0
                t_ready = self.t + jitter * (self.cfg["link_delay_s"] + self.cfg["ctrl_delay_s"])
                self.tx_events.append((t_ready, wf_id, sid, place))
        else:
            self._enqueue(place, Job(place, wf_id, sid, st.cycles))
        return penalty

    def _on_job_done(self, job: Job):
        if job.wf < 0:  # background
            return
        wf = self.wfs.get(job.wf)
        if wf is None or wf.status != "ACTIVE":
            return
        st = wf.stages[job.sid]
        # Mode-2 remote: CPU finish only starts output TX; stage DONE on OUTPUT_READY
        if (
            self.cfg.get("radio_mode") == "mode2_event"
            and job.veh != ORIGIN
            and st.status == "RUNNING"
        ):
            self.radio.on_cpu_finished(self.t, job.wf, job.sid)
            return
        st.status = "DONE"
        st.finish_t = self.t
        self._refresh_eligible()
        if job.sid == "D":
            if self.t <= wf.deadline + 1e-9:
                wf.status = "SUCCESS"
                self.n_success += 1
                self.total_value += wf.value
            else:
                wf.status = "EXPIRED"
                self.n_expired += 1

    def _handle_mode2_events(self, events) -> float:
        reward = 0.0
        for kind, wf_id, sid, helper in events:
            wf = self.wfs.get(wf_id)
            if wf is None or wf.status != "ACTIVE":
                continue
            st = wf.stages.get(sid)
            if st is None:
                continue
            if kind == "INPUT_READY":
                # only now may remote CPU start
                self._enqueue(helper, Job(helper, wf_id, sid, st.cycles))
            elif kind == "OUTPUT_READY":
                st.status = "DONE"
                st.finish_t = self.t
                self._refresh_eligible()
                if sid == "D":
                    if self.t <= wf.deadline + 1e-9:
                        wf.status = "SUCCESS"
                        self.n_success += 1
                        self.total_value += wf.value
                        reward += self._success_reward(wf)
                    else:
                        wf.status = "EXPIRED"
                        self.n_expired += 1
            elif kind == "FAIL":
                if st.status == "RUNNING":
                    st.status = "WAITING"
                    st.place = -1
                    st.ban_helpers = tuple(set(st.ban_helpers) | {helper})
                    st.blocked_until = max(st.blocked_until, self.cfg["contact_return"])
                self._refresh_eligible()
        return reward

    def _expire_check(self) -> float:
        r = 0.0
        for wf in self.wfs.values():
            if wf.status != "ACTIVE":
                continue
            if self.t > wf.deadline + 1e-9:
                wf.status = "EXPIRED"
                self.n_expired += 1
                if self.cfg.get("reward_mode") == "utility_focus":
                    if wf.value >= self.cfg["value_high"] - 1e-9:
                        r -= 10.0
                    else:
                        r -= 0.5
                else:
                    r -= 1.0
                # cancel eligible/running stages soft
                for st in wf.stages.values():
                    if st.status in ("ELIGIBLE", "WAITING", "RUNNING"):
                        st.status = "DONE"  # dead
        return r

    def _expire_all_remaining(self):
        for wf in self.wfs.values():
            if wf.status == "ACTIVE":
                wf.status = "EXPIRED"
                self.n_expired += 1

    def _service(self, dt: float):
        if dt <= 0:
            return
        # work each CPU
        for v in range(N_VEH):
            left = dt
            while left > 1e-12:
                if self.busy_job[v] is None:
                    if self.queues[v]:
                        j = self.queues[v].pop(0)
                        self.busy_job[v] = j
                        self.busy_remain[v] = j.remain
                    else:
                        break
                hz = self.cfg["cpu_hz"][v]
                need = self.busy_remain[v] / hz
                if need <= left + 1e-15:
                    left -= need
                    j = self.busy_job[v]
                    self.busy_job[v] = None
                    self.busy_remain[v] = 0.0
                    self.t_service_end = self.t + (dt - left)  # not used
                    # completion at current time + consumed
                    # We'll set absolute time after outer advance; mark with remain 0
                    j.remain = 0.0
                    j.t_done = self.t + (dt - left)
                    self._pending_completes.append(j)
                else:
                    self.busy_remain[v] -= left * hz
                    self.busy_job[v].remain = self.busy_remain[v]
                    left = 0.0

    def _ensure_decision_or_advance(self) -> Tuple[np.ndarray, float]:
        reward = 0.0
        self._pending_completes: List[Job] = []
        while True:
            if self.t >= self.cfg["horizon_s"] - 1e-12:
                self._expire_all_remaining()
                self.done = True
                return self._obs_idle(), reward
            self._ingest_arrivals()
            self._update_contact()
            self._refresh_eligible()
            # deliver tx events due
            self.tx_events.sort(key=lambda x: x[0])
            while self.tx_events and self.tx_events[0][0] <= self.t + 1e-12:
                te, wf_id, sid, place = self.tx_events.pop(0)
                if sid.startswith("HOLD_"):
                    real = sid[5:]
                    wf = self.wfs.get(wf_id)
                    if wf and wf.status == "ACTIVE" and wf.stages[real].status == "WAITING":
                        # only if preds still satisfied
                        pass
                    self._refresh_eligible()
                    continue
                wf = self.wfs.get(wf_id)
                if wf is None or wf.status != "ACTIVE":
                    continue
                st = wf.stages[sid]
                if st.status != "RUNNING":
                    continue
                self._enqueue(place, Job(place, wf_id, sid, st.cycles))

            elig = self._next_eligible()
            if elig is not None:
                self.pending_decision = elig
                return self._obs(), reward

            # no decision: advance to next event
            next_t = self.cfg["horizon_s"]
            if self.pending_arrivals:
                next_t = min(next_t, self.pending_arrivals[0])
            if self.tx_events:
                next_t = min(next_t, self.tx_events[0][0])
            if self.cfg.get("radio_mode") == "mode2_event":
                rt = self.radio.next_event_time(self.t)
                if rt is not None:
                    next_t = min(next_t, rt)
                if self._next_adv_t > self.t + 1e-12:
                    next_t = min(next_t, self._next_adv_t)
            # next CPU completion
            for v in range(N_VEH):
                if self.busy_job[v] is not None:
                    next_t = min(next_t, self.t + self.busy_remain[v] / self.cfg["cpu_hz"][v])
            # contact change points (private schedule — ground truth only)
            for tc in (self.cfg["contact_exit"], self.cfg["contact_return"]):
                if tc > self.t + 1e-12:
                    next_t = min(next_t, tc)
            for wf in self.wfs.values():
                if wf.status != "ACTIVE":
                    continue
                for st in wf.stages.values():
                    if st.status == "WAITING" and st.blocked_until > self.t + 1e-12:
                        next_t = min(next_t, st.blocked_until)
            if next_t <= self.t + 1e-15:
                next_t = self.t + 1e-3
            dt = next_t - self.t
            self._pending_completes = []
            self._service(dt)
            t0 = self.t
            self.t = next_t
            if self.cfg.get("radio_mode") == "mode2_event":
                evs = self.radio.advance(t0, self.t, self.contact, self.rng)
                reward += self._handle_mode2_events(evs)
            # apply completes in order
            self._pending_completes.sort(key=lambda j: j.t_done)
            for j in self._pending_completes:
                prev_succ = self.n_success
                self._on_job_done(j)
                if j.wf >= 0 and self.wfs[j.wf].status == "SUCCESS" and self.n_success > prev_succ:
                    reward += self._success_reward(self.wfs[j.wf])
            reward += self._expire_check()
            # if all workflows terminal and no arrivals
            if not self.pending_arrivals and all(
                w.status != "ACTIVE" for w in self.wfs.values()
            ):
                self.done = True
                return self._obs_idle(), reward


# ----- baselines -----
def policy_b0(obs, mask, env: WorkflowMDP) -> int:
    return A_LOCAL if mask[A_LOCAL] else int(np.argmax(mask))


def policy_b2(obs, mask, env: WorkflowMDP) -> int:
    """Shortest expected finish among reachable (queue-aware)."""
    if env.pending_decision is None:
        return A_LOCAL
    wf_id, sid = env.pending_decision
    if sid in ("A", "D") or not mask[A_H2] and not mask[A_H3]:
        return A_LOCAL
    cycles = STAGE_CYCLES[sid]
    cands = [(A_LOCAL, ORIGIN)]
    if mask[A_H2]:
        cands.append((A_H2, H2))
    if mask[A_H3]:
        cands.append((A_H3, H3))
    best_a, best_t = A_LOCAL, 1e9
    for a, v in cands:
        q = env._queued_cycles(v)
        t = q / env.cfg["cpu_hz"][v] + cycles / env.cfg["cpu_hz"][v]
        if v != ORIGIN:
            t += env.cfg["link_delay_s"] + env.cfg["ctrl_delay_s"]
        if t < best_t:
            best_t, best_a = t, a
    return best_a


def policy_b3(obs, mask, env: WorkflowMDP) -> int:
    """Mobility-aware using causal contact/advertisement only (no future schedule)."""
    if env.pending_decision is None:
        return A_LOCAL
    _, sid = env.pending_decision
    if sid in ("A", "D"):
        return A_LOCAL
    mask2 = mask.copy()
    if mask2[A_H2]:
        # Young contact or stale advertisement → prefer stable H3
        age = env.contact_age(H2)
        adv = env._advertisement_age_norm(H2)
        if age < 0.05 or adv >= 0.95:
            mask2[A_H2] = False
    return policy_b2(obs, mask2, env)


def policy_b4(obs, mask, env: WorkflowMDP) -> int:
    """Joint heuristic: put B on H3, C on H2 (if up) else both on best."""
    if env.pending_decision is None:
        return A_LOCAL
    _, sid = env.pending_decision
    if sid == "B" and mask[A_H3]:
        return A_H3
    if sid == "C" and mask[A_H2]:
        return A_H2
    if sid == "C" and mask[A_H3]:
        return A_H3
    return policy_b3(obs, mask, env)


def policy_value_greedy(obs, mask, env: WorkflowMDP) -> int:
    """Drop low-value when overloaded; protect helpers for high-value."""
    if env.pending_decision is None:
        return A_LOCAL
    wf_id, sid = env.pending_decision
    wf = env.wfs[wf_id]
    n_active = sum(1 for w in env.wfs.values() if w.status == "ACTIVE")
    # Overloaded + low value → drop (admission control)
    if wf.value < env.cfg["value_high"] - 1e-9 and n_active >= 4 and mask[A_DROP]:
        return A_DROP
    if sid in ("A", "D"):
        return A_LOCAL
    return policy_b3(obs, mask, env)


def policy_always_drop_low(obs, mask, env: WorkflowMDP) -> int:
    """Paper B2: Always-Drop-Low (admission competitor + BC teacher)."""
    if env.pending_decision is None:
        return A_LOCAL
    wf_id, sid = env.pending_decision
    wf = env.wfs[wf_id]
    if wf.value < env.cfg["value_high"] - 1e-9 and mask[A_DROP]:
        return A_DROP
    if sid in ("A", "D"):
        return A_LOCAL
    return policy_b3(obs, mask, env)


def policy_local_edf(obs, mask, env: WorkflowMDP) -> int:
    """Paper B0: Local EDF — no cooperation; LOCAL if legal else mask fallback."""
    if mask[A_LOCAL]:
        return A_LOCAL
    if mask[A_DROP]:
        return A_DROP
    return int(np.argmax(mask))


def _obs_sid(stype: float) -> str:
    """Invert causal obs stage-type encoding {A:0, B:0.33, C:0.66, D:1}."""
    opts = [(0.0, "A"), (0.33, "B"), (0.66, "C"), (1.0, "D")]
    return min(opts, key=lambda x: abs(x[0] - float(stype)))[1]


def policy_priority_edf_eft(obs, mask, env: WorkflowMDP) -> int:
    """Paper B3: Priority-EDF–EFT (adapted), reduced 16-D causal rule.

    Uses only the DDQN observation vector, legal mask, public cfg, and known
    diamond-4 structure. Does NOT read true helper queues or future contact.

    Admission (16-D feasible form of priority): reject low-value when DROP legal.
    Placement: earliest-finish among LOCAL/H2/H3 from observable queue norms,
    contact bits, and advertisement age; remaining-DAG estimate for B‖C→D.
    """
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    qnorm = obs[0:3]
    c2, c3 = float(obs[3]), float(obs[4])
    slack = float(obs[5])
    stype = float(obs[6])
    f10 = float(obs[10])
    f11 = float(obs[11])
    val_n = float(obs[14])
    n_elig = float(obs[15])

    sid = _obs_sid(stype)
    deadline_s = float(env.cfg["deadline_s"])
    cpu = np.asarray(env.cfg["cpu_hz"], dtype=np.float64)
    link = float(env.cfg.get("link_delay_s", 0.024))
    ctrl = float(env.cfg.get("ctrl_delay_s", 0.0))
    rem_deadline = slack * deadline_s
    high = val_n >= 0.85

    # Priority admission (no n_active>=4): free capacity for high-value work
    if (not high) and mask[A_DROP]:
        return A_DROP

    def queue_wait(v: int) -> float:
        return float(np.clip(qnorm[v], 0.0, 5.0)) * deadline_s

    def comm_in(v: int) -> float:
        return 0.0 if v == ORIGIN else (link + ctrl)

    def dest_legal(a: int) -> bool:
        if a == A_LOCAL:
            return bool(mask[A_LOCAL])
        if a == A_H2:
            return bool(mask[A_H2]) and c2 >= 0.5 and f11 < 0.99
        if a == A_H3:
            return bool(mask[A_H3]) and c3 >= 0.5
        return False

    def stage_finish(stage_id: str, a: int) -> Tuple[float, float]:
        v = {A_LOCAL: ORIGIN, A_H2: H2, A_H3: H3}[a]
        if stage_id in ("A", "D"):
            v, a = ORIGIN, A_LOCAL
        fin = comm_in(v) + queue_wait(v) + STAGE_CYCLES[stage_id] / max(cpu[v], 1.0)
        if a == A_H2 and f11 > 0.85:
            fin += 0.06
        # Parallel-branch aware EFT: separate B/C across helpers when both reachable
        # (DAG sync at D — distinct from Always-Drop-Low's mobility shortest-queue).
        if stage_id == "B" and a == A_H3:
            fin -= 0.015
        if stage_id == "C" and a == A_H2 and f11 < 0.85:
            fin -= 0.015
        return fin, comm_in(v)

    def remaining_after(cur_sid: str, cur_fin: float) -> float:
        def proc(stage_id: str, v: int) -> float:
            return STAGE_CYCLES[stage_id] / max(cpu[v], 1.0)

        # Fixed DAG residual using best observable dest compute rates (no re-queue stack)
        if cur_sid == "D":
            return cur_fin
        # B,C prefer fastest reachable helper; D local
        h2_ok = c2 >= 0.5 and f11 < 0.95
        h3_ok = c3 >= 0.5
        best_hz = cpu[ORIGIN]
        if h2_ok:
            best_hz = max(best_hz, cpu[H2])
        if h3_ok:
            best_hz = max(best_hz, cpu[H3])
        t_bc = max(STAGE_CYCLES["B"], STAGE_CYCLES["C"]) / best_hz + link
        t_d = STAGE_CYCLES["D"] / cpu[ORIGIN]
        if cur_sid == "A":
            return cur_fin + t_bc + t_d
        if cur_sid in ("B", "C"):
            return cur_fin + 0.5 * t_bc + t_d
        return cur_fin

    if sid in ("A", "D"):
        if mask[A_LOCAL]:
            return A_LOCAL
        if mask[A_DROP]:
            return A_DROP
        return int(np.argmax(mask))

    cands = []
    for a in (A_LOCAL, A_H2, A_H3):
        if not dest_legal(a):
            continue
        fin, cc = stage_finish(sid, a)
        wf_fin = remaining_after(sid, fin)
        cands.append((a, fin, wf_fin, cc))

    if not cands:
        if mask[A_HOLD] and n_elig > 0.15:
            return A_HOLD
        if mask[A_LOCAL]:
            return A_LOCAL
        return int(np.argmax(mask))

    feasible = [c for c in cands if c[2] <= rem_deadline + 1e-9]
    pool = feasible if feasible else cands
    pool.sort(key=lambda x: (x[2], x[3], x[0]))
    return int(pool[0][0])
