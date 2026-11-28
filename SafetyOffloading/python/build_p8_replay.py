#!/usr/bin/env python3
"""Build offline P8 replay from joint schedule_epochs + assignment_features + rewards.

Uses P6/P7 worlds on train seeds only. Does not touch P10 held-out seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ASSIGN_FEATS = [
    "is_local",
    "processor_cpu_ghz",
    "estimated_input_latency_ms",
    "estimated_result_latency_ms",
    "estimated_helper_queueing_ms",
    "estimated_computation_latency_ms",
    "predicted_deadline_slack_ms",
    "predicted_completion_probability",
    "predicted_conditional_safety_risk",
    "relative_distance_m",
    "estimated_resource_block_slots",
]
CTX = [
    "pending_tasks",
    "candidate_assignments",
    "committed_offloads",
    "mean_p_hat",
    "mean_risk",
    "mean_slack",
]


def num(s, default=0.0):
    try:
        v = float(s)
        return v if v == v else default
    except Exception:
        return default


def load_world(so: Path, configuration_id: int, seed: int, policy: str, partition: str):
    ep = pd.read_csv(so / "schedule_epochs.csv")
    af = pd.read_csv(so / "assignment_features.csv")
    rew = (
        pd.read_csv(so / "rewards.csv")
        if (so / "rewards.csv").is_file()
        else pd.DataFrame(columns=["task_id", "reward"])
    )
    reward_by_task = {
        int(r.task_id): float(r.reward)
        for r in rew.itertuples(index=False)
        if pd.notna(getattr(r, "reward", None))
    }
    rows = []
    committed = 0
    ep = ep.sort_values("decision_id")
    af_by_dec = {int(d): g for d, g in af.groupby("decision_id")}
    decisions = ep.to_dict("records")
    for i, r in enumerate(decisions):
        did = int(r["decision_id"])
        atype = str(r["action_type"]).upper()
        tid = int(num(r.get("task_id", 0)))
        pid = int(num(r.get("processor_id", 0)))
        feats = af_by_dec.get(did, pd.DataFrame())
        unmasked = feats[feats["masked"].fillna(0).astype(int) == 0] if len(feats) else feats
        mean_p = float(unmasked["predicted_completion_probability"].mean()) if len(unmasked) else 0.0
        mean_r = float(unmasked["predicted_conditional_safety_risk"].mean()) if len(unmasked) else 0.0
        mean_s = float(unmasked["predicted_deadline_slack_ms"].mean()) if len(unmasked) else 0.0
        ctx = {
            "pending_tasks": num(r.get("pending_tasks", 0)),
            "candidate_assignments": num(r.get("candidate_assignments", 0)),
            "committed_offloads": committed,
            "mean_p_hat": mean_p,
            "mean_risk": mean_r,
            "mean_slack": mean_s,
        }
        # Chosen assignment features (zeros for REJECT/STOP).
        chosen = {f: 0.0 for f in ASSIGN_FEATS}
        if atype == "ASSIGN" and len(feats):
            hit = feats[(feats["task_id"] == tid) & (feats["processor_id"] == pid)]
            if len(hit):
                h = hit.iloc[0]
                for f in ASSIGN_FEATS:
                    chosen[f] = num(h.get(f, 0))
                if not bool(int(num(h.get("is_local", 0)))):
                    committed += 1
            else:
                # ASSIGN without feature row — still count remote heuristically.
                if pid > 0:
                    committed += 1
        elif atype == "STOP":
            committed = 0  # epoch boundary resets tentative commitments in logs
        reward = 0.0
        if atype == "ASSIGN" and tid in reward_by_task:
            reward = reward_by_task[tid]
        nxt = decisions[i + 1] if i + 1 < len(decisions) else None
        done = 1.0 if atype == "STOP" or nxt is None else 0.0
        row = {
            "configuration_id": configuration_id,
            "seed": seed,
            "policy": policy,
            "partition": partition,
            "decision_id": did,
            "action_type": atype,
            "task_id": tid,
            "processor_id": pid,
            "reward": reward,
            "done": done,
            **{f"ctx_{k}": ctx[k] for k in CTX},
            **{f"a_{k}": chosen[k] for k in ASSIGN_FEATS},
        }
        if nxt is not None:
            row["next_decision_id"] = int(nxt["decision_id"])
            row["next_pending"] = num(nxt.get("pending_tasks", 0))
            row["next_candidates"] = num(nxt.get("candidate_assignments", 0))
        else:
            row["next_decision_id"] = -1
            row["next_pending"] = 0
            row["next_candidates"] = 0
        rows.append(row)
        # Candidate bag for masking diagnostics (count only).
        row["n_unmasked_candidates"] = int(len(unmasked))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p6-root", type=Path, default=Path("outputs/p6_myopic_eval"))
    ap.add_argument("--p7-root", type=Path, default=Path("outputs/p7_calibrated_myopic_eval"))
    ap.add_argument("--out", type=Path, default=Path("outputs/p8_replay"))
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--configs", type=int, nargs="+", default=[1, 5, 6, 22, 31])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    sources = []
    for root, pol in [(args.p6_root, "risk_myopic_p6"), (args.p7_root, "risk_myopic_p7")]:
        if not root.is_dir():
            continue
        for cfg in args.configs:
            for seed in args.train_seeds:
                so = root / f"cfg{cfg:02d}_seed{seed}" / pol / "safety_offloading"
                if not (so / "schedule_epochs.csv").is_file():
                    continue
                part = "train"
                rows = load_world(so, cfg, seed, pol, part)
                all_rows.extend(rows)
                sources.append({"root": str(root), "cfg": cfg, "seed": seed, "policy": pol, "n": len(rows)})
    df = pd.DataFrame(all_rows)
    out_csv = args.out / "replay.csv"
    df.to_csv(out_csv, index=False)
    manifest = {
        "rows": int(len(df)),
        "assign": int((df.action_type == "ASSIGN").sum()) if len(df) else 0,
        "reject": int((df.action_type == "REJECT").sum()) if len(df) else 0,
        "stop": int((df.action_type == "STOP").sum()) if len(df) else 0,
        "train_seeds": args.train_seeds,
        "configs": args.configs,
        "sources": sources,
        "ctx_features": CTX,
        "assignment_features": ASSIGN_FEATS,
        "note": "Offline replay for masked DDQN / value-assist; rewards from task completions only; no policy harm broadcast.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "assign", "reject", "stop")}, indent=2))


if __name__ == "__main__":
    main()
