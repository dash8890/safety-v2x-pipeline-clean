#!/usr/bin/env python3
"""Main baseline comparison per docs/BASELINE_ABLATION_SPEC.md (LOCKED).

B0 Local EDF | B1 ValueGreedy | B2 Always-Drop-Low | B3 Priority-EDF-EFT (adapted) | P Full DDQN

Prerequisite gate: VG≈78.40, Always-Drop-Low≈99.30, Full≈101.80 on seeds 300–339.
Logs obs+mask at decisions for B3 (no privileged-state leak check).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ddqn_agent import DDQNAgent
from run_drl_experiment import make_agent_policy
from workflow_mdp import (
    N_ACTIONS,
    WorkflowMDP,
    policy_always_drop_low,
    policy_local_edf,
    policy_priority_edf_eft,
    policy_value_greedy,
)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
REPORT = OUT / "main_baseline_holdout.json"
CKPT = OUT / "ddqn_causal_dueling.pt"
HOLDOUT = list(range(300, 340))

CFG = {
    "obs_mode": "causal",
    "radio_mode": "abstract",
    "reward_mode": "utility_focus",
    "phased_values": True,
    "phase_split_s": 0.16,
    "p_high_early": 0.05,
    "p_high_late": 0.75,
    "value_high": 12.0,
    "value_low": 1.0,
    "cpu_hz": np.array([2.8e8, 4.0e9, 0.8e9], dtype=np.float64),
    "deadline_s": 0.35,
    "bg_cycles_origin": 20e6,
    "contact_exit": 0.10,
    "contact_return": 0.62,
}


def make_env(seed=0):
    return WorkflowMDP(seed=seed, cfg=dict(CFG))


def eval_pol(name, fn, seeds, log_decisions: bool = False):
    rows = []
    decision_log = []
    t0 = time.time()
    for s in seeds:
        env = make_env(s)
        obs = env.reset(s)
        done = False
        n_dec = 0
        while not done:
            mask = env.action_mask()
            if log_decisions and n_dec < 5:
                decision_log.append(
                    {
                        "seed": s,
                        "t": float(env.t),
                        "obs": np.asarray(obs, dtype=np.float32).tolist(),
                        "mask": np.asarray(mask, dtype=bool).tolist(),
                    }
                )
            a = fn(obs, mask, env)
            obs, _, done, info = env.step(a)
            n_dec += 1
        rows.append(info)
    v = np.array([r["total_value"] for r in rows], float)
    n = np.array([r["n_success"] for r in rows], float)
    out = {
        "policy": name,
        "mean_value": float(v.mean()),
        "std_value": float(v.std()),
        "mean_success": float(n.mean()),
        "std_success": float(n.std()),
        "per_seed_value": v.tolist(),
        "per_seed_success": n.tolist(),
        "runtime_s": time.time() - t0,
    }
    if log_decisions:
        out["sample_decisions"] = decision_log
    return out


def paired(a, b):
    va = np.array(a["per_seed_value"])
    vb = np.array(b["per_seed_value"])
    d = va - vb
    return {
        "delta_mean": float(d.mean()),
        "delta_std": float(d.std()),
        "wins": int((d > 0).sum()),
        "ties": int((d == 0).sum()),
        "losses": int((d < 0).sum()),
    }


def main():
    print("Reproducing gate targets VG=78.40 ADL=99.30 Full=101.80", flush=True)
    b0 = eval_pol("B0_LocalEDF", policy_local_edf, HOLDOUT)
    b1 = eval_pol("B1_ValueGreedy", policy_value_greedy, HOLDOUT)
    b2 = eval_pol("B2_AlwaysDropLow", policy_always_drop_low, HOLDOUT)
    print(
        f"GATE B1={b1['mean_value']:.2f} B2={b2['mean_value']:.2f} "
        f"(expect 78.40 / 99.30)",
        flush=True,
    )
    assert abs(b1["mean_value"] - 78.4) < 0.05, b1["mean_value"]
    assert abs(b2["mean_value"] - 99.3) < 0.05, b2["mean_value"]

    b3 = eval_pol("B3_PriorityEDF_EFT", policy_priority_edf_eft, HOLDOUT, log_decisions=True)
    print(f"B3 Priority-EDF-EFT mean={b3['mean_value']:.2f}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env0 = make_env(0)
    agent = DDQNAgent(env0.obs_dim, N_ACTIONS, device=device, dueling=True)
    agent.load(str(CKPT))
    agent.eps = 0.0
    p = eval_pol("P_CausalDuelingDDQN", make_agent_policy(agent), HOLDOUT)
    print(f"P Full DDQN mean={p['mean_value']:.2f} (expect 101.80)", flush=True)
    assert abs(p["mean_value"] - 101.8) < 0.05, p["mean_value"]

    table = [b0, b1, b2, b3, p]
    strongest = max(table[:-1], key=lambda x: x["mean_value"])
    report = {
        "spec": "docs/BASELINE_ABLATION_SPEC.md",
        "holdout": HOLDOUT,
        "cfg": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in CFG.items()},
        "main_table": {r["policy"]: r for r in table},
        "ordered_means": [(r["policy"], r["mean_value"]) for r in table],
        "primary_vs_strongest_baseline": {
            "baseline": strongest["policy"],
            "baseline_mean": strongest["mean_value"],
            "proposed_mean": p["mean_value"],
            "paired": paired(p, strongest),
        },
        "paired_vs_B1_ValueGreedy": paired(p, b1),
        "paired_vs_B2_AlwaysDropLow": paired(p, b2),
        "paired_vs_B3": paired(p, b3),
        "gate_reproduced": True,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ordered_means": report["ordered_means"], "vs_strongest": report["primary_vs_strongest_baseline"]}, indent=2), flush=True)
    print(f"WROTE {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
