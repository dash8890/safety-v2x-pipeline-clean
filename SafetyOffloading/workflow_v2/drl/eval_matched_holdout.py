"""Matched holdout comparison: frozen DDQN vs ValueGreedy (and B0–B4). No retraining."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ddqn_agent import DDQNAgent
from run_drl_experiment import eval_policy, make_agent_policy
from workflow_mdp import (
    N_ACTIONS,
    WorkflowMDP,
    policy_b0,
    policy_b2,
    policy_b3,
    policy_b4,
    policy_value_greedy,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
HOLDOUT = list(range(300, 340))


def main():
    ckpt = OUT / "ddqn_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = WorkflowMDP(seed=0)
    agent = DDQNAgent(env.obs_dim, N_ACTIONS, device=device)
    agent.load(str(ckpt))
    agent.eps = 0.0

    rows = [
        eval_policy("B0_local", policy_b0, HOLDOUT),
        eval_policy("B2_queue", policy_b2, HOLDOUT),
        eval_policy("B3_mobility", policy_b3, HOLDOUT),
        eval_policy("B4_joint", policy_b4, HOLDOUT),
        eval_policy("ValueGreedy", policy_value_greedy, HOLDOUT),
        eval_policy("DDQN_frozen", make_agent_policy(agent), HOLDOUT),
    ]

    print("MATCHED HOLDOUT seeds 300-339")
    for r in rows:
        print(
            f"{r['policy']:14s} V={r['mean_value']:.4f}+/-{r['std_value']:.4f}  "
            f"N={r['mean_success']:.4f}+/-{r['std_success']:.4f}"
        )

    vg = next(r for r in rows if r["policy"] == "ValueGreedy")
    dd = next(r for r in rows if r["policy"] == "DDQN_frozen")
    dV = dd["mean_value"] - vg["mean_value"]
    dN = dd["mean_success"] - vg["mean_success"]
    relV = dV / max(vg["mean_value"], 1e-9)
    relN = dN / max(vg["mean_success"], 1e-9)
    v_dd = np.array(dd["per_seed_value"], dtype=float)
    v_vg = np.array(vg["per_seed_value"], dtype=float)
    n_dd = np.array(dd["per_seed_success"], dtype=float)
    n_vg = np.array(vg["per_seed_success"], dtype=float)

    print(f"DELTA DDQN-VG value {dV:.4f} rel {100*relV:.2f}%")
    print(f"DELTA DDQN-VG n_success {dN:.4f} rel {100*relN:.2f}%")
    print(
        "paired value wins/ties/losses",
        int((v_dd > v_vg).sum()),
        int((v_dd == v_vg).sum()),
        int((v_dd < v_vg).sum()),
    )
    print(
        "paired n wins/ties/losses",
        int((n_dd > n_vg).sum()),
        int((n_dd == n_vg).sum()),
        int((n_dd < n_vg).sum()),
    )

    out = {
        "holdout_seeds": HOLDOUT,
        "checkpoint": str(ckpt.resolve()),
        "frozen": True,
        "no_retrain": True,
        "policies": rows,
        "ddqn_vs_valuegreedy": {
            "delta_mean_value": float(dV),
            "relative_value_pct": float(100 * relV),
            "delta_mean_success": float(dN),
            "relative_success_pct": float(100 * relN),
            "paired_value_wins_ties_losses": [
                int((v_dd > v_vg).sum()),
                int((v_dd == v_vg).sum()),
                int((v_dd < v_vg).sum()),
            ],
            "paired_success_wins_ties_losses": [
                int((n_dd > n_vg).sum()),
                int((n_dd == n_vg).sum()),
                int((n_dd < n_vg).sum()),
            ],
            "mean_paired_delta_value": float((v_dd - v_vg).mean()),
            "std_paired_delta_value": float((v_dd - v_vg).std()),
        },
        "claim_scope": {
            "substantial_vs_value_blind_B0_B4": True,
            "substantial_utility_vs_ValueGreedy": bool(relV >= 0.10 and dV >= 2.0),
            "completion_advantage_vs_ValueGreedy": bool(dN > 0),
        },
    }
    path = OUT / "r4_matched_holdout.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"WROTE {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
