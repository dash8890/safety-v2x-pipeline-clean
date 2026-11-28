#!/usr/bin/env python3
"""Evaluate the mandated multi-milestone, matched-scenario convergence evidence.

Supports 12-block (60-rollout) and 16-block (80-rollout) training runs.
Default milestones: 4,8,12.  Pass --milestones 4,8,12,16 for 16-block runs.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

DEFAULT_MILESTONES = "4,8,12"

def utility(path: Path) -> float:
    return float(json.loads(path.read_text())["total_value"])

def paired_change(before: np.ndarray, after: np.ndarray) -> dict:
    change = after - before
    mean = float(change.mean()); se = float(change.std(ddof=1) / math.sqrt(len(change)))
    lo, hi = mean - 12.706 * se, mean + 12.706 * se
    return {"per_scenario": change.tolist(), "mean": mean, "std": float(change.std(ddof=1)),
            "ci95": [lo, hi], "positive_scenarios": int(np.count_nonzero(change > 0))}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--milestones", type=str, default=DEFAULT_MILESTONES,
                   help="Comma-separated block numbers for milestone validation (default: 4,8,12)")
    a = p.parse_args()
    milestones = tuple(int(x) for x in a.milestones.split(','))
    assert len(milestones) >= 3, "Need at least 3 milestones"

    values, ends = {}, {}
    for block in milestones:
        root = a.run_root / "validation" / f"block{block:02d}"
        vals = []
        for seed in (1301, 1302):
            vals.append(utility(root / f"seed{seed}" / "workflow_v2" / "summary.json"))
        values[block] = np.asarray(vals, float)
        report = json.loads((a.run_root / "training" / f"update{block:02d}" / "optimizer_updates.json").read_text())
        if report.get("architecture") != "dueling_ddqn":
            raise RuntimeError("non-dueling training report")
        ends[block] = report["history"][-1]

    all_ends = {}
    for block in range(1, milestones[-1] + 1):
        report = json.loads((a.run_root / "training" / f"update{block:02d}" / "optimizer_updates.json").read_text())
        all_ends[block] = report["history"][-1]

    # Paired changes for all consecutive milestone pairs
    paired = {}
    for i in range(len(milestones) - 1):
        b1, b2 = milestones[i], milestones[i + 1]
        paired[f"M{i+1}_to_M{i+2}"] = paired_change(values[b1], values[b2])

    means = {b: float(values[b].mean()) for b in milestones}
    final_b = milestones[-1]
    second_last_b = milestones[-2]
    margin = 0.05 * abs(means[final_b])

    # Stage 1: progressive learning at first milestone pair
    m12 = paired["M1_to_M2"]
    stage1 = means[milestones[1]] >= 1.20 * means[milestones[0]] and m12["positive_scenarios"] == 2

    # Stage 2: policy refinement — all pairs from M2 onward retain >= 95%
    stage2 = all(means[milestones[i + 1]] >= 0.95 * means[milestones[i]] for i in range(1, len(milestones) - 1))

    # Stage 3: no regression on final milestone pair
    final_pair_key = f"M{len(milestones)-1}_to_M{len(milestones)}"
    m_final = paired[final_pair_key]
    equivalence = m_final["ci95"][0] >= -margin
    action_ok = ends[final_b]["fixed_state_action_agreement"] >= 0.90
    td_ok = ends[final_b]["mean_absolute_td_error"] <= 1.10 * ends[second_last_b]["mean_absolute_td_error"]
    q_ok = ends[final_b]["q_mean_absolute_drift"] <= 1.10 * ends[second_last_b]["q_mean_absolute_drift"]

    out = {
        "architecture": "dueling_ddqn",
        "run_root": str(a.run_root),
        "validation_seeds": [1301, 1302],
        "milestones": list(milestones),
        "milestone_utilities": {str(k): values[k].tolist() for k in milestones},
        "milestone_means": {str(k): means[k] for k in milestones},
        "paired_changes": paired,
        "stage1_progressive_learning": {"passes": bool(stage1), "target_improvement": 0.20},
        "stage2_policy_refinement": {"passes": bool(stage2), "retention_target": 0.95},
        "stage3_stable_convergence": {
            "paired_equivalence_margin": margin,
            "final_pair_no_regression": bool(equivalence),
            "final_action_agreement": ends[final_b]["fixed_state_action_agreement"],
            "action_agreement_passes": bool(action_ok),
            "td_error_stable": bool(td_ok),
            "q_drift_stable": bool(q_ok),
        },
        "all_block_checkpoint_diagnostics": {str(k): all_ends[k] for k in range(1, milestones[-1] + 1)},
        "converged": bool(stage1 and stage2 and equivalence and action_ok and td_ok and q_ok),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out))

if __name__ == "__main__":
    main()
