#!/usr/bin/env python3
"""Report training-only Dueling-DDQN convergence evidence efficiently."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

MILESTONES = (4, 8, 12, 16)

def load_summary(path: Path) -> float:
    return float(json.loads(path.read_text())["total_value"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    validation, block_updates = [], []
    for block in range(1, 17):
        d = a.run_root / "validation" / f"block{block:02d}"
        values = {int(x.parent.name.replace("seed", "")): load_summary(x)
                  for x in d.glob("seed*/workflow_v2/summary.json")}
        if 1301 not in values:
            raise RuntimeError(f"missing common validation seed for block {block}")
        validation.append({"block": block, "common_seed_utility": values[1301],
                           "seed_utilities": values,
                           "milestone_mean": float(np.mean(list(values.values()))) if block in MILESTONES else None})
        report = a.run_root / "training" / f"update{block:02d}" / "optimizer_updates.json"
        data = json.loads(report.read_text())
        if data.get("architecture") != "dueling_ddqn":
            raise RuntimeError(f"non-dueling report: {report}")
        block_updates.append(data["history"][-1])
    single = np.asarray([x["common_seed_utility"] for x in validation])
    milestone = {x["block"]: x["milestone_mean"] for x in validation if x["milestone_mean"] is not None}
    early, later = float(np.mean(single[:4])), float(np.mean(single[4:8]))
    progressive = later >= 1.10 * early and np.count_nonzero(np.diff(single[:8]) > 0) >= 3
    refine = bool(np.min(single[7:12]) >= .95 * np.max(single[:8]) and
                  np.mean(np.abs(np.diff(single[9:12]))) <= np.mean(np.abs(np.diff(single[7:10]))))
    preceding_best = max(milestone[4], milestone[8], milestone[12])
    stable = bool(milestone[16] >= .95 * preceding_best and np.min(single[11:]) >= .95 * np.mean(single[11:]))
    late = block_updates[11:]
    agreement = min(x["fixed_state_action_agreement"] for x in late)
    td = np.asarray([x["mean_absolute_td_error"] for x in late])
    q = np.asarray([x["q_p50"] for x in late])
    qdrift = np.asarray([x["q_mean_absolute_drift"] for x in late])
    value_stable = bool(np.all(np.isfinite(td)) and np.all(np.isfinite(q)) and
                        td[-1] <= 1.10 * np.mean(td) and qdrift[-1] <= 1.10 * np.mean(qdrift))
    out = {"run_root": str(a.run_root), "architecture": "dueling_ddqn",
           "validation_schedule": {"common_seed": 1301, "milestone_blocks": MILESTONES,
                                   "validation_count": 32},
           "validation": validation, "block_end_optimizer_diagnostics": block_updates,
           "stage1_progressive_learning": {"early_common_seed_mean": early, "later_common_seed_mean": later,
                                           "improvement_fraction": later / early - 1 if early else None, "passes": bool(progressive)},
           "stage2_policy_refinement": {"passes": refine},
           "stage3_stable_convergence": {"final_milestone_mean": milestone[16], "best_preceding_milestone_mean": preceding_best,
                                          "performance_passes": stable, "late_min_action_agreement": agreement,
                                          "policy_passes": agreement >= .90, "value_learning_passes": value_stable},
           "converged": bool(progressive and refine and stable and agreement >= .90 and value_stable)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)); print(json.dumps(out))

if __name__ == "__main__": main()
