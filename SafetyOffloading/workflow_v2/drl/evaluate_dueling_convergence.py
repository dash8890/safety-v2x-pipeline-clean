#!/usr/bin/env python3
"""Evaluate the mandated three-milestone, matched-scenario convergence evidence.

Adapted for 12-block (60-rollout) training runs with milestones at blocks 4, 8, 12.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

MILESTONES = (4, 8, 12)

def utility(path: Path) -> float:
    return float(json.loads(path.read_text())["total_value"])

def paired_change(before: np.ndarray, after: np.ndarray) -> dict:
    change = after - before
    mean = float(change.mean()); se = float(change.std(ddof=1) / math.sqrt(len(change)))
    lo, hi = mean - 12.706 * se, mean + 12.706 * se
    return {"per_scenario": change.tolist(), "mean": mean, "std": float(change.std(ddof=1)),
            "ci95": [lo, hi], "positive_scenarios": int(np.count_nonzero(change > 0))}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    values, ends = {}, {}
    for block in MILESTONES:
        root=a.run_root/"validation"/f"block{block:02d}"
        vals=[]
        for seed in (1301,1302):
            vals.append(utility(root/f"seed{seed}"/"workflow_v2"/"summary.json"))
        values[block]=np.asarray(vals,float)
        report=json.loads((a.run_root/"training"/f"update{block:02d}"/"optimizer_updates.json").read_text())
        if report.get("architecture") != "dueling_ddqn": raise RuntimeError("non-dueling training report")
        ends[block]=report["history"][-1]
    all_ends={}
    for block in range(1,13):
        report=json.loads((a.run_root/"training"/f"update{block:02d}"/"optimizer_updates.json").read_text())
        all_ends[block]=report["history"][-1]
    m12=paired_change(values[4],values[8]); m23=paired_change(values[8],values[12])
    mean_m1=float(values[4].mean()); mean_m2=float(values[8].mean()); mean_m3=float(values[12].mean())
    margin=.05*abs(mean_m3)
    stage1=mean_m2 >= 1.20*mean_m1 and m12["positive_scenarios"] == 2
    stage2=mean_m3 >= .95*mean_m2
    # Equivalence/improvement: M2→M3 95% CI lower bound above -margin (no regression beyond 5%)
    equivalence = m23["ci95"][0] >= -margin
    action_ok=ends[12]["fixed_state_action_agreement"] >= .90
    td_ok=ends[12]["mean_absolute_td_error"] <= 1.10*ends[8]["mean_absolute_td_error"]
    q_ok=ends[12]["q_mean_absolute_drift"] <= 1.10*ends[8]["q_mean_absolute_drift"]
    out={"architecture":"dueling_ddqn","run_root":str(a.run_root),"validation_seeds":[1301,1302],
         "milestone_utilities":{str(k):values[k].tolist() for k in MILESTONES},
         "milestone_means":{str(k):float(values[k].mean()) for k in MILESTONES},
         "paired_changes":{"M1_to_M2":m12,"M2_to_M3":m23},
         "stage1_progressive_learning":{"passes":bool(stage1),"target_improvement":.20},
         "stage2_policy_refinement":{"passes":bool(stage2),"retention_target":.95},
         "stage3_stable_convergence":{"paired_equivalence_margin":margin,"m2_to_m3_no_regression":bool(equivalence),"final_action_agreement":ends[12]["fixed_state_action_agreement"],"action_agreement_passes":bool(action_ok),"td_error_stable":bool(td_ok),"q_drift_stable":bool(q_ok)},
         "all_12_block_checkpoint_diagnostics":{str(k):all_ends[k] for k in range(1,13)},
         "converged":bool(stage1 and stage2 and equivalence and action_ok and td_ok and q_ok)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out))

if __name__=="__main__": main()
