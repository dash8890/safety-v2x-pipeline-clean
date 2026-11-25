#!/usr/bin/env python3
"""Evaluate the mandated four-milestone, matched-scenario convergence evidence."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

MILESTONES = (4, 8, 12, 16)

def utility(path: Path) -> float:
    return float(json.loads(path.read_text())["total_value"])

def paired_change(before: np.ndarray, after: np.ndarray) -> dict:
    change = after - before
    mean = float(change.mean()); se = float(change.std(ddof=1) / math.sqrt(len(change)))
    # Prespecified two-sided 95% t interval for n=5 paired scenarios, df=4.
    lo, hi = mean - 2.776 * se, mean + 2.776 * se
    return {"per_scenario": change.tolist(), "mean": mean, "std": float(change.std(ddof=1)),
            "ci95": [lo, hi], "positive_scenarios": int(np.count_nonzero(change > 0))}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    values, ends = {}, {}
    for block in MILESTONES:
        root=a.run_root/"validation"/f"block{block:02d}"
        vals=[]
        for seed in range(1301,1306):
            vals.append(utility(root/f"seed{seed}"/"workflow_v2"/"summary.json"))
        values[block]=np.asarray(vals,float)
        report=json.loads((a.run_root/"training"/f"update{block:02d}"/"optimizer_updates.json").read_text())
        if report.get("architecture") != "dueling_ddqn": raise RuntimeError("non-dueling training report")
        ends[block]=report["history"][-1]
    m12=paired_change(values[4],values[8]); m23=paired_change(values[8],values[12]); m34=paired_change(values[12],values[16])
    mean_m1=float(values[4].mean()); mean_m2=float(values[8].mean()); margin=.05*abs(float(values[12].mean()))
    stage1=mean_m2 >= 1.20*mean_m1 and m12["positive_scenarios"] >= 4
    stage2=values[12].mean() >= .95*values[8].mean()
    # Equivalence is established only if the complete paired 95% CI lies
    # inside the prespecified +/-5% M3 utility margin.
    equivalence = m34["ci95"][0] >= -margin and m34["ci95"][1] <= margin
    action_ok=ends[16]["fixed_state_action_agreement"] >= .90
    td_ok=ends[16]["mean_absolute_td_error"] <= 1.10*ends[12]["mean_absolute_td_error"]
    q_ok=ends[16]["q_mean_absolute_drift"] <= 1.10*ends[12]["q_mean_absolute_drift"]
    out={"architecture":"dueling_ddqn","run_root":str(a.run_root),"validation_seeds":[1301,1302,1303,1304,1305],
         "milestone_utilities":{str(k):values[k].tolist() for k in MILESTONES},
         "milestone_means":{str(k):float(values[k].mean()) for k in MILESTONES},
         "paired_changes":{"M1_to_M2":m12,"M2_to_M3":m23,"M3_to_M4":m34},
         "stage1_progressive_learning":{"passes":bool(stage1),"target_improvement":.20},
         "stage2_policy_refinement":{"passes":bool(stage2),"retention_target":.95},
         "stage3_stable_convergence":{"paired_equivalence_margin":margin,"paired_equivalence_passes":bool(equivalence),"final_action_agreement":ends[16]["fixed_state_action_agreement"],"action_agreement_passes":bool(action_ok),"td_error_stable":bool(td_ok),"q_drift_stable":bool(q_ok)},
         "block_end_diagnostics":{str(k):ends[k] for k in MILESTONES},
         "converged":bool(stage1 and stage2 and equivalence and action_ok and td_ok and q_ok)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out))

if __name__=="__main__": main()
