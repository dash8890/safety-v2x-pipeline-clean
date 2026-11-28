"""P7.2 coverage audit over assembled datasets (or dry-run over sources).

Reports Class II/III, helper diversity, offload success/fail, local execution,
counterfactual candidates, sequential transitions, and seed-partition integrity.
Does not launch simulations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "p7_data_generation.json").read_text())


def _partition_ok(df: pd.DataFrame) -> dict:
    train_seeds = set(CFG["sources"]["train_seeds"])
    val_seeds = set(CFG["sources"]["validation_seeds"])
    held = set(CFG["sources"]["heldout_seeds_reserved_for_p10"])
    out = {
        "train_val_overlap": sorted(train_seeds & val_seeds),
        "heldout_overlap": sorted((train_seeds | val_seeds) & held),
        "ok": True,
    }
    if out["train_val_overlap"] or out["heldout_overlap"]:
        out["ok"] = False
    if "partition" in df.columns and "seed" in df.columns:
        bad_train = sorted(set(df.loc[df["partition"] == "train", "seed"]) - train_seeds)
        bad_val = sorted(set(df.loc[df["partition"] == "validation", "seed"]) - val_seeds)
        out["mistagged_train_seeds"] = bad_train
        out["mistagged_validation_seeds"] = bad_val
        if bad_train or bad_val:
            out["ok"] = False
    return out


def audit_assembled(data_root: Path) -> dict:
    comp = pd.read_csv(data_root / "completion_prediction.csv") if (data_root / "completion_prediction.csv").exists() else pd.DataFrame()
    exp = pd.read_csv(data_root / "sequential_experience.csv") if (data_root / "sequential_experience.csv").exists() else pd.DataFrame()
    action = pd.read_csv(data_root / "action_conditional_risk.csv") if (data_root / "action_conditional_risk.csv").exists() else pd.DataFrame()
    policy = data_root / "policy_safety_costs.csv"
    if not policy.exists():
        policy = data_root / "safety_prediction.csv"
    pol = pd.read_csv(policy) if policy.exists() else pd.DataFrame()

    realized = comp
    if "is_realized" in comp.columns:
        realized = comp[comp["is_realized"] == True]  # noqa: E712
    counterfactual = int((~comp["is_realized"]).sum()) if "is_realized" in comp.columns else 0

    def _count(series, val):
        return int((series.astype(str) == val).sum()) if len(series) else 0

    feas = realized["feasibility_class"].astype(str) if "feasibility_class" in realized.columns else pd.Series(dtype=str)
    action_col = realized["action"].astype(str) if "action" in realized.columns else pd.Series(dtype=str)
    off_ok = off_fail = 0
    if len(realized) and "completed_on_time" in realized.columns:
        off = realized[action_col == "offload"]
        off_ok = int((off["completed_on_time"] == 1).sum())
        off_fail = int((off["completed_on_time"] == 0).sum())
    helpers = 0
    if len(realized) and "processor_id" in realized.columns:
        helpers = int(realized.loc[action_col == "offload", "processor_id"].nunique())

    seq = 0
    if len(exp) and "next_decision_id" in exp.columns:
        seq = int(exp["next_decision_id"].notna().sum())

    safety_sep = {
        "action_table_rows": int(len(action)),
        "policy_table_rows": int(len(pol)),
        "action_has_policy_harm_column": bool(
            len(action) and any(c in action.columns for c in ("harmful_flip", "case_harm_cost"))
        ),
        "experience_broadcasts_policy_harm": bool(
            len(exp) and any(c in exp.columns for c in ("harm_cost_total", "failure_cost_total"))
        ),
    }

    report = {
        "source": str(data_root),
        "partition": _partition_ok(comp if len(comp) else action if len(action) else pol),
        "action_coverage": {
            "class_II": _count(feas, "II"),
            "class_III": _count(feas, "III"),
            "local": _count(action_col, "local"),
            "offload_success": off_ok,
            "offload_fail": off_fail,
            "distinct_helpers": helpers,
            "counterfactual_candidates": counterfactual,
        },
        "sequential": {
            "experience_rows": int(len(exp)),
            "transitions_with_next": seq,
        },
        "safety_label_separation": safety_sep,
        "deficiencies_file": str(data_root / "deficiencies.json")
        if (data_root / "deficiencies.json").exists()
        else None,
    }
    if report["deficiencies_file"]:
        report["deficiencies"] = json.loads(Path(report["deficiencies_file"]).read_text())
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "outputs" / "p7_datasets",
    )
    args = ap.parse_args()
    report = audit_assembled(args.data_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
