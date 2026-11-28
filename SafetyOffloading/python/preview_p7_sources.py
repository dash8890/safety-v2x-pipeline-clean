"""Dry-run inventory of P4/P6 sources for P7 without launching MATLAB/sims."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "p7_data_generation.json").read_text())


def main() -> None:
    p4 = ROOT / CFG["sources"]["p4_baseline_root"]
    p6 = ROOT / CFG["sources"]["p6_myopic_root"]
    configs = (
        CFG["sources"]["primary_configs"]
        + CFG["sources"]["negative_control_configs"]
        + CFG["sources"]["generalization_configs"]
    )
    train = set(CFG["sources"]["train_seeds"])
    val = set(CFG["sources"]["validation_seeds"])
    held = set(CFG["sources"]["heldout_seeds_reserved_for_p10"])
    seeds = sorted(train | val)
    assert not (train & val), "train/validation seed overlap"
    assert not ((train | val) & held), "held-out seed overlap"

    rows = []
    for cfg_id in configs:
        for seed in seeds:
            case = p4 / f"cfg{cfg_id:02d}_seed{seed}"
            myopic_dir = p6 / f"cfg{cfg_id:02d}_seed{seed}" / "risk_myopic_p6"
            myopic = myopic_dir / "safety_offloading"
            entry = {
                "configuration_id": cfg_id,
                "seed": seed,
                "partition": "train" if seed in train else "validation",
                "p4_case": case.is_dir(),
                "deadline_local": (case / "deadline_local" / "safety_offloading" / "tasks.csv").is_file(),
                "p6_myopic_complete": (p6 / f"cfg{cfg_id:02d}_seed{seed}" / "CASE_COMPLETE.mat").is_file()
                or (myopic_dir / "COMPLETE.mat").is_file()
                or (myopic / "tasks.csv").is_file(),
                "assignment_features": False,
                "schedule_epochs": False,
                "worlds": [],
            }
            if case.is_dir():
                for child in sorted(case.iterdir()):
                    so = child / "safety_offloading"
                    if so.is_dir() and (so / "tasks.csv").is_file():
                        entry["worlds"].append(child.name)
                        if (so / "assignment_features.csv").is_file():
                            entry["assignment_features"] = True
                        if (so / "schedule_epochs.csv").is_file():
                            entry["schedule_epochs"] = True
            if myopic.is_dir() and (myopic / "tasks.csv").is_file():
                entry["worlds"].append("risk_myopic_p6")
                entry["assignment_features"] |= (myopic / "assignment_features.csv").is_file()
                entry["schedule_epochs"] |= (myopic / "schedule_epochs.csv").is_file()
            rows.append(entry)

    ready = sum(1 for r in rows if r["deadline_local"])
    p6_done = sum(1 for r in rows if r["p6_myopic_complete"])
    assign = sum(1 for r in rows if r["assignment_features"])
    epochs = sum(1 for r in rows if r["schedule_epochs"])
    print(
        json.dumps(
            {
                "inventory_rows": len(rows),
                "p4_deadline_local_ready": ready,
                "p6_myopic_complete": p6_done,
                "with_assignment_features": assign,
                "with_schedule_epochs": epochs,
                "partition": {
                    "train_seeds": sorted(train),
                    "validation_seeds": sorted(val),
                    "heldout_reserved_p10": sorted(held),
                },
                "expected_gaps_before_p6_finish": [
                    "assignment_features / schedule_epochs appear only after P5/P6 worlds",
                    "validation seeds 4–5 have P4 but not the current 15-instance P6 plan",
                ],
                "detail": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
