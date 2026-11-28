"""Assemble P7 datasets from finished P4/P6 outputs without MATLAB.

Mirrors buildP7Datasets.m:
  - scenario-seed train/validation partition (no shuffle, no held-out leakage)
  - action coverage rows + counterfactual candidates when present
  - action-level conditional risk separate from policy-level paired safety
  - deficiency report before any additional experience generation

Does not launch simulations.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_cfg(path: Path) -> dict:
    return json.loads(path.read_text())


def _assert_partitions(cfg: dict) -> None:
    train = set(cfg["sources"]["train_seeds"])
    val = set(cfg["sources"]["validation_seeds"])
    held = set(cfg["sources"]["heldout_seeds_reserved_for_p10"])
    if train & val:
        raise SystemExit(f"Train/validation seed overlap: {sorted(train & val)}")
    if (train | val) & held:
        raise SystemExit("Train/validation collide with P10 held-out seeds.")


def _tag_partition(df: pd.DataFrame, train: set[int], val: set[int]) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["partition"] = pd.Series(dtype="string")
        return df
    part = np.full(len(df), "other", dtype=object)
    part[df["seed"].isin(train)] = "train"
    part[df["seed"].isin(val)] = "validation"
    out = df.copy()
    out["partition"] = part
    return out


def _discover_worlds(case_dir: Path, p6_root: Path, cfg_id: int, seed: int) -> list[tuple[str, Path]]:
    worlds: list[tuple[str, Path]] = []
    local = case_dir / "deadline_local" / "safety_offloading"
    if local.is_dir():
        worlds.append(("deadline_local", local))

    summary = case_dir / "case_summary.csv"
    if summary.is_file():
        tab = pd.read_csv(summary)
        if "source_world" in tab.columns:
            for _, row in tab.iterrows():
                src = str(row.get("source_world", "") or "")
                method = str(row.get("method", src) or src)
                if src in ("", "deadline_local") or method.startswith("safety_cand"):
                    continue
                folder = case_dir / src / "safety_offloading"
                if folder.is_dir():
                    worlds.append((method, folder))
    if len(worlds) <= 1:
        for name in ("radio_search", "safety_search", "risk_myopic", "ideal_oracle"):
            folder = case_dir / name / "safety_offloading"
            if folder.is_dir():
                worlds.append((name, folder))

    myopic = p6_root / f"cfg{cfg_id:02d}_seed{seed}" / "risk_myopic_p6" / "safety_offloading"
    if myopic.is_dir():
        worlds.append(("risk_myopic_p6", myopic))

    # Deduplicate by folder path, keep first name.
    seen: set[str] = set()
    uniq: list[tuple[str, Path]] = []
    for name, folder in worlds:
        key = str(folder.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, folder))
    return uniq


def _pair_safety(baseline: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame | None:
    keys = ["flow_id", "seq", "tx_id", "rx_id"]
    for k in keys:
        if k not in baseline.columns or k not in policy.columns:
            return None
    b = baseline[keys + ["success_on_time", "distance_bin"]].rename(
        columns={"success_on_time": "baseline_success"}
    )
    p = policy[keys + ["success_on_time"]].rename(columns={"success_on_time": "policy_success"})
    paired = b.merge(p, on=keys, how="inner")
    if len(paired) != len(b) or len(paired) != len(p):
        return None
    paired["harmful_flip"] = paired["baseline_success"].astype(bool) & ~paired["policy_success"].astype(bool)
    paired["avoided_failure"] = ~paired["baseline_success"].astype(bool) & paired["policy_success"].astype(bool)
    return paired


def _extract_policy(
    so: Path, cfg_id: int, seed: int, policy_name: str, base_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks_path = so / "tasks.csv"
    if not tasks_path.is_file():
        return tuple(pd.DataFrame() for _ in range(5))  # type: ignore[return-value]

    tasks = pd.read_csv(tasks_path)
    features = pd.read_csv(so / "assignment_features.csv") if (so / "assignment_features.csv").is_file() else pd.DataFrame()
    decisions = pd.read_csv(so / "decisions.csv") if (so / "decisions.csv").is_file() else pd.DataFrame()

    # Feasibility lookup from decisions
    feas_map: dict[int, str] = {}
    if not decisions.empty and "feasibility_class" in decisions.columns:
        for _, d in decisions.iterrows():
            feas_map[int(d["task_id"])] = str(d["feasibility_class"])

    comp_rows = []
    for _, t in tasks.iterrows():
        tid = int(t["task_id"])
        proc = t.get("destination", np.nan)
        if not np.isfinite(proc) or proc == 0:
            proc = t["origin_id"]
        act = str(t["action"])
        row = {
            "configuration_id": cfg_id,
            "seed": seed,
            "policy": policy_name,
            "task_id": tid,
            "processor_id": proc,
            "action": act,
            "is_offload": act == "offload",
            "is_local_action": act == "local",
            "completed_on_time": float(t["completed_on_time"]),
            "cycles": float(t["cycles"]),
            "in_bytes": float(t["in_bytes"]),
            "out_bytes": float(t["out_bytes"]),
            "deadline_horizon_ms": float(t["deadline_ms"] - t["release_ms"]),
            "feasibility_class": feas_map.get(tid, "NA"),
            "is_realized": True,
        }
        if not features.empty:
            hit = features[(features["task_id"] == tid) & (features.get("chosen", 0) == 1)]
            if hit.empty:
                hit = features[(features["task_id"] == tid) & (features["processor_id"] == proc)]
            if not hit.empty:
                f = hit.iloc[0]
                for c in (
                    "predicted_completion_probability",
                    "predicted_deadline_slack_ms",
                    "predicted_conditional_safety_risk",
                    "processor_cpu_ghz",
                    "is_local",
                    "estimated_input_latency_ms",
                    "estimated_helper_queueing_ms",
                    "estimated_computation_latency_ms",
                    "estimated_result_latency_ms",
                    "relative_distance_m",
                ):
                    if c in f.index:
                        row[c] = f[c]
                if "feasibility_class" in f.index and row["feasibility_class"] == "NA":
                    row["feasibility_class"] = str(f["feasibility_class"])
        comp_rows.append(row)

    if not features.empty:
        for _, f in features.iterrows():
            if "chosen" in f.index and bool(f["chosen"]):
                continue
            is_local = bool(f["is_local"]) if "is_local" in f.index else False
            act = "candidate_local" if is_local else "candidate_offload"
            row = {
                "configuration_id": cfg_id,
                "seed": seed,
                "policy": policy_name,
                "task_id": int(f["task_id"]),
                "processor_id": float(f["processor_id"]),
                "action": act,
                "is_offload": act == "candidate_offload",
                "is_local_action": act == "candidate_local",
                "completed_on_time": np.nan,
                "cycles": np.nan,
                "in_bytes": np.nan,
                "out_bytes": np.nan,
                "deadline_horizon_ms": np.nan,
                "feasibility_class": str(f["feasibility_class"]) if "feasibility_class" in f.index else "NA",
                "is_realized": False,
            }
            for c in (
                "predicted_completion_probability",
                "predicted_deadline_slack_ms",
                "predicted_conditional_safety_risk",
                "processor_cpu_ghz",
                "is_local",
                "estimated_input_latency_ms",
                "estimated_helper_queueing_ms",
                "estimated_computation_latency_ms",
                "estimated_result_latency_ms",
                "relative_distance_m",
            ):
                if c in f.index:
                    row[c] = f[c]
            comp_rows.append(row)

    completion = pd.DataFrame(comp_rows)

    off = tasks[tasks["action"].astype(str) == "offload"].copy()
    lat_rows = []
    for _, t in off.iterrows():
        lat_rows.append(
            {
                "configuration_id": cfg_id,
                "seed": seed,
                "policy": policy_name,
                "task_id": int(t["task_id"]),
                "processor_id": t["destination"],
                "completed_on_time": float(t["completed_on_time"]),
                "input_latency_ms": t["input_delivered_ms"] - t["release_ms"],
                "helper_queue_ms": t["cpu_start_ms"] - t["input_delivered_ms"],
                "compute_ms": t["cpu_finish_ms"] - t["cpu_start_ms"],
                "result_latency_ms": t["result_delivered_ms"] - t["cpu_finish_ms"],
                "e2e_ms": t["result_delivered_ms"] - t["release_ms"],
            }
        )
    latency = pd.DataFrame(lat_rows)

    action_rows = []
    if not features.empty:
        for _, f in features.iterrows():
            action_rows.append(
                {
                    "configuration_id": cfg_id,
                    "seed": seed,
                    "policy": policy_name,
                    "decision_id": f.get("decision_id", np.nan),
                    "task_id": int(f["task_id"]),
                    "processor_id": f["processor_id"],
                    "is_local": bool(f["is_local"]) if "is_local" in f.index else False,
                    "chosen": bool(f["chosen"]) if "chosen" in f.index else False,
                    "feasibility_class": str(f["feasibility_class"]) if "feasibility_class" in f.index else "NA",
                    "action_conditional_risk": f.get("predicted_conditional_safety_risk", np.nan),
                    "predicted_completion_probability": f.get("predicted_completion_probability", np.nan),
                    "relative_distance_m": f.get("relative_distance_m", np.nan),
                    "masked": bool(f["masked"]) if "masked" in f.index else False,
                    "label_kind": "action_level_provisional",
                }
            )
    elif not decisions.empty:
        for _, d in decisions.iterrows():
            action_rows.append(
                {
                    "configuration_id": cfg_id,
                    "seed": seed,
                    "policy": policy_name,
                    "decision_id": d.get("decision_id", np.nan),
                    "task_id": int(d["task_id"]),
                    "processor_id": d.get("helper_id", np.nan),
                    "is_local": str(d.get("chosen_action", "")) == "local",
                    "chosen": True,
                    "feasibility_class": str(d.get("feasibility_class", "NA")),
                    "action_conditional_risk": d.get("risk_upper", np.nan),
                    "predicted_completion_probability": np.nan,
                    "relative_distance_m": d.get("helper_distance_m", np.nan),
                    "masked": False,
                    "label_kind": "action_level_provisional_p4",
                }
            )
    action_risk = pd.DataFrame(action_rows)

    policy_safety = pd.DataFrame()
    base_packets = base_dir / "safety_packets.csv"
    pol_packets = so / "safety_packets.csv"
    if base_packets.is_file() and pol_packets.is_file() and policy_name != "deadline_local":
        paired = _pair_safety(pd.read_csv(base_packets), pd.read_csv(pol_packets))
        if paired is not None and len(paired):
            n = len(paired)
            harm = int(paired["harmful_flip"].sum())
            fail = int((~paired["policy_success"].astype(bool)).sum())
            policy_safety = paired.copy()
            policy_safety["configuration_id"] = cfg_id
            policy_safety["seed"] = seed
            policy_safety["policy"] = policy_name
            policy_safety["policy_failure"] = ~policy_safety["policy_success"].astype(bool)
            policy_safety["label_kind"] = "policy_level_paired"
            policy_safety["case_harm_cost"] = harm - 0.01 * n
            policy_safety["case_failure_cost"] = fail - 0.10 * n
            cols = [
                "configuration_id",
                "seed",
                "policy",
                "flow_id",
                "tx_id",
                "rx_id",
                "distance_bin",
                "baseline_success",
                "policy_success",
                "harmful_flip",
                "avoided_failure",
                "policy_failure",
                "label_kind",
                "case_harm_cost",
                "case_failure_cost",
            ]
            policy_safety = policy_safety[cols]

    exp_rows = []
    ep_path = so / "schedule_epochs.csv"
    if ep_path.is_file():
        ep = pd.read_csv(ep_path).sort_values(["epoch_ms", "decision_id"]).reset_index(drop=True)
        for i, e in ep.iterrows():
            next_id = float(ep.loc[i + 1, "decision_id"]) if i + 1 < len(ep) else np.nan
            exp_rows.append(
                {
                    "configuration_id": cfg_id,
                    "seed": seed,
                    "policy": policy_name,
                    "decision_id": e["decision_id"],
                    "time_ms": e["epoch_ms"],
                    "action_type": str(e["action_type"]),
                    "task_id": e["task_id"],
                    "processor_id": e["processor_id"],
                    "pending_tasks": e.get("pending_tasks", np.nan),
                    "candidate_assignments": e.get("candidate_assignments", np.nan),
                    "next_decision_id": next_id,
                    "done": i + 1 == len(ep),
                    "reward": np.nan,
                    "action_level_harm_cost": np.nan,
                    "label_kind": "per_decision_no_policy_harm",
                }
            )
    elif not decisions.empty:
        dec = decisions.sort_values(["time_ms", "decision_id"]).reset_index(drop=True)
        for i, d in dec.iterrows():
            next_id = float(dec.loc[i + 1, "decision_id"]) if i + 1 < len(dec) else np.nan
            exp_rows.append(
                {
                    "configuration_id": cfg_id,
                    "seed": seed,
                    "policy": policy_name,
                    "decision_id": d["decision_id"],
                    "time_ms": d["time_ms"],
                    "action_type": str(d["chosen_action"]),
                    "task_id": d["task_id"],
                    "processor_id": d.get("helper_id", np.nan),
                    "pending_tasks": np.nan,
                    "candidate_assignments": np.nan,
                    "next_decision_id": next_id,
                    "done": i + 1 == len(dec),
                    "reward": np.nan,
                    "action_level_harm_cost": np.nan,
                    "label_kind": "legacy_decisions_chain",
                }
            )
    experience = pd.DataFrame(exp_rows)
    rew_path = so / "rewards.csv"
    if not experience.empty and rew_path.is_file():
        rew = pd.read_csv(rew_path)
        if "task_id" in rew.columns and "reward" in rew.columns:
            by_task = rew.groupby("task_id")["reward"].sum()
            experience["reward"] = experience["task_id"].map(by_task)

    return completion, latency, action_risk, policy_safety, experience


def _audit(
    completion: pd.DataFrame,
    action_risk: pd.DataFrame,
    experience: pd.DataFrame,
    inventory: list[dict],
    cfg: dict,
    p4: Path,
    p6: Path,
    configs: list[int],
    seeds: list[int],
) -> dict:
    realized = completion[completion["is_realized"] == True] if "is_realized" in completion.columns else completion  # noqa: E712
    feas = realized["feasibility_class"].astype(str) if len(realized) and "feasibility_class" in realized.columns else pd.Series(dtype=str)
    acts = realized["action"].astype(str) if len(realized) else pd.Series(dtype=str)
    off = realized[acts == "offload"] if len(realized) else realized
    n_helpers = int(off["processor_id"].nunique()) if len(off) and "processor_id" in off.columns else 0
    missing_p6 = []
    missing_val_p4 = []
    for cfg_id in configs:
        for seed in seeds:
            case = p6 / f"cfg{cfg_id:02d}_seed{seed}"
            ok = (
                (case / "CASE_COMPLETE.mat").is_file()
                or (case / "risk_myopic_p6" / "COMPLETE.mat").is_file()
                or (case / "risk_myopic_p6" / "safety_offloading").is_dir()
            )
            if not ok:
                missing_p6.append(f"cfg{cfg_id:02d}_seed{seed}")
            if seed in cfg["sources"]["validation_seeds"]:
                p4t = p4 / f"cfg{cfg_id:02d}_seed{seed}" / "deadline_local" / "safety_offloading" / "tasks.csv"
                if not p4t.is_file():
                    missing_val_p4.append(f"cfg{cfg_id:02d}_seed{seed}")

    return {
        "n_realized_rows": int(len(realized)),
        "n_counterfactual_candidate_rows": int((~completion["is_realized"]).sum())
        if "is_realized" in completion.columns
        else 0,
        "n_class_II": int((feas == "II").sum()),
        "n_class_III": int((feas == "III").sum()),
        "n_local_actions": int((acts == "local").sum()),
        "n_offload_success": int(((off["action"].astype(str) == "offload") & (off["completed_on_time"] == 1)).sum())
        if len(off)
        else 0,
        "n_offload_fail": int(((off["action"].astype(str) == "offload") & (off["completed_on_time"] == 0)).sum())
        if len(off)
        else 0,
        "n_distinct_helpers": n_helpers,
        "has_assignment_features_any": any(i["has_assignment_features"] for i in inventory),
        "has_schedule_epochs_any": any(i["has_schedule_epochs"] for i in inventory),
        "n_experience_rows": int(len(experience)),
        "n_sequential_transitions": int(experience["next_decision_id"].notna().sum())
        if len(experience) and "next_decision_id" in experience.columns
        else 0,
        "n_action_risk_rows": int(len(action_risk)),
        "missing_p6_instances": missing_p6,
        "missing_validation_p4": missing_val_p4,
        "train_seeds": cfg["sources"]["train_seeds"],
        "validation_seeds": cfg["sources"]["validation_seeds"],
        "checks": {
            "data_partitioning": "train/validation by scenario seed; held-out reserved for P10",
            "action_coverage": "Classes II/III, helpers, offload success/fail, local",
            "safety_labels": "action_conditional_risk.csv vs policy_safety_costs.csv; no policy harm on actions",
        },
    }


def _deficiencies(audit: dict, cfg: dict) -> dict:
    items = []
    if audit["n_class_II"] < 1:
        items.append(
            {"code": "missing_class_II", "severity": "blocking", "detail": "No Class II realized actions."}
        )
    if audit["n_class_III"] < 1:
        items.append(
            {"code": "missing_class_III", "severity": "blocking", "detail": "No Class III realized actions."}
        )
    if audit["n_local_actions"] < 1:
        items.append(
            {
                "code": "missing_local_execution",
                "severity": "blocking",
                "detail": "No local-execution outcomes.",
            }
        )
    if audit["n_offload_success"] < 1 or audit["n_offload_fail"] < 1:
        items.append(
            {
                "code": "incomplete_offload_outcomes",
                "severity": "blocking",
                "detail": "Need both successful and unsuccessful offloads.",
            }
        )
    if audit["n_distinct_helpers"] < 2:
        items.append(
            {
                "code": "insufficient_helper_diversity",
                "severity": "warning",
                "detail": "Fewer than two distinct helpers among offloads.",
            }
        )
    if not audit["has_assignment_features_any"]:
        items.append(
            {
                "code": "no_counterfactual_assignment_features",
                "severity": "warning",
                "detail": "P4 lacks assignment_features.csv; counterfactual coverage depends on P6.",
            }
        )
    if not audit["has_schedule_epochs_any"]:
        items.append(
            {
                "code": "no_schedule_epochs",
                "severity": "warning",
                "detail": "No schedule_epochs.csv yet; sequential transitions use legacy decisions chains.",
            }
        )
    if audit["n_sequential_transitions"] < 1:
        items.append(
            {
                "code": "no_sequential_transitions",
                "severity": "warning",
                "detail": "Experience table has no next_decision links.",
            }
        )
    if audit["n_counterfactual_candidate_rows"] < 1:
        items.append(
            {
                "code": "no_counterfactual_candidates",
                "severity": "warning",
                "detail": "No unchosen assignment candidates.",
            }
        )
    if audit["missing_p6_instances"]:
        items.append(
            {
                "code": "p6_incomplete",
                "severity": "info",
                "detail": f"P6 myopic worlds missing for {len(audit['missing_p6_instances'])} cfg/seed instances.",
            }
        )
    if audit["missing_validation_p4"]:
        items.append(
            {
                "code": "validation_p4_missing",
                "severity": "blocking",
                "detail": "Validation seeds lack P4 deadline_local sources.",
            }
        )
    val_seeds = set(cfg["sources"]["validation_seeds"])
    p6_val_missing = 0
    for inst in audit["missing_p6_instances"]:
        m = re.search(r"seed(\d+)$", inst)
        if m and int(m.group(1)) in val_seeds:
            p6_val_missing += 1
    if p6_val_missing > 0:
        items.append(
            {
                "code": "p6_validation_seed_gap",
                "severity": "warning",
                "detail": "Validation seeds lack risk_myopic_p6 worlds.",
            }
        )

    blocking = sum(1 for i in items if i["severity"] == "blocking")
    warning_n = sum(1 for i in items if i["severity"] == "warning")
    needs_extra = any(
        i["code"]
        in {
            "no_counterfactual_candidates",
            "no_schedule_epochs",
            "no_sequential_transitions",
            "no_counterfactual_assignment_features",
        }
        for i in items
    )
    return {
        "items": items,
        "blocking_count": blocking,
        "warning_count": warning_n,
        "ready_for_calibration": blocking == 0 and audit["n_realized_rows"] > 0,
        "needs_additional_experience": needs_extra,
        "note": "Generate missing experience only after reviewing deficiencies; do not relaunch P4/P5.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "p7_data_generation.json")
    ap.add_argument("--output-root", type=Path, default=None)
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    _assert_partitions(cfg)

    out_root = args.output_root or (ROOT / "outputs" / "p7_datasets")
    out_root.mkdir(parents=True, exist_ok=True)
    p4 = ROOT / cfg["sources"]["p4_baseline_root"]
    p6 = ROOT / cfg["sources"]["p6_myopic_root"]
    configs = sorted(
        set(
            cfg["sources"]["primary_configs"]
            + cfg["sources"]["negative_control_configs"]
            + cfg["sources"]["generalization_configs"]
        )
    )
    train = set(cfg["sources"]["train_seeds"])
    val = set(cfg["sources"]["validation_seeds"])
    held = set(cfg["sources"]["heldout_seeds_reserved_for_p10"])
    seeds = sorted(train | val)

    comps, lats, risks, pols, exps = [], [], [], [], []
    inventory = []
    for cfg_id in configs:
        for seed in seeds:
            if seed in held:
                raise SystemExit(f"Refused held-out seed {seed}")
            case_dir = p4 / f"cfg{cfg_id:02d}_seed{seed}"
            if not case_dir.is_dir():
                continue
            worlds = _discover_worlds(case_dir, p6, cfg_id, seed)
            base_dir = case_dir / "deadline_local" / "safety_offloading"
            for name, folder in worlds:
                inventory.append(
                    {
                        "configuration_id": cfg_id,
                        "seed": seed,
                        "policy": name,
                        "folder": str(folder),
                        "has_tasks": (folder / "tasks.csv").is_file(),
                        "has_assignment_features": (folder / "assignment_features.csv").is_file(),
                        "has_schedule_epochs": (folder / "schedule_epochs.csv").is_file(),
                        "has_safety_packets": (folder / "safety_packets.csv").is_file(),
                    }
                )
                c, l, a, s, e = _extract_policy(folder, cfg_id, seed, name, base_dir)
                if len(c):
                    comps.append(c)
                if len(l):
                    lats.append(l)
                if len(a):
                    risks.append(a)
                if len(s):
                    pols.append(s)
                if len(e):
                    exps.append(e)

    completion = _tag_partition(pd.concat(comps, ignore_index=True) if comps else pd.DataFrame(), train, val)
    latency = _tag_partition(pd.concat(lats, ignore_index=True) if lats else pd.DataFrame(), train, val)
    action_risk = _tag_partition(pd.concat(risks, ignore_index=True) if risks else pd.DataFrame(), train, val)
    policy_safety = _tag_partition(pd.concat(pols, ignore_index=True) if pols else pd.DataFrame(), train, val)
    experience = _tag_partition(pd.concat(exps, ignore_index=True) if exps else pd.DataFrame(), train, val)

    completion.to_csv(out_root / "completion_prediction.csv", index=False)
    latency.to_csv(out_root / "latency_prediction.csv", index=False)
    action_risk.to_csv(out_root / "action_conditional_risk.csv", index=False)
    policy_safety.to_csv(out_root / "policy_safety_costs.csv", index=False)
    policy_safety.to_csv(out_root / "safety_prediction.csv", index=False)
    experience.to_csv(out_root / "sequential_experience.csv", index=False)

    audit = _audit(completion, action_risk, experience, inventory, cfg, p4, p6, configs, seeds)
    deficiencies = _deficiencies(audit, cfg)
    (out_root / "coverage_audit.json").write_text(json.dumps(audit, indent=2))
    (out_root / "deficiencies.json").write_text(json.dumps(deficiencies, indent=2))

    manifest = {
        "output_root": str(out_root),
        "completion_rows": int(len(completion)),
        "latency_rows": int(len(latency)),
        "action_conditional_risk_rows": int(len(action_risk)),
        "policy_safety_rows": int(len(policy_safety)),
        "experience_rows": int(len(experience)),
        "configs": configs,
        "seeds": seeds,
        "train_seeds": sorted(train),
        "validation_seeds": sorted(val),
        "partition_rule": "scenario_seed_strict",
        "safety_label_rule": "action_conditional_risk_separate_from_policy_costs",
        "deficiency_count": len(deficiencies["items"]),
        "ready_for_calibration": deficiencies["ready_for_calibration"],
        "needs_additional_experience": deficiencies["needs_additional_experience"],
        "builder": "python/build_p7_datasets.py",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
