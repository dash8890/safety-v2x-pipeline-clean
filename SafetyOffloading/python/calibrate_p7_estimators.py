"""P7 calibration: shared completion/latency/risk estimators (one initial round).

Train/validation by scenario seed only. Normalization and fits use train rows.
Action-level risk is recalibrated from provisional scores + distance-bin empirical
policy rates (never copies case harm onto individual action label columns).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _seed_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "partition" in df.columns:
        train = df[df["partition"] == "train"].copy()
        val = df[df["partition"] == "validation"].copy()
        if len(train) == 0 and len(val) == 0:
            raise SystemExit("No train/validation rows after partition filter.")
        return train, val
    seeds = sorted(df["seed"].unique())
    cut = max(1, int(0.7 * len(seeds)))
    return df[df["seed"].isin(seeds[:cut])].copy(), df[df["seed"].isin(seeds[cut:])].copy()


def _assert_no_seed_leak(train: pd.DataFrame, val: pd.DataFrame) -> None:
    if "seed" not in train.columns or "seed" not in val.columns:
        return
    leak = set(train["seed"]).intersection(set(val["seed"]))
    if leak:
        raise SystemExit(f"Seed leakage between train and validation: {sorted(leak)}")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _logit_fit(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    means = np.nanmean(x, axis=0)
    scales = np.nanstd(x, axis=0)
    scales[scales < 1e-6] = 1.0
    z = np.nan_to_num((x - means) / scales, nan=0.0)
    target = 2 * y - 1
    design = np.column_stack([np.ones(len(z)), z])
    w, *_ = np.linalg.lstsq(design, target, rcond=None)
    return {
        "means": means.tolist(),
        "scales": scales.tolist(),
        "weights": w.tolist(),
        "feature_dim": int(x.shape[1]),
        "fit_n": int(len(y)),
    }


def _predict_logit(model: dict, x: np.ndarray) -> np.ndarray:
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    w = np.asarray(model["weights"], dtype=float)
    z = np.nan_to_num((x - means) / scales, nan=0.0)
    design = np.column_stack([np.ones(len(z)), z])
    return _sigmoid(design @ w)


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    n = len(y)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if not np.any(m):
            continue
        total += (np.sum(m) / n) * abs(float(np.mean(p[m]) - np.mean(y[m])))
    return float(total)


def _auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    y = y.astype(int)
    if y.min() == y.max():
        return None
    order = np.argsort(p)
    y_sorted = y[order]
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    tp = 0.0
    fp = 0.0
    scores = []
    for label in y_sorted[::-1]:
        if label == 1:
            tp += 1
        else:
            fp += 1
        scores.append((fp / n_neg, tp / n_pos))
    # Trapezoid on ROC
    xs = [0.0] + [s[0] for s in scores] + [1.0]
    ys = [0.0] + [s[1] for s in scores] + [1.0]
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return float(area)


def calibrate_completion(path: Path, out: Path) -> dict:
    df = pd.read_csv(path)
    if "is_realized" in df.columns:
        df = df[df["is_realized"] == True]  # noqa: E712
    train, val = _seed_split(df)
    _assert_no_seed_leak(train, val)
    feature_cols = [
        c
        for c in [
            "predicted_completion_probability",
            "predicted_deadline_slack_ms",
            "predicted_conditional_safety_risk",
            "processor_cpu_ghz",
            "is_local",
            "is_local_action",
            "relative_distance_m",
        ]
        if c in train.columns
    ]
    if "completed_on_time" not in train.columns or len(train) == 0:
        model = {"type": "constant", "p": 0.5, "fit_partition": "train"}
    elif not feature_cols:
        model = {
            "type": "constant",
            "p": float(train["completed_on_time"].mean()),
            "fit_partition": "train",
            "fit_n": int(len(train)),
        }
    else:
        x = train[feature_cols].to_numpy(dtype=float)
        y = train["completed_on_time"].to_numpy(dtype=float)
        model = {
            "type": "logistic",
            "features": feature_cols,
            "fit_partition": "train",
            **_logit_fit(x, y),
        }
        p_tr = _predict_logit(model, x)
        model["train_metrics"] = {
            "brier": _brier(y, p_tr),
            "ece": _ece(y, p_tr),
            "base_rate": float(y.mean()),
            "auroc": _auroc(y, p_tr),
        }
        if len(val):
            xv = val[feature_cols].to_numpy(dtype=float)
            yv = val["completed_on_time"].to_numpy(dtype=float)
            pv = _predict_logit(model, xv)
            model["validation_n"] = int(len(val))
            model["validation_metrics"] = {
                "brier": _brier(yv, pv),
                "ece": _ece(yv, pv),
                "base_rate": float(yv.mean()),
                "auroc": _auroc(yv, pv),
            }
    out.write_text(json.dumps(model, indent=2))
    return model


def calibrate_latency(path: Path, out: Path) -> dict:
    df = pd.read_csv(path)
    train, val = _seed_split(df)
    _assert_no_seed_leak(train, val)
    labels = ["input_latency_ms", "helper_queue_ms", "compute_ms", "result_latency_ms", "e2e_ms"]
    model: dict = {
        "type": "empirical_quantiles",
        "fit_partition": "train",
        "fit_n": int(len(train)),
        "labels": {},
    }
    for lab in labels:
        if lab not in train.columns:
            continue
        s = train[lab].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        model["labels"][lab] = {
            "p50": float(s.quantile(0.5)),
            "p90": float(s.quantile(0.9)),
            "mean": float(s.mean()),
        }
    # Point predictor: train median e2e; report val MAE / P90 abs error
    if "e2e_ms" in train.columns and len(train):
        pred = float(train["e2e_ms"].replace([np.inf, -np.inf], np.nan).median())
        model["point_e2e_ms"] = pred
        if len(val) and "e2e_ms" in val.columns:
            err = (val["e2e_ms"].replace([np.inf, -np.inf], np.nan) - pred).abs().dropna()
            if len(err):
                model["validation_metrics"] = {
                    "n": int(len(err)),
                    "mae": float(err.mean()),
                    "p90_abs_error": float(err.quantile(0.9)),
                    "e2e_mean": float(val["e2e_ms"].mean()),
                }
    out.write_text(json.dumps(model, indent=2))
    return model


def _distance_bin_rates(policy_path: Path | None) -> dict[str, float]:
    if policy_path is None or not policy_path.exists():
        return {}
    df = pd.read_csv(policy_path)
    train, _ = _seed_split(df)
    if "distance_bin" not in train.columns or "harmful_flip" not in train.columns:
        return {}
    out: dict[str, float] = {}
    for b, g in train.groupby("distance_bin"):
        out[str(int(b))] = float(g["harmful_flip"].mean())
    return out


def calibrate_action_risk(path: Path, out: Path, policy_path: Path | None = None) -> dict:
    """Recalibrate provisional action risk; priority estimator for P7."""
    df = pd.read_csv(path)
    train, val = _seed_split(df)
    _assert_no_seed_leak(train, val)
    col = "action_conditional_risk"
    if col not in train.columns:
        model = {"type": "identity", "note": "missing action_conditional_risk column"}
        out.write_text(json.dumps(model, indent=2))
        return model

    s = train[col].replace([np.inf, -np.inf], np.nan).dropna()
    # Reliability bins on provisional score (train only).
    # empirical_upper uses max(bin mean provisional, distance-bin policy harm floor).
    dist_rates = _distance_bin_rates(policy_path)
    overall_floor = float(np.mean(list(dist_rates.values()))) if dist_rates else float(s.mean())

    qs = np.linspace(0, 1, 11)
    edges = s.quantile(qs).to_numpy(dtype=float)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([float(s.min()), float(s.max()) + 1e-9])

    bins = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (train[col] >= lo) & (train[col] <= hi if i == len(edges) - 2 else train[col] < hi)
        chunk = train.loc[m, col].dropna()
        if chunk.empty:
            continue
        emp = float(chunk.mean())
        # Inflate toward observed policy harm floor so underestimation is reduced.
        upper = max(emp, overall_floor, lo)
        bins.append(
            {
                "p_min": lo,
                "p_max": hi,
                "empirical_mean": emp,
                "empirical_upper": min(1.0, upper),
                "n": int(len(chunk)),
            }
        )

    model = {
        "type": "action_risk_reliability_bins",
        "label_kind": "action_conditional_risk",
        "fit_partition": "train",
        "fit_n": int(len(s)),
        "train_mean": float(s.mean()),
        "train_p50": float(s.quantile(0.5)),
        "policy_harm_floor": overall_floor,
        "distance_bin_harm_rates": dist_rates,
        "calibration_bins": bins,
        "note": "Maps provisional action risk through train reliability bins; "
        "empirical_upper floored by train policy harm. Does not write policy harm into action tables.",
    }

    # Validation: underestimation rate = fraction where provisional < policy floor while in val
    if len(val) and col in val.columns:
        vs = val[col].replace([np.inf, -np.inf], np.nan).dropna()
        model["validation_n"] = int(len(vs))
        model["validation_mean"] = float(vs.mean()) if len(vs) else None
        if len(vs):
            # Apply bins
            cal = []
            for p in vs.to_numpy(dtype=float):
                adj = p
                for b in bins:
                    if b["p_min"] <= p <= b["p_max"]:
                        adj = max(p, b["empirical_upper"])
                        break
                cal.append(adj)
            cal = np.asarray(cal)
            model["validation_metrics"] = {
                "provisional_mean": float(vs.mean()),
                "calibrated_mean": float(cal.mean()),
                "mean_inflation": float((cal - vs.to_numpy()).mean()),
                "frac_inflated": float(np.mean(cal > vs.to_numpy() + 1e-12)),
                "risk_underestimation_proxy": float(np.mean(vs.to_numpy() < overall_floor)),
            }
    out.write_text(json.dumps(model, indent=2))
    return model


def calibrate_policy_risk(path: Path, out: Path) -> dict:
    df = pd.read_csv(path)
    train, val = _seed_split(df)
    _assert_no_seed_leak(train, val)
    rate = float(train["harmful_flip"].mean()) if len(train) and "harmful_flip" in train.columns else 0.0
    fail = float(train["policy_failure"].mean()) if len(train) and "policy_failure" in train.columns else None
    by_bin: dict[str, float] = {}
    if "distance_bin" in train.columns and "harmful_flip" in train.columns:
        for b, g in train.groupby("distance_bin"):
            by_bin[str(int(b))] = float(g["harmful_flip"].mean())
    model = {
        "type": "policy_level_empirical_harm_rate",
        "label_kind": "policy_level_paired",
        "fit_partition": "train",
        "overall_harm_rate": rate,
        "overall_failure_rate": fail,
        "by_distance_bin": by_bin,
        "validation_n": int(len(val)),
        "validation_harm_rate": float(val["harmful_flip"].mean())
        if len(val) and "harmful_flip" in val.columns
        else None,
        "note": "Aggregate policy costs for P2 accounting; never assign to individual actions.",
    }
    if len(val) and "harmful_flip" in val.columns:
        # Constant predictor = train rate; Brier vs val flips
        y = val["harmful_flip"].to_numpy(dtype=float)
        p = np.full_like(y, rate, dtype=float)
        model["validation_metrics"] = {
            "brier_constant_train_rate": _brier(y, p),
            "base_rate": float(y.mean()),
        }
    out.write_text(json.dumps(model, indent=2))
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("outputs/p7_datasets"))
    ap.add_argument("--model-root", type=Path, default=Path("models"))
    args = ap.parse_args()
    args.model_root.mkdir(parents=True, exist_ok=True)

    deficiencies = args.data_root / "deficiencies.json"
    if deficiencies.exists():
        d = json.loads(deficiencies.read_text())
        if not d.get("ready_for_calibration", True):
            raise SystemExit(
                "deficiencies.json reports ready_for_calibration=false; "
                "resolve blocking coverage gaps before fitting estimators."
            )

    results = {}
    comp = args.data_root / "completion_prediction.csv"
    lat = args.data_root / "latency_prediction.csv"
    action = args.data_root / "action_conditional_risk.csv"
    policy = args.data_root / "policy_safety_costs.csv"
    if not policy.exists():
        policy = args.data_root / "safety_prediction.csv"

    if comp.exists():
        results["completion"] = calibrate_completion(comp, args.model_root / "p7_completion_calibrated.json")
    if lat.exists():
        results["latency"] = calibrate_latency(lat, args.model_root / "p7_latency_calibrated.json")
    if action.exists():
        results["action_risk"] = calibrate_action_risk(
            action, args.model_root / "p7_action_risk_calibrated.json", policy if policy.exists() else None
        )
    if policy.exists():
        results["policy_risk"] = calibrate_policy_risk(policy, args.model_root / "p7_policy_risk_calibrated.json")
        (args.model_root / "p7_risk_calibrated.json").write_text(json.dumps(results["policy_risk"], indent=2))

    summary = {
        "calibrated": list(results.keys()),
        "model_root": str(args.model_root.resolve()),
        "reference_for_p8": "calibrated_myopic_after_p7_not_provisional_p6",
        "fit_rule": "train_seeds_only_normalization_and_calibration",
        "round": "initial",
        "needs_additional_experience": False,
        "validation": {
            "completion": results.get("completion", {}).get("validation_metrics"),
            "latency": results.get("latency", {}).get("validation_metrics"),
            "action_risk": results.get("action_risk", {}).get("validation_metrics"),
            "policy_risk": results.get("policy_risk", {}).get("validation_metrics"),
        },
    }
    (args.model_root / "p7_calibration_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
