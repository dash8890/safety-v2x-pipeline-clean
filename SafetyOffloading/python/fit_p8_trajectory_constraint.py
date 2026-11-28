#!/usr/bin/env python3
"""Fit one trajectory-aware cumulative exposure model for P8.C (no architecture search).

Predicts marginal policy-harm exposure of admitting an offload given trajectory
features. Fit on train seeds only; labels are case-level P2 harm rates from the
frozen P7 matched summaries (policy-level), not per-action harm broadcast as a
reward. Inference uses the model as a cumulative budget gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATS = [
    "recent_cbr",
    "outstanding_tx",
    "cum_offloads_norm",
    "action_risk",
    "remaining_frac",
    "mean_selection_delay_proxy",
]


def num(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def collect_case_rows(p7_root: Path, p7_summary: Path, configs, train_seeds):
    summary = pd.read_csv(p7_summary)
    rows = []
    for cfg in configs:
        for seed in train_seeds:
            case = p7_root / f"cfg{cfg:02d}_seed{seed}"
            so = case / "risk_myopic_p7" / "safety_offloading"
            if not (so / "schedule_epochs.csv").is_file():
                continue
            href = summary[
                (summary.configuration_id.astype(int) == cfg)
                & (summary.seed.astype(int) == seed)
            ]
            if len(href) == 0:
                continue
            harm_rate = float(href.iloc[0]["harmful_failure_probability"])
            ep = pd.read_csv(so / "schedule_epochs.csv")
            af = (
                pd.read_csv(so / "assignment_features.csv")
                if (so / "assignment_features.csv").is_file()
                else pd.DataFrame()
            )
            tasks = pd.read_csv(so / "tasks.csv")
            t_end = float(tasks["deadline_ms"].max()) if len(tasks) else 22000.0
            t_start = float(tasks["release_ms"].min()) if len(tasks) else 2000.0
            horizon = max(1.0, t_end - t_start)
            cum_off = 0
            # Proxy outstanding: count in-flight offloads approx via prior ASSIGNs
            # to non-local without matching completion yet — use cum_off as scale.
            for r in ep.sort_values("decision_id").itertuples(index=False):
                at = str(getattr(r, "action_type", "")).upper()
                if at != "ASSIGN":
                    continue
                tid = int(num(getattr(r, "task_id", 0)))
                pid = int(num(getattr(r, "processor_id", 0)))
                hit = af[(af.decision_id == getattr(r, "decision_id")) & (af.task_id == tid) & (af.processor_id == pid)]
                is_local = True
                risk = 0.0
                slack = 0.0
                if len(hit):
                    is_local = bool(int(num(hit.iloc[0].get("is_local", 1))))
                    risk = num(hit.iloc[0].get("predicted_conditional_safety_risk", 0))
                    slack = num(hit.iloc[0].get("predicted_deadline_slack_ms", 0))
                if is_local:
                    continue
                cum_off += 1
                epoch_ms = num(getattr(r, "epoch_ms", t_start))
                # CBR / outstanding not logged per epoch; use risk and cum as trajectory proxies
                # plus selection-window delay proxy from negative slack magnitude.
                feat = {
                    "recent_cbr": min(1.0, risk * 50.0),  # monotone proxy if CBR absent in epoch log
                    "outstanding_tx": float(cum_off),  # prior offload commitments still "live" proxy
                    "cum_offloads_norm": cum_off / 50.0,
                    "action_risk": risk,
                    "remaining_frac": max(0.0, min(1.0, (t_end - epoch_ms) / horizon)),
                    "mean_selection_delay_proxy": max(0.0, -slack) / 100.0,
                    "configuration_id": cfg,
                    "seed": seed,
                    "harm_rate": harm_rate,
                    "label_harm": float(harm_rate > 0.01),
                }
                rows.append(feat)
    return pd.DataFrame(rows)


def fit_logistic(X, y, l2=1e-2, steps=800, lr=0.05):
    n, d = X.shape
    w = np.zeros(d + 1)
    for _ in range(steps):
        z = X @ w[1:] + w[0]
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        g = np.zeros_like(w)
        g[0] = np.mean(p - y)
        g[1:] = X.T @ (p - y) / n + l2 * w[1:]
        w -= lr * g
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p7-root", type=Path, default=Path("outputs/p7_calibrated_myopic_eval"))
    ap.add_argument("--p7-summary", type=Path, default=Path("outputs/p7_calibrated_myopic_eval/p7_summary.csv"))
    ap.add_argument("--out", type=Path, default=Path("models/p8_trajectory_constraint.json"))
    ap.add_argument("--configs", type=int, nargs="+", default=[1, 5, 6, 22, 31])
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    df = collect_case_rows(args.p7_root, args.p7_summary, args.configs, args.train_seeds)
    if len(df) < 20:
        raise SystemExit(f"insufficient offload trajectory rows: {len(df)}")

    X = df[FEATS].to_numpy(dtype=float)
    means = X.mean(axis=0)
    scales = np.maximum(X.std(axis=0), 1e-6)
    Xn = (X - means) / scales
    # Primary label: case harm violation; secondary regression target = harm_rate
    y = df["label_harm"].to_numpy(dtype=float)
    w = fit_logistic(Xn, y)

    # Exposure scale: map sigmoid score to marginal exposure units near action_risk magnitude
    z = Xn @ w[1:] + w[0]
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
    # Calibrate so mean predicted marginal ≈ mean action_risk on violating cases
    viol = df["harm_rate"] > 0.01
    target = float(df.loc[viol, "action_risk"].mean()) if viol.any() else float(df["action_risk"].mean())
    scale = target / max(float(p.mean()), 1e-6)

    artifact = {
        "type": "trajectory_constraint",
        "features": FEATS,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "weights": w.tolist(),  # bias + coeffs on normalized features
        "marginal_scale": scale,
        "harm_limit": 0.01,
        "defer_cbr_threshold": 0.25,
        "defer_outstanding_threshold": 8.0,
        "defer_interval_ms": 40.0,
        "fit": {
            "n_offload_rows": int(len(df)),
            "train_configs": args.configs,
            "train_seeds": args.train_seeds,
            "pos_rate": float(y.mean()),
            "mean_pred": float(p.mean()),
            "marginal_scale": scale,
            "note": "Labels are case-level P2 harm feasibility from P7 summaries; gate is cumulative exposure, not a safety certificate.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["fit"], indent=2))


if __name__ == "__main__":
    main()
