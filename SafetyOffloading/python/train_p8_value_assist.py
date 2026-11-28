#!/usr/bin/env python3
"""Train one future-value network for learning-assisted myopic (Alternative B)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

FEATURES = [
    "ctx_pending_tasks",
    "ctx_committed_offloads",
    "ctx_mean_p_hat",
    "ctx_candidate_assignments",
    "ctx_mean_risk",
    "ctx_mean_slack",
]


def relu(x):
    return x if x > 0 else 0.0


def forward(x, w1, b1, w2, b2):
    h = [relu(sum(w * v for w, v in zip(row, x)) + b) for row, b in zip(w1, b1)]
    y = sum(w * v for w, v in zip(w2[0], h)) + b2[0]
    return h, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, default=Path("outputs/p8_replay/replay.csv"))
    ap.add_argument("--output", type=Path, default=Path("models/p8_value_assist.json"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=0.001)
    args = ap.parse_args()
    random.seed(args.seed)
    df = pd.read_csv(args.replay)
    train = df[df["configuration_id"].isin([1, 5, 6, 22])].copy()
    val = df[df["configuration_id"] == 31].copy()
    if len(train) < 100:
        train = df.copy()
        val = df.sample(min(500, len(df)), random_state=args.seed)

    # Target: discounted remaining completion reward within the same (cfg,seed,policy) stream.
    train = train.sort_values(["configuration_id", "seed", "policy", "decision_id"])
    targets = []
    groups = train.groupby(["configuration_id", "seed", "policy"], sort=False)
    for _, g in groups:
        rewards = g["reward"].fillna(0.0).tolist()
        n = len(rewards)
        ret = [0.0] * n
        acc = 0.0
        for i in range(n - 1, -1, -1):
            acc = float(rewards[i]) + args.gamma * acc
            ret[i] = acc
        targets.extend(ret)
    train = train.copy()
    train["return_to_go"] = targets

    means = [float(train[f].mean()) for f in FEATURES]
    scales = [max(float(train[f].std(ddof=0)), 1e-6) for f in FEATURES]

    def state(row):
        return [
            ((float(row[f]) if pd.notna(row[f]) else 0.0) - m) / s
            for f, m, s in zip(FEATURES, means, scales)
        ]

    rows = train.to_dict("records")
    n, h = len(FEATURES), args.hidden
    w1 = [[random.uniform(-0.08, 0.08) for _ in range(n)] for _ in range(h)]
    b1 = [0.0] * h
    w2 = [[random.uniform(-0.08, 0.08) for _ in range(h)]]
    b2 = [0.0]

    for epoch in range(args.epochs):
        random.shuffle(rows)
        for row in rows:
            x = state(row)
            hidden, y = forward(x, w1, b1, w2, b2)
            target = float(row["return_to_go"])
            err = max(-5.0, min(5.0, y - target))
            old = w2[0][:]
            for j in range(h):
                w2[0][j] -= args.lr * err * hidden[j]
            b2[0] -= args.lr * err
            for j in range(h):
                if hidden[j] > 0:
                    g = err * old[j]
                    for k in range(n):
                        w1[j][k] -= args.lr * g * x[k]
                    b1[j] -= args.lr * g

    def mse(data):
        if len(data) == 0:
            return {"rows": 0, "mse": None}
        # Build returns for val the same way if columns exist
        ss = 0.0
        n_ = 0
        for row in data.to_dict("records"):
            x = state(row)
            _, y = forward(x, w1, b1, w2, b2)
            # Without return_to_go on val, use reward as weak proxy for reporting only.
            t = float(row["reward"]) if "return_to_go" not in row else float(row["return_to_go"])
            ss += (y - t) ** 2
            n_ += 1
        return {"rows": n_, "mse": ss / max(1, n_)}

    # Attach returns on val for a fairer metric.
    val2 = val.sort_values(["configuration_id", "seed", "policy", "decision_id"]).copy()
    vt = []
    for _, g in val2.groupby(["configuration_id", "seed", "policy"], sort=False):
        rewards = g["reward"].fillna(0.0).tolist()
        acc = 0.0
        ret = [0.0] * len(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            acc = float(rewards[i]) + args.gamma * acc
            ret[i] = acc
        vt.extend(ret)
    if len(vt) == len(val2):
        val2["return_to_go"] = vt

    artifact = {
        "type": "value_assist",
        "features": [
            "pending_left",
            "committed_offloads",
            "score_or_p_hat",
            "n_plan_or_candidates",
            "risk",
            "completion",
        ],
        "train_feature_order": FEATURES,
        "means": means,
        "scales": scales,
        "hidden_weights": w1,
        "hidden_bias": b1,
        "output_weights": w2,
        "output_bias": b2,
        "training": {
            "epochs": args.epochs,
            "gamma": args.gamma,
            "lr": args.lr,
            "train_rows": int(len(train)),
            "val_cfg": 31,
        },
        "validation": {"train": mse(train.sample(min(2000, len(train)), random_state=1)), "val": mse(val2)},
        "note": "V̂ predicts discounted remaining completion return; plugged into frozen risk-constrained myopic.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["validation"], indent=2))


if __name__ == "__main__":
    main()
