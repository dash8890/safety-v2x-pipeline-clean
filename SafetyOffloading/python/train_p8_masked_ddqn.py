#!/usr/bin/env python3
"""Train one Masked Double DQN candidate from P8 offline replay (Alternative A)."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd

CTX = [
    "ctx_pending_tasks",
    "ctx_candidate_assignments",
    "ctx_committed_offloads",
    "ctx_mean_p_hat",
    "ctx_mean_risk",
    "ctx_mean_slack",
]
AFEAT = [
    "a_is_local",
    "a_processor_cpu_ghz",
    "a_estimated_input_latency_ms",
    "a_estimated_result_latency_ms",
    "a_estimated_helper_queueing_ms",
    "a_estimated_computation_latency_ms",
    "a_predicted_deadline_slack_ms",
    "a_predicted_completion_probability",
    "a_predicted_conditional_safety_risk",
    "a_relative_distance_m",
    "a_estimated_resource_block_slots",
]
FEATURES = CTX + AFEAT


def relu(x):
    return x if x > 0 else 0.0


def forward(x, w1, b1, w2, b2):
    h = [relu(sum(w * v for w, v in zip(row, x)) + b) for row, b in zip(w1, b1)]
    q = [sum(w * v for w, v in zip(row, h)) + b for row, b in zip(w2, b2)]
    return h, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, default=Path("outputs/p8_replay/replay.csv"))
    ap.add_argument("--output", type=Path, default=Path("models/p8_masked_ddqn.json"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=0.001)
    args = ap.parse_args()
    random.seed(args.seed)
    df = pd.read_csv(args.replay)
    # Hold out cfg31 for independent validation (dev generalization); train on 1,5,6,22.
    train = df[df["configuration_id"].isin([1, 5, 6, 22])].copy()
    val = df[df["configuration_id"] == 31].copy()
    if len(train) < 100:
        train = df.copy()
        val = df.sample(min(500, len(df)), random_state=args.seed)

    means = [float(train[f].mean()) for f in FEATURES]
    scales = [
        max(float(train[f].std(ddof=0)), 1e-6) if f in train else 1.0 for f in FEATURES
    ]

    def state(row, zero_assign=False):
        x = []
        for f, m, s in zip(FEATURES, means, scales):
            if zero_assign and f.startswith("a_"):
                v = 0.0
            else:
                v = float(row[f]) if f in row and pd.notna(row[f]) else 0.0
            x.append((v - m) / s)
        return x

    # Action classes: 0=ASSIGN(chosen features), 1=REJECT, 2=STOP
    def action_id(row):
        t = str(row["action_type"]).upper()
        if t == "ASSIGN":
            return 0
        if t == "REJECT":
            return 1
        return 2

    transitions = train.to_dict("records")
    n, h, na = len(FEATURES), args.hidden, 3
    w1 = [[random.uniform(-0.08, 0.08) for _ in range(n)] for _ in range(h)]
    b1 = [0.0] * h
    w2 = [[random.uniform(-0.08, 0.08) for _ in range(h)] for _ in range(na)]
    b2 = [0.0] * na
    tw1 = [r[:] for r in w1]
    tb1 = b1[:]
    tw2 = [r[:] for r in w2]
    tb2 = b2[:]

    def next_state_features(row):
        # Approximate next context from logged next_pending/candidates; assignment zeros.
        fake = {f: 0.0 for f in FEATURES}
        fake["ctx_pending_tasks"] = float(row.get("next_pending", 0) or 0)
        fake["ctx_candidate_assignments"] = float(row.get("next_candidates", 0) or 0)
        fake["ctx_committed_offloads"] = float(row.get("ctx_committed_offloads", 0) or 0)
        fake["ctx_mean_p_hat"] = float(row.get("ctx_mean_p_hat", 0) or 0)
        fake["ctx_mean_risk"] = float(row.get("ctx_mean_risk", 0) or 0)
        fake["ctx_mean_slack"] = float(row.get("ctx_mean_slack", 0) or 0)
        return state(fake)

    for epoch in range(args.epochs):
        random.shuffle(transitions)
        for row in transitions:
            aid = action_id(row)
            x = state(row, zero_assign=(aid != 0))
            hidden, q = forward(x, w1, b1, w2, b2)
            reward = float(row.get("reward", 0) or 0)
            done = float(row.get("done", 0) or 0) >= 0.5
            target = reward
            if not done:
                nx = next_state_features(row)
                # Double DQN: online argmax, target evaluation.
                _, nq_online = forward(nx, w1, b1, w2, b2)
                a_star = max(range(na), key=lambda j: nq_online[j])
                _, nq_tgt = forward(nx, tw1, tb1, tw2, tb2)
                target += args.gamma * nq_tgt[a_star]
            err = max(-5.0, min(5.0, q[aid] - target))
            old = w2[aid][:]
            for j in range(h):
                w2[aid][j] -= args.lr * err * hidden[j]
            b2[aid] -= args.lr * err
            for j in range(h):
                if hidden[j] > 0:
                    g = err * old[j]
                    for k in range(n):
                        w1[j][k] -= args.lr * g * x[k]
                    b1[j] -= args.lr * g
        if (epoch + 1) % 5 == 0:
            tw1 = [r[:] for r in w1]
            tb1 = b1[:]
            tw2 = [r[:] for r in w2]
            tb2 = b2[:]

    def behavior_match(data):
        if len(data) == 0:
            return {"rows": 0, "match": 0.0}
        ok = 0
        for row in data.to_dict("records"):
            aid = action_id(row)
            x = state(row, zero_assign=(aid != 0))
            _, q = forward(x, w1, b1, w2, b2)
            # Compare ASSIGN vs non using chosen-action class.
            pred = max(range(3), key=lambda j: q[j])
            ok += int(pred == aid)
        return {"rows": int(len(data)), "match": ok / max(1, len(data))}

    # Export features as assignment-only names expected by MATLAB (context prepended separately).
    assign_feature_names = [
        "is_local",
        "processor_cpu_ghz",
        "estimated_input_latency_ms",
        "estimated_result_latency_ms",
        "estimated_helper_queueing_ms",
        "estimated_computation_latency_ms",
        "predicted_deadline_slack_ms",
        "predicted_completion_probability",
        "predicted_conditional_safety_risk",
        "relative_distance_m",
        "estimated_resource_block_slots",
    ]
    # MATLAB forward uses ctx(6)+assignment features; store full means/scales in order CTX||AFEAT
    artifact = {
        "type": "masked_ddqn",
        "algorithm": "offline_double_dqn",
        "features": assign_feature_names,
        "context_features": [
            "pending_tasks",
            "candidate_assignments",
            "committed_offloads",
            "mean_p_hat",
            "mean_risk",
            "mean_slack",
        ],
        "feature_order": FEATURES,
        "means": means,
        "scales": scales,
        "hidden_weights": w1,
        "hidden_bias": b1,
        # Single shared trunk; class heads via output rows 0/1/2. MATLAB uses row0 for ASSIGN
        # and head_bias for REJECT/STOP relative to ASSIGN trunk output.
        "output_weights": [w2[0]],
        "output_bias": [b2[0]],
        "head_bias": {"reject": b2[1] - b2[0], "stop": b2[2] - b2[0]},
        "full_output_weights": w2,
        "full_output_bias": b2,
        "training": {
            "epochs": args.epochs,
            "gamma": args.gamma,
            "lr": args.lr,
            "train_rows": int(len(train)),
            "val_cfg": 31,
            "train_configs": [1, 5, 6, 22],
        },
        "validation": {
            "train_behavior": behavior_match(train.sample(min(2000, len(train)), random_state=1)),
            "val_behavior": behavior_match(val),
        },
        "risk_shield": "cumulative_base*(1+committed) <= harm_limit hard mask at inference",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["validation"], indent=2))


if __name__ == "__main__":
    main()
