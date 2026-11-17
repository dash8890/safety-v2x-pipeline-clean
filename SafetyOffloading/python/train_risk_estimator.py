#!/usr/bin/env python3
"""Train and calibrate the deployable safety-harm estimator.

Uses only the Python standard library. Splits by scenario seed, fits a
regularized logistic model, and derives one-sided Wilson calibration bounds
without treating packet rows as independent training examples.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

FEATURES = [
    "cbr", "density", "candidate_helpers", "helper_distance_m",
    "origin_queue_length", "helper_queue_length", "input_bytes",
    "output_bytes", "cycles", "deadline_ms",
]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("risk dataset is empty")
    missing = set(FEATURES + ["seed", "any_harm"]) - set(rows[0])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return rows


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except ValueError:
        return 1.0 if value.lower() in {"true", "yes"} else 0.0


def split(rows: list[dict[str, str]]):
    seeds = sorted({int(float(r["seed"])) for r in rows})
    if len(seeds) < 3:
        raise ValueError("at least three independent seeds are required")
    calibration_seed, test_seed = seeds[-2], seeds[-1]
    train = [r for r in rows if int(float(r["seed"])) not in {calibration_seed, test_seed}]
    calibration = [r for r in rows if int(float(r["seed"])) == calibration_seed]
    test = [r for r in rows if int(float(r["seed"])) == test_seed]
    return train, calibration, test, calibration_seed, test_seed


def transform(rows, means=None, scales=None):
    raw = [[number(r, f) for f in FEATURES] for r in rows]
    if means is None:
        means = [sum(x[j] for x in raw) / len(raw) for j in range(len(FEATURES))]
        scales = []
        for j, mean in enumerate(means):
            variance = sum((x[j] - mean) ** 2 for x in raw) / max(1, len(raw) - 1)
            scales.append(max(math.sqrt(variance), 1e-12))
    x = [[1.0] + [(v - means[j]) / scales[j] for j, v in enumerate(row)] for row in raw]
    y = [int(number(r, "any_harm") > 0) for r in rows]
    return x, y, means, scales


def fit(x, y, steps: int, rate: float, l2: float):
    weights = [0.0] * len(x[0])
    for step in range(steps):
        gradient = [0.0] * len(weights)
        for row, target in zip(x, y):
            error = sigmoid(sum(a * b for a, b in zip(weights, row))) - target
            for j, value in enumerate(row):
                gradient[j] += error * value
        for j in range(len(weights)):
            penalty = 0.0 if j == 0 else l2 * weights[j]
            weights[j] -= rate / math.sqrt(1.0 + step / 100.0) * (gradient[j] / len(x) + penalty)
    return weights


def predict(weights, x):
    return [sigmoid(sum(a * b for a, b in zip(weights, row))) for row in x]


def metrics(y, p):
    eps = 1e-12
    loss = -sum(t * math.log(max(q, eps)) + (1 - t) * math.log(max(1 - q, eps)) for t, q in zip(y, p)) / len(y)
    return {"rows": len(y), "event_rate": sum(y) / len(y), "brier": sum((t-q)**2 for t, q in zip(y,p))/len(y), "log_loss": loss}


def wilson_upper(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    if total == 0:
        return 1.0
    p = successes / total
    denominator = 1 + z*z/total
    center = p + z*z/(2*total)
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))
    return min(1.0, (center + radius) / denominator)


def calibration_bins(y, p, count=5):
    order = sorted(range(len(p)), key=p.__getitem__)
    bins = []
    for b in range(count):
        ids = order[b*len(order)//count:(b+1)*len(order)//count]
        if not ids:
            continue
        observed = sum(y[i] for i in ids)
        bins.append({"p_min": min(p[i] for i in ids), "p_max": max(p[i] for i in ids),
                     "count": len(ids), "events": observed,
                     "risk_upper_95": wilson_upper(observed, len(ids))})
    return bins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("risk_estimator.json"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    rows = load(args.dataset)
    train, calibration, test, calibration_seed, test_seed = split(rows)
    if args.validate_only:
        print(json.dumps({"rows": len(rows), "train": len(train), "calibration": len(calibration),
                          "test": len(test), "calibration_seed": calibration_seed,
                          "held_out_test_seed": test_seed}, indent=2))
        return
    x_train, y_train, means, scales = transform(train)
    x_cal, y_cal, _, _ = transform(calibration, means, scales)
    x_test, y_test, _, _ = transform(test, means, scales)
    weights = fit(x_train, y_train, args.steps, args.learning_rate, args.l2)
    p_train, p_cal, p_test = predict(weights, x_train), predict(weights, x_cal), predict(weights, x_test)
    artifact = {
        "model": "regularized_logistic_harm_classifier",
        "target": "any harmful flip within the post-decision safety window",
        "features": FEATURES,
        "means": means, "scales": scales, "weights": weights,
        "split": {"calibration_seed": calibration_seed, "held_out_test_seed": test_seed},
        "metrics": {"train": metrics(y_train, p_train), "calibration": metrics(y_cal, p_cal), "test": metrics(y_test, p_test)},
        "calibration_bins": calibration_bins(y_cal, p_cal),
        "ood_rule": "reject optional offload outside training feature ranges",
        "feature_ranges": {f: [min(number(r, f) for r in train), max(number(r, f) for r in train)] for f in FEATURES},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["metrics"], indent=2))


if __name__ == "__main__":
    main()
