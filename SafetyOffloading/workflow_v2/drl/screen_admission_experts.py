#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
from pathlib import Path

import train_wilab_convergence as w

OUT = w.WF2 / "outputs" / "wilab_admission_expert_screen"
BASE = w.OUT / "ddqn_wilab_outcome_a_fixed_best.weights.json"


def run(ep: int, seed: int = 302) -> tuple[int, float]:
    case = OUT / f"ep{ep:02d}_seed{seed}"
    cfg = OUT / f"ep{ep:02d}_seed{seed}_cfg.json"
    expert = w.OUT / f"ddqn_wilab_admission_ep{ep:02d}.weights.json"
    ov = {
        "ddqn_admission_weights_json": str(expert).replace("\\", "/"),
        "ddqn_placement_weights_json": str(BASE).replace("\\", "/"),
    }
    result = w.run_wilab(case, cfg, seed, 0.0, False, 240, 0.0, ov)
    return ep, float(result["summary"]["total_value"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    w.WEIGHTS = BASE
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, ep) for ep in range(1, 9)]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"expert_ep{row[0]:02d} seed302={row[1]:.1f}", flush=True)
    print(sorted(rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
