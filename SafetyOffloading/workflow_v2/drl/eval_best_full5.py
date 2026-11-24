#!/usr/bin/env python3
"""Parallel FULL5 on BEST weights (admit shield). One MATLAB per seed; no global kill."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WF2 = ROOT.parent
OUT = WF2 / "outputs" / "wilab_best_full5"
BASE_CFG = WF2 / "config" / "wilab_causal_eval.json"
MATLAB = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
BEST_W = ROOT / "outputs" / "ddqn_wilab_converged_best.weights.json"
FALLBACK_W = ROOT / "outputs" / "ddqn_wilab_causal.weights.json"
HOLD = [300, 301, 302, 303, 304]
PDP = {300: 48.0, 301: 48.0, 302: 24.0, 303: 48.0, 304: 60.0}
TARGET = 52.0


def run(seed: int, weights: Path) -> tuple[int, float]:
    tag = f"best_s{seed}"
    case = OUT / tag
    cfg_path = OUT / f"{tag}_cfg.json"
    cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    cfg["workflow"]["ddqn_weights_json"] = str(weights).replace("\\", "/")
    cfg["workflow"]["ddqn_eps"] = 0.0
    cfg["workflow"]["collect_transitions"] = False
    cfg["workflow"]["collect_demos"] = False
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if case.exists():
        shutil.rmtree(case, ignore_errors=True)
    case.mkdir(parents=True, exist_ok=True)
    log_path = case / "matlab.log"
    script = (
        f"addpath(genpath('{WF2.as_posix()}')); "
        f"try; s=runR3PilotCase('{case.as_posix()}','workflow_causal_ddqn',{seed},"
        f"'config','{cfg_path.as_posix()}','radio_mode','wilab_mode2'); disp(jsonencode(s)); "
        f"catch ME; disp(getReport(ME)); exit(1); end; exit(0);"
    )
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            [MATLAB, "-batch", script],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        summary = case / "workflow_v2" / "summary.json"
        t0 = time.time()
        last_n = 0
        last_prog = t0
        while True:
            if summary.is_file():
                time.sleep(0.5)
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except Exception:
                        proc.kill()
                break
            try:
                n = sum(1 for _ in case.rglob("*") if _.is_file())
            except Exception:
                n = last_n
            if n > last_n:
                last_n = n
                last_prog = time.time()
            elif time.time() - t0 > 45 and time.time() - last_prog > 120:
                proc.kill()
                break
            if proc.poll() is not None:
                break
            if time.time() - t0 > 240:
                proc.kill()
                break
            time.sleep(0.4)
    if not summary.is_file():
        raise RuntimeError(
            f"seed={seed} no summary\n{log_path.read_text(encoding='utf-8', errors='ignore')[-600:]}"
        )
    v = float(json.loads(summary.read_text(encoding="utf-8"))["total_value"])
    print(f"seed={seed} V={v:.1f} pdp={PDP[seed]:.1f} delta={v - PDP[seed]:+.1f}", flush=True)
    return seed, v


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    weights = BEST_W if BEST_W.is_file() else FALLBACK_W
    if not weights.is_file():
        raise FileNotFoundError("no BEST/causal weights")
    shutil.copy2(weights, FALLBACK_W)
    print(f"FULL5 parallel weights={weights}", flush=True)
    per = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(run, s, weights): s for s in HOLD}
        for fut in as_completed(futs):
            seed, v = fut.result()
            per[seed] = v
    mean = sum(per.values()) / len(HOLD)
    pdp_mean = sum(PDP.values()) / len(HOLD)
    report = {
        "weights": str(weights),
        "per_seed": {str(k): per[k] for k in HOLD},
        "mean": mean,
        "pdp_per_seed": PDP,
        "pdp_mean": pdp_mean,
        "target": TARGET,
        "beats_pdp": mean > pdp_mean,
        "hits_target": mean >= TARGET,
        "admit_shield": True,
        "mode": "parallel",
    }
    (OUT / "full5_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"FULL5_MEAN={mean:.2f} PDP={pdp_mean:.2f} target52={mean >= TARGET}", flush=True)
    return 0 if mean >= TARGET else (1 if mean > pdp_mean else 2)


if __name__ == "__main__":
    raise SystemExit(main())
