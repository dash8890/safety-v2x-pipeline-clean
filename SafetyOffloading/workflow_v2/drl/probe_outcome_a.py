#!/usr/bin/env python3
"""Outcome A gate: full-5 mean must substantially beat PDP 45.6 (target >=52)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WF2 = ROOT.parent
OUT = WF2 / "outputs" / "wilab_outcome_a"
BASE_CFG = WF2 / "config" / "wilab_causal_eval.json"
MATLAB = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
HOLD = [300, 301, 302, 303, 304]
PDP = {300: 48.0, 301: 48.0, 302: 24.0, 303: 48.0, 304: 60.0}
TARGET = 52.0


def run(policy: str, seed: int) -> float:
    tag = f"{policy}_s{seed}"
    case = OUT / tag
    cfg_path = OUT / f"{tag}_cfg.json"
    cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    cfg["workflow"]["collect_transitions"] = False
    cfg["workflow"]["collect_demos"] = False
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if case.exists():
        shutil.rmtree(case, ignore_errors=True)
    case.mkdir(parents=True, exist_ok=True)
    script = (
        f"addpath(genpath('{WF2.as_posix()}')); "
        f"try; s=runR3PilotCase('{case.as_posix()}','{policy}',{seed},"
        f"'config','{cfg_path.as_posix()}','radio_mode','wilab_mode2'); disp(jsonencode(s)); "
        f"catch ME; disp(getReport(ME)); exit(1); end; exit(0);"
    )
    r = subprocess.run([MATLAB, "-batch", script], capture_output=True, text=True, timeout=300)
    sp = case / "workflow_v2" / "summary.json"
    if not sp.is_file():
        raise RuntimeError(f"{tag} failed rc={r.returncode}\n{(r.stdout or '')[-800:]}")
    return float(json.loads(sp.read_text(encoding="utf-8"))["total_value"])


def eval_policy(policy: str) -> dict:
    per = {}
    for s in HOLD:
        v = run(policy, s)
        per[s] = v
        print(f"{policy} seed={s} V={v:.1f} pdp={PDP[s]:.1f}", flush=True)
    mean = sum(per.values()) / len(HOLD)
    return {
        "policy": policy,
        "per_seed": per,
        "mean": mean,
        "beats_pdp": mean > 45.6,
        "hits_target": mean >= TARGET,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # PDP full-5 already measured: do NOT re-run Drop-Low. Only probe new policy.
    pdp_ref = {
        "policy": "workflow_drop_low",
        "per_seed": PDP,
        "mean": 45.6,
        "beats_pdp": False,
        "hits_target": False,
        "source": "prior_holdout",
    }
    results = [pdp_ref]
    congest = eval_policy("workflow_drop_low_congest")
    results.append(congest)
    print(
        f"MEAN congest={congest['mean']:.2f} "
        f"beats={congest['beats_pdp']} target52={congest['hits_target']}",
        flush=True,
    )
    best = max(results, key=lambda r: r["mean"])
    report = {"results": results, "best": best, "pdp_mean": 45.6, "target": TARGET}
    path = OUT / "outcome_a_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"WROTE {path}", flush=True)
    if congest["mean"] >= TARGET:
        return 0
    if congest["mean"] > 45.6:
        return 1  # above PDP but short of 52
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
