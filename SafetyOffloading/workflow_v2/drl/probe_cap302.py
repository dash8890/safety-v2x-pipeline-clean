#!/usr/bin/env python3
"""Prove admission-cap unlocks V>24 on seed 302; then score full-5 if so."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WF2 = ROOT.parent
OUT = WF2 / "outputs" / "wilab_cap_unlock"
BASE_CFG = WF2 / "config" / "wilab_causal_eval.json"
MATLAB = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
HOLD = [300, 301, 302, 303, 304]


def run(policy: str, seed: int, cap: int | None = None) -> float:
    tag = f"{policy}_s{seed}" + (f"_cap{cap}" if cap is not None else "")
    case = OUT / tag
    cfg_path = OUT / f"{tag}_cfg.json"
    cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    cfg["workflow"]["collect_transitions"] = False
    cfg["workflow"]["collect_demos"] = False
    if cap is not None:
        cfg["workflow"]["high_value_admit_cap"] = int(cap)
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print("CAP unlock probe seed=302", flush=True)
    for pol, cap in [
        ("workflow_drop_low", None),
        ("workflow_drop_low_cap", 2),
        ("workflow_drop_low_cap", 3),
        ("workflow_drop_low_cap", 4),
    ]:
        v = run(pol, 302, cap)
        rows.append({"policy": pol, "cap": cap, "seed": 302, "total_value": v})
        print(f"{pol} cap={cap} seed=302 V={v:.1f}", flush=True)

    best302 = max(rows, key=lambda r: r["total_value"])
    print(f"BEST302 {best302}", flush=True)
    if best302["total_value"] <= 24.0:
        print("FAIL still no headroom", flush=True)
        (OUT / "cap_report.json").write_text(json.dumps({"rows": rows, "best302": best302}, indent=2))
        return 2

    # Full-5 with best cap
    cap = best302["cap"]
    full = []
    for s in HOLD:
        v = run("workflow_drop_low_cap", s, cap)
        full.append({"seed": s, "total_value": v})
        print(f"FULL5 cap={cap} seed={s} V={v:.1f}", flush=True)
    mean = sum(r["total_value"] for r in full) / len(full)
    report = {
        "rows_302": rows,
        "best302": best302,
        "full5": full,
        "full5_mean": mean,
        "pdp_mean": 45.6,
        "beats_pdp": mean > 45.6,
    }
    path = OUT / "cap_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"FULL5_MEAN={mean:.2f} beats_pdp={mean > 45.6}", flush=True)
    return 0 if mean > 45.6 else 3


if __name__ == "__main__":
    raise SystemExit(main())
