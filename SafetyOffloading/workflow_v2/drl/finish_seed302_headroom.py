#!/usr/bin/env python3
"""Finish remaining seed-302 headroom cases and write report."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WF2 = ROOT.parent
OUT = WF2 / "outputs" / "wilab_seed302_headroom"
BASE_CFG = WF2 / "config" / "wilab_causal_eval.json"
MATLAB = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
SEED = 302


def run(policy: str, eps: float = 0.0) -> float:
    case = OUT / f"{policy}_eps{eps:.2f}"
    cfg_path = OUT / f"{policy}_eps{eps:.2f}_cfg.json"
    cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    cfg["workflow"]["ddqn_eps"] = float(eps)
    w = ROOT / "outputs" / "ddqn_wilab_causal.weights.json"
    cfg["workflow"]["ddqn_weights_json"] = str(w).replace("\\", "/")
    cfg["workflow"]["collect_transitions"] = False
    cfg["workflow"]["collect_demos"] = False
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if case.exists():
        shutil.rmtree(case, ignore_errors=True)
    case.mkdir(parents=True, exist_ok=True)
    script = (
        f"addpath(genpath('{WF2.as_posix()}')); "
        f"try; s=runR3PilotCase('{case.as_posix()}','{policy}',{SEED},"
        f"'config','{cfg_path.as_posix()}','radio_mode','wilab_mode2'); disp(jsonencode(s)); "
        f"catch ME; disp(getReport(ME)); exit(1); end; exit(0);"
    )
    r = subprocess.run([MATLAB, "-batch", script], capture_output=True, text=True, timeout=300)
    sp = case / "workflow_v2" / "summary.json"
    if not sp.is_file():
        raise RuntimeError(f"{policy} failed rc={r.returncode}\n{(r.stdout or '')[-800:]}")
    return float(json.loads(sp.read_text(encoding="utf-8"))["total_value"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(OUT.glob("*_eps*")):
        if not d.is_dir():
            continue
        sp = d / "workflow_v2" / "summary.json"
        if not sp.is_file():
            continue
        name = d.name
        i = name.rfind("_eps")
        pol, eps = name[:i], float(name[i + 4 :])
        v = float(json.loads(sp.read_text(encoding="utf-8"))["total_value"])
        rows.append({"policy": pol, "eps": eps, "total_value": v})
        print(f"HAVE {pol} eps={eps:.2f} V={v:.1f}", flush=True)

    todo = [
        ("workflow_b4_joint", 0.0),
        ("workflow_causal_ddqn", 0.3),
        ("workflow_causal_ddqn", 0.5),
    ]
    have = {(r["policy"], r["eps"]) for r in rows}
    for pol, eps in todo:
        if (pol, eps) in have:
            print(f"SKIP {pol} eps={eps}", flush=True)
            continue
        print(f"RUN {pol} eps={eps:.2f} ...", flush=True)
        try:
            v = run(pol, eps)
            rows.append({"policy": pol, "eps": eps, "total_value": v})
            print(f"{pol} eps={eps:.2f} V={v:.1f}", flush=True)
        except Exception as e:
            print(f"FAIL {pol} eps={eps}: {e}", flush=True)

    best = max(rows, key=lambda r: r["total_value"]) if rows else None
    report = {
        "seed": SEED,
        "pdp_baseline": 24.0,
        "rows": rows,
        "best": best,
        "headroom_above_pdp": bool(best and best["total_value"] > 24.0),
    }
    path = OUT / "headroom_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"WROTE {path}", flush=True)
    verdict = "HEADROOM" if report["headroom_above_pdp"] else "NO_HEADROOM max==PDP"
    print(f"VERDICT {verdict}", flush=True)
    return 0 if report["headroom_above_pdp"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
