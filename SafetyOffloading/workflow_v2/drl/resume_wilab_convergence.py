#!/usr/bin/env python3
"""Resume WiLab convergence from BEST + last completed episode.

Never restart from EP1 after a hang. Usage:
  python resume_wilab_convergence.py
Reads convergence_history.csv, keeps BEST weights, continues episode count.
"""
from __future__ import annotations

import csv
import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HIST = ROOT.parent / "outputs" / "wilab_ddqn_convergence" / "convergence_history.csv"
BEST_PT = ROOT / "outputs" / "ddqn_wilab_converged_best.pt"
BEST_W = ROOT / "outputs" / "ddqn_wilab_converged_best.weights.json"


def last_episode() -> int:
    if not HIST.is_file():
        return 0
    rows = list(csv.DictReader(HIST.open(encoding="utf-8")))
    if not rows:
        return 0
    return max(int(r["episode"]) for r in rows)


def main():
    ep = last_episode()
    if not BEST_PT.is_file() or not BEST_W.is_file():
        raise SystemExit(f"missing BEST checkpoint: {BEST_PT} / {BEST_W}")
    start = ep + 1
    print(f"RESUME from episode {start} (last finished={ep})", flush=True)
    os.environ["WILAB_RESUME_EP"] = str(start)
    # train_wilab_convergence reads WILAB_RESUME_EP if set
    runpy.run_path(str(ROOT / "train_wilab_convergence.py"), run_name="__main__")


if __name__ == "__main__":
    main()
