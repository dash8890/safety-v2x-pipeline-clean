#!/usr/bin/env python3
"""Watch active train terminal + case dir. Prints status every 15s. Exit 2 on HUNG."""
from __future__ import annotations

import time
from pathlib import Path

import psutil

TRAIN = Path(__file__).resolve().parents[1] / "outputs" / "wilab_ddqn_convergence"
TERM_DIR = Path(r"C:\Users\dash\.cursor\projects\c-Users-dash-Downloads-NewProject\terminals")
POLL = 15.0


def newest_train_term() -> Path | None:
    cands = []
    for p in TERM_DIR.glob("*.txt"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "train_wilab_convergence.py" not in txt:
            continue
        # only live jobs
        head = "\n".join(txt.splitlines()[:12])
        if "status: running" not in head:
            continue
        cands.append(p)
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def matlab_cpu_delta(seconds: float = 3.0) -> float:
    procs = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if (p.info["name"] or "").lower().startswith("matlab"):
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if not procs:
        return -1.0
    c0 = {}
    for p in procs:
        try:
            ct = p.cpu_times()
            c0[p.pid] = ct.user + ct.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not c0:
        return -1.0
    time.sleep(seconds)
    d = 0.0
    for p in procs:
        try:
            if p.pid in c0:
                ct = p.cpu_times()
                d += (ct.user + ct.system) - c0[p.pid]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return d


def latest_case() -> Path | None:
    if not TRAIN.is_dir():
        return None
    cases = [p for p in TRAIN.glob("train_ep*") if p.is_dir()]
    return max(cases, key=lambda p: p.stat().st_mtime) if cases else None


def main() -> int:
    print("WATCH active", flush=True)
    while True:
        term = newest_train_term()
        hint = ""
        alive_term = False
        if term and term.is_file():
            lines = term.read_text(encoding="utf-8", errors="ignore").splitlines()
            alive_term = any(l.startswith("status: running") for l in lines[:10])
            for ln in reversed(lines[-40:]):
                if ln.startswith("EP ") or "NEW BEST" in ln or "FULL5" in ln or "RETRY" in ln or "killed_" in ln:
                    hint = ln.strip()
                    break
        case = latest_case()
        nfiles = 0
        summary = False
        file_age = -1.0
        if case:
            files = [p for p in case.rglob("*") if p.is_file()]
            nfiles = len(files)
            summary = (case / "workflow_v2" / "summary.json").is_file()
            if files:
                file_age = time.time() - max(p.stat().st_mtime for p in files)
        try:
            cpu_d = matlab_cpu_delta(3.0)
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
            cpu_d = -1.0
        status = "ALIVE"
        if not alive_term and term:
            status = "TRAIN_EXITED"
        elif summary:
            status = "SUMMARY_READY"
        elif cpu_d < 0 and not summary:
            status = "NO_MATLAB"
        elif cpu_d < 0.05 and nfiles <= 1 and file_age > 75:
            status = "HUNG"
        elif cpu_d < 0.05 and file_age > 90 and nfiles > 1 and not summary:
            status = "HUNG"
        print(
            f"{status} case={case.name if case else None} nfiles={nfiles} "
            f"cpu_d3={cpu_d:.2f} file_age={file_age:.0f}s | {hint}",
            flush=True,
        )
        if status == "HUNG":
            return 2
        if status == "TRAIN_EXITED":
            return 1
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
