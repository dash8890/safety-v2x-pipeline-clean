#!/usr/bin/env python3
"""Fast path: parallel congest demos → BC → parallel DDQN full-5."""
from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bc_wilab_droplow import export_weights, load_demos
from ddqn_agent import DDQNAgent

ROOT = Path(__file__).resolve().parent
WF2 = ROOT.parent
OUT = WF2 / "outputs" / "wilab_bc_congest"
BASE_CFG = WF2 / "config" / "wilab_bc_demos.json"
MATLAB = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
HOLD = [300, 301, 302, 303, 304]
TARGET = 52.0
PDP_MEAN = 45.6
N_PARALLEL = 3


def run_case(policy: str, seed: int, collect_demos: bool, weights: Path | None = None) -> tuple[int, float]:
    tag = f"{policy}_s{seed}"
    case = OUT / tag
    cfg_path = OUT / f"{tag}_cfg.json"
    cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    cfg["workflow"]["collect_demos"] = bool(collect_demos)
    cfg["workflow"]["collect_transitions"] = False
    if weights is not None:
        cfg["workflow"]["ddqn_weights_json"] = str(weights).replace("\\", "/")
        cfg["workflow"]["ddqn_eps"] = 0.0
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
    return seed, float(json.loads(sp.read_text(encoding="utf-8"))["total_value"])


def has_demos(seed: int) -> bool:
    p = OUT / f"workflow_drop_low_congest_s{seed}" / "workflow_v2" / "demos.json"
    return p.is_file() and p.stat().st_size > 50


def bc_fit(demo_root: Path) -> Path:
    obs, acts, masks = load_demos([demo_root])
    print(f"BC demos N={len(acts)} drop_frac={(acts == 4).mean():.3f}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DDQNAgent(16, 5, lr=1e-3, gamma=0.99, device=device, dueling=True)
    ckpt = ROOT / "outputs" / "ddqn_wilab_causal.pt"
    if ckpt.is_file():
        agent.load(ckpt)
    opt = torch.optim.Adam(agent.q.parameters(), lr=1e-3)
    n = len(acts)
    epochs = 40
    bs = min(128, max(16, n // 4))
    for ep in range(epochs):
        idx = np.random.permutation(n)
        total = 0.0
        steps = 0
        for i in range(0, n, bs):
            b = idx[i : i + bs]
            o = torch.as_tensor(obs[b], device=device)
            a = torch.as_tensor(acts[b], device=device)
            m = torch.as_tensor(masks[b], device=device)
            logits = agent.q(o).masked_fill(m < 0.5, -1e9)
            loss = F.cross_entropy(logits, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            with torch.no_grad():
                pred = agent.q(torch.as_tensor(obs, device=device))
                pred = pred.masked_fill(torch.as_tensor(masks, device=device) < 0.5, -1e9)
                acc = (pred.argmax(-1).cpu().numpy() == acts).mean()
            print(f"BC {ep+1}/{epochs} loss={total/max(steps,1):.4f} acc={acc:.3f}", flush=True)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    pt = out_dir / "ddqn_wilab_congest.pt"
    wj = out_dir / "ddqn_wilab_congest.weights.json"
    agent.save(pt)
    export_weights(agent, wj)
    shutil.copy2(wj, out_dir / "ddqn_wilab_causal.weights.json")
    shutil.copy2(pt, out_dir / "ddqn_wilab_causal.pt")
    print(f"installed {wj}", flush=True)
    return wj


def parallel_map(fn, seeds, label):
    out = {}
    with ThreadPoolExecutor(max_workers=N_PARALLEL) as ex:
        futs = {ex.submit(fn, s): s for s in seeds}
        for fut in as_completed(futs):
            seed, v = fut.result()
            out[seed] = v
            print(f"{label} seed={seed} V={v:.1f}", flush=True)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    need = [s for s in HOLD if not has_demos(s)]
    print(f"demos have={[s for s in HOLD if has_demos(s)]} need={need}", flush=True)
    if need:
        parallel_map(
            lambda s: run_case("workflow_drop_low_congest", s, True),
            need,
            "teacher",
        )
    for s in HOLD:
        if not has_demos(s):
            raise RuntimeError(f"missing demos seed={s}")

    print("BC fit", flush=True)
    weights = bc_fit(OUT)

    print("DDQN full-5 parallel", flush=True)
    per = parallel_map(
        lambda s: run_case("workflow_causal_ddqn", s, False, weights),
        HOLD,
        "ddqn",
    )
    mean = sum(per.values()) / len(HOLD)
    report = {
        "per_seed": per,
        "mean": mean,
        "pdp_mean": PDP_MEAN,
        "target": TARGET,
        "beats_pdp": mean > PDP_MEAN,
        "hits_target": mean >= TARGET,
        "weights": str(weights),
    }
    path = OUT / "ddqn_outcome_a.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"DDQN_FULL5_MEAN={mean:.2f} target52={mean >= TARGET}", flush=True)
    return 0 if mean >= TARGET else (1 if mean > PDP_MEAN else 2)


if __name__ == "__main__":
    raise SystemExit(main())
