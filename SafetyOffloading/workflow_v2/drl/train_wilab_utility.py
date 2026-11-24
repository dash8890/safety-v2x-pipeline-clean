"""WiLab online DDQN: maximize deadline-valid utility.

Principal baseline = ValueGreedy. PDP teacher is for learning-contribution analysis only.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from ddqn_agent import DDQNAgent
from bc_wilab_droplow import export_weights

ROOT = Path(__file__).resolve().parent
WF2 = ROOT.parent
OUT = ROOT / "outputs"
WEIGHTS = OUT / "ddqn_wilab_utility.weights.json"
CKPT = OUT / "ddqn_wilab_utility.pt"
BEST_W = OUT / "ddqn_wilab_utility_best.weights.json"
BEST_PT = OUT / "ddqn_wilab_utility_best.pt"
BASE_CFG = WF2 / "config" / "wilab_causal_eval.json"
MATLAB = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
N_ACTIONS = 5
OBS_DIM = 16


def write_cfg(path: Path, eps: float, seed: int):
    cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    cfg["workflow"]["ddqn_eps"] = float(eps)
    cfg["workflow"]["ddqn_weights_json"] = str(WEIGHTS).replace("\\", "/")
    cfg["workflow"]["collect_transitions"] = True
    cfg["workflow"]["collect_demos"] = False
    cfg["workflow"]["drop_reward_bonus"] = 0.0
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def run_episode(case: Path, cfg: Path, seed: int, eps: float) -> dict:
    write_cfg(cfg, eps, seed)
    if case.exists():
        shutil.rmtree(case, ignore_errors=True)
    case.mkdir(parents=True, exist_ok=True)
    script = (
        f"addpath(genpath('{WF2.as_posix()}')); "
        f"try; s=runR3PilotCase('{case.as_posix()}','workflow_causal_ddqn',{seed},"
        f"'config','{cfg.as_posix()}','radio_mode','wilab_mode2'); disp(jsonencode(s)); "
        f"catch ME; disp(getReport(ME)); exit(1); end; exit(0);"
    )
    r = subprocess.run([MATLAB, "-batch", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout or "")[-1500:] + "\n" + (r.stderr or "")[-1500:])
    tr_path = case / "workflow_v2" / "transitions.json"
    sum_path = case / "workflow_v2" / "summary.json"
    return {
        "summary": json.loads(sum_path.read_text(encoding="utf-8")),
        "transitions": json.loads(tr_path.read_text(encoding="utf-8")) if tr_path.is_file() else None,
    }


def high_value_success(case: Path) -> int:
    csv = case / "workflow_v2" / "workflows.csv"
    if not csv.is_file():
        return 0
    import csv as csvmod

    n = 0
    with csv.open(newline="", encoding="utf-8") as f:
        for row in csvmod.DictReader(f):
            if row.get("status") == "SUCCEEDED" and float(row.get("value", 0)) >= 11.5:
                n += 1
    return n


def push_episode(agent: DDQNAgent, tr: dict, terminal_bonus: float):
    if tr is None:
        return 0
    obs = np.asarray(tr["obs"], np.float32)
    acts = np.asarray(tr["action_idx"], np.int64) - 1
    rew = np.asarray(tr["reward"], np.float32).copy()
    nobs = np.asarray(tr["next_obs"], np.float32)
    done = np.asarray(tr["done"], np.float32)
    mask = np.asarray(tr["mask"], np.float32)
    nmask = np.asarray(tr["next_mask"], np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if len(rew):
        rew[-1] = rew[-1] + terminal_bonus
    n = 0
    for i in range(len(acts)):
        a = int(acts[i])
        if 0 <= a < N_ACTIONS:
            agent.push(obs[i], a, float(rew[i]), nobs[i], float(done[i]), mask[i], nmask[i])
            n += 1
    return n


def main():
    OUT.mkdir(exist_ok=True)
    # init from current best WiLab weights but do not BC toward DropLow again
    src = OUT / "ddqn_wilab_causal.weights.json"
    shutil.copy(src, WEIGHTS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DDQNAgent(OBS_DIM, N_ACTIONS, lr=4e-4, gamma=0.99, device=device, dueling=True)
    agent.load(OUT / "ddqn_wilab_causal.pt")
    agent.eps = 0.40
    agent.eps_min = 0.08
    agent.eps_decay = 0.96

    # stop competing beat-droplow job conceptually: this run owns utility objective
    train_root = WF2 / "outputs" / "wilab_ddqn_utility"
    train_root.mkdir(parents=True, exist_ok=True)
    history = []
    best_v = -1.0
    eps = 0.40
    n_episodes = 20

    for ep in range(n_episodes):
        seed = 900 + ep  # fresh seeds; holdout 300-304 reserved for eval
        case = train_root / f"ep{ep:03d}_seed{seed}"
        cfg = train_root / f"ep{ep:03d}_config.json"
        print(f"EP {ep+1}/{n_episodes} seed={seed} eps={eps:.3f} objective=utility", flush=True)
        out = run_episode(case, cfg, seed, eps)
        tv = float(out["summary"].get("total_value", 0))
        nh = high_value_success(case)
        # terminal shaping: total utility + extra weight on high-value completions
        terminal_bonus = 0.05 * tv + 0.5 * nh
        n_tr = push_episode(agent, out["transitions"], terminal_bonus)
        loss = 0.0
        for _ in range(max(50, 2 * n_tr)):
            loss = agent.train_step(64)
        if (ep + 1) % 2 == 0:
            agent.sync_target()
        agent.decay_eps()
        eps = max(agent.eps_min, eps * 0.96)
        agent.save(CKPT)
        export_weights(agent, WEIGHTS)
        row = {
            "ep": ep,
            "seed": seed,
            "total_value": tv,
            "n_high_success": nh,
            "n_transitions": n_tr,
            "loss": loss,
            "eps": eps,
        }
        history.append(row)
        print(f"  V={tv:.1f} high_succ={nh} tr={n_tr} loss={loss:.4f}", flush=True)
        (train_root / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if tv > best_v:
            best_v = tv
            shutil.copy(WEIGHTS, BEST_W)
            shutil.copy(CKPT, BEST_PT)
            print(f"  NEW_BEST V={best_v}", flush=True)

    print(f"TRAIN_DONE best_v={best_v} best={BEST_W}", flush=True)


if __name__ == "__main__":
    main()
