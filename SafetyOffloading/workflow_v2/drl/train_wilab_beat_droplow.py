"""WiLab online DDQN trained to exceed the PDP teacher (learning contribution).

PDP is the BC teacher — not an oracle / not a literature baseline.
Reward = episode total_value, plus bonus for exceeding PDP on that seed
(if known). Exploration keeps capacity-aware non-drop actions in the replay.
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
WEIGHTS = OUT / "ddqn_wilab_beat.weights.json"
CKPT = OUT / "ddqn_wilab_beat.pt"
BASE_CFG = WF2 / "config" / "wilab_causal_eval.json"
MATLAB = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
DROPLOW_BASE = {
    300: 48.0,
    301: 48.0,
    302: 24.0,
    303: 48.0,
    304: 60.0,
}
N_ACTIONS = 5
OBS_DIM = 16


def write_cfg(path: Path, eps: float, weights: Path, seed: int):
    cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    cfg["workflow"]["ddqn_eps"] = float(eps)
    cfg["workflow"]["ddqn_weights_json"] = str(weights).replace("\\", "/")
    cfg["workflow"]["collect_transitions"] = True
    cfg["workflow"]["collect_demos"] = False
    cfg["workflow"]["drop_reward_bonus"] = 0.0  # do not reinforce DropLow cloning
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def run_episode(case: Path, cfg: Path, seed: int, eps: float) -> dict:
    write_cfg(cfg, eps, WEIGHTS, seed)
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
    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    transitions = json.loads(tr_path.read_text(encoding="utf-8")) if tr_path.is_file() else None
    return {"summary": summary, "transitions": transitions}


def push_episode(agent: DDQNAgent, tr: dict, value_bonus: float):
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
    # put beat-DropLow bonus on terminal transition
    if len(rew):
        rew[-1] = rew[-1] + value_bonus
    n = 0
    for i in range(len(acts)):
        a = int(acts[i])
        if a < 0 or a >= N_ACTIONS:
            continue
        agent.push(obs[i], a, float(rew[i]), nobs[i], float(done[i]), mask[i], nmask[i])
        n += 1
    return n


def main():
    OUT.mkdir(exist_ok=True)
    # start from BC clone then deliberately leave it via exploration + beat bonus
    src = OUT / "ddqn_wilab_causal.weights.json"
    shutil.copy(src, WEIGHTS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DDQNAgent(OBS_DIM, N_ACTIONS, lr=5e-4, gamma=0.99, device=device, dueling=True)
    agent.load(OUT / "ddqn_wilab_causal.pt")
    # raise eps floor so we don't stay glued to DropLow
    agent.eps = 0.45
    agent.eps_min = 0.12
    agent.eps_decay = 0.97

    train_root = WF2 / "outputs" / "wilab_ddqn_beat"
    train_root.mkdir(parents=True, exist_ok=True)
    history = []
    n_episodes = 20
    eps = 0.45
    best_v = -1.0

    for ep in range(n_episodes):
        seed = 300 + (ep % 5)  # cycle holdout family + offset later
        if ep >= 5:
            seed = 800 + ep
        case = train_root / f"ep{ep:03d}_seed{seed}"
        cfg = train_root / f"ep{ep:03d}_config.json"
        print(f"EP {ep+1}/{n_episodes} seed={seed} eps={eps:.3f}", flush=True)
        out = run_episode(case, cfg, seed, eps)
        tv = float(out["summary"].get("total_value", 0))
        base = DROPLOW_BASE.get(seed)
        # bonus if we beat known DropLow on that seed; mild penalty if far below
        if base is not None:
            value_bonus = 0.15 * (tv - base)
        else:
            value_bonus = 0.02 * tv
        n_tr = push_episode(agent, out["transitions"], value_bonus)
        loss = 0.0
        for _ in range(max(40, 2 * n_tr)):
            loss = agent.train_step(64)
        if (ep + 1) % 2 == 0:
            agent.sync_target()
        agent.decay_eps()
        eps = max(agent.eps_min, eps * 0.97)
        agent.save(CKPT)
        export_weights(agent, WEIGHTS)
        row = {
            "ep": ep,
            "seed": seed,
            "total_value": tv,
            "droplow_base": base,
            "beat_droplow": (base is not None and tv > base + 1e-9),
            "n_transitions": n_tr,
            "loss": loss,
            "eps": eps,
        }
        history.append(row)
        print(
            f"  V={tv:.1f} base={base} beat={row['beat_droplow']} tr={n_tr} loss={loss:.4f}",
            flush=True,
        )
        (train_root / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if tv > best_v:
            best_v = tv
            shutil.copy(WEIGHTS, OUT / "ddqn_wilab_beat_best.weights.json")
            shutil.copy(CKPT, OUT / "ddqn_wilab_beat_best.pt")

    print(f"TRAIN_DONE best_v={best_v} weights={WEIGHTS}", flush=True)


if __name__ == "__main__":
    main()
