"""Online Double DQN training where EACH episode is a real WiLab Mode-2 run."""
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
WEIGHTS = OUT / "ddqn_wilab_online.weights.json"
CKPT = OUT / "ddqn_wilab_online.pt"
BASE_CFG = WF2 / "config" / "wilab_causal_eval.json"
MATLAB = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
N_ACTIONS = 5
OBS_DIM = 16


def write_episode_cfg(path: Path, eps: float, weights: Path, seed: int):
    cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    cfg["workflow"]["ddqn_eps"] = float(eps)
    cfg["workflow"]["ddqn_weights_json"] = str(weights).replace("\\", "/")
    cfg["workflow"]["collect_transitions"] = True
    cfg["workflow"]["collect_demos"] = False
    cfg["experiment"]["seeds"] = [int(seed)]
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def run_wilab_episode(case: Path, cfg: Path, seed: int, eps: float) -> dict:
    write_episode_cfg(cfg, eps, WEIGHTS, seed)
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
        raise RuntimeError(r.stdout[-2000:] + "\n" + r.stderr[-2000:])
    tr_path = case / "workflow_v2" / "transitions.json"
    sum_path = case / "workflow_v2" / "summary.json"
    summary = json.loads(sum_path.read_text(encoding="utf-8")) if sum_path.is_file() else {}
    transitions = json.loads(tr_path.read_text(encoding="utf-8")) if tr_path.is_file() else None
    return {"summary": summary, "transitions": transitions, "stdout_tail": r.stdout[-500:]}


def push_episode(agent: DDQNAgent, tr: dict):
    if tr is None:
        return 0
    obs = np.asarray(tr["obs"], dtype=np.float32)
    acts = np.asarray(tr["action_idx"], dtype=np.int64) - 1  # MATLAB 1-based
    rew = np.asarray(tr["reward"], dtype=np.float32)
    nobs = np.asarray(tr["next_obs"], dtype=np.float32)
    done = np.asarray(tr["done"], dtype=np.float32)
    mask = np.asarray(tr["mask"], dtype=np.float32)
    nmask = np.asarray(tr["next_mask"], dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    n = len(acts)
    for i in range(n):
        a = int(acts[i])
        if a < 0 or a >= N_ACTIONS:
            continue
        agent.push(obs[i], a, float(rew[i]), nobs[i], float(done[i]), mask[i], nmask[i])
    return n


def main():
    OUT.mkdir(exist_ok=True)
    # start from WiLab-BC weights if present, else abstract causal
    src = OUT / "ddqn_wilab_causal.weights.json"
    if not src.is_file():
        src = OUT / "ddqn_causal_dueling.weights.json"
    shutil.copy(src, WEIGHTS)
    print(f"init weights from {src}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DDQNAgent(OBS_DIM, N_ACTIONS, lr=4e-4, gamma=0.99, device=device, dueling=True)
    pt_src = OUT / "ddqn_wilab_causal.pt"
    if pt_src.is_file():
        agent.load(pt_src)
    elif (OUT / "ddqn_causal_dueling.pt").is_file():
        agent.load(OUT / "ddqn_causal_dueling.pt")

    train_root = WF2 / "outputs" / "wilab_ddqn_train"
    train_root.mkdir(parents=True, exist_ok=True)
    n_episodes = 24
    eps = 0.35
    history = []

    for ep in range(n_episodes):
        seed = 700 + ep
        case = train_root / f"ep{ep:03d}_seed{seed}"
        cfg = train_root / f"ep{ep:03d}_config.json"
        print(f"EP {ep+1}/{n_episodes} seed={seed} eps={eps:.3f} -> WiLab Mode-2", flush=True)
        out = run_wilab_episode(case, cfg, seed, eps)
        n_tr = push_episode(agent, out["transitions"])
        loss = 0.0
        for _ in range(max(20, n_tr)):
            loss = agent.train_step(64)
        if (ep + 1) % 2 == 0:
            agent.sync_target()
        agent.decay_eps()
        eps = max(0.05, eps * 0.92)
        agent.save(CKPT)
        export_weights(agent, WEIGHTS)
        tv = float(out["summary"].get("total_value", 0))
        ns = float(out["summary"].get("n_success", 0))
        history.append({"ep": ep, "seed": seed, "total_value": tv, "n_success": ns, "n_transitions": n_tr, "loss": loss, "eps": eps})
        print(f"  V={tv:.1f} N={ns:.0f} transitions={n_tr} loss={loss:.4f}", flush=True)
        (train_root / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"TRAIN_DONE ckpt={CKPT} weights={WEIGHTS}", flush=True)


if __name__ == "__main__":
    main()
