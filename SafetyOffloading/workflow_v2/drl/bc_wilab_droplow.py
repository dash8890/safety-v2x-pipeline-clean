"""Behavioral-clone causal dueling DDQN from WiLab PDP (Always-Drop-Low) demos.

PDP is the BC teacher only — not an optimization oracle, not a literature baseline.
Produces the BC-only warm-start used before RL fine-tuning.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ddqn_agent import DDQNAgent

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
N_ACTIONS = 5
OBS_DIM = 16


def load_demos(demo_roots: list[Path]):
    obs, acts, masks = [], [], []
    for root in demo_roots:
        for p in root.glob("**/demos.json"):
            d = json.loads(p.read_text(encoding="utf-8"))
            o = np.asarray(d["obs"], dtype=np.float32)
            a = np.asarray(d["action_idx"], dtype=np.int64) - 1  # MATLAB 1-based -> 0-based
            m = np.asarray(d["mask"], dtype=np.float32)
            if o.ndim == 1:
                o = o.reshape(1, -1)
                a = a.reshape(1)
                m = m.reshape(1, -1)
            # keep only valid action indices
            keep = (a >= 0) & (a < N_ACTIONS)
            obs.append(o[keep])
            acts.append(a[keep])
            masks.append(m[keep])
    if not obs:
        raise FileNotFoundError("no demos.json found")
    return (
        np.concatenate(obs, axis=0),
        np.concatenate(acts, axis=0),
        np.concatenate(masks, axis=0),
    )


def export_weights(agent: DDQNAgent, path: Path):
    sd = agent.q.state_dict()
    keymap = {
        "fc1.weight": "fc1_weight",
        "fc1.bias": "fc1_bias",
        "fc2.weight": "fc2_weight",
        "fc2.bias": "fc2_bias",
        "v.weight": "v_weight",
        "v.bias": "v_bias",
        "a.weight": "a_weight",
        "a.bias": "a_bias",
    }
    weights = {keymap[k]: v.detach().cpu().numpy().tolist() for k, v in sd.items()}
    path.write_text(
        json.dumps(
            {"dueling": True, "obs_dim": OBS_DIM, "n_actions": N_ACTIONS, "hidden": 256, "weights": weights}
        ),
        encoding="utf-8",
    )


def main():
    demo_root = ROOT.parent / "outputs" / "wilab_bc_demos"
    obs, acts, masks = load_demos([demo_root])
    print(f"demos N={len(acts)} drop_frac={(acts==4).mean():.3f}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DDQNAgent(OBS_DIM, N_ACTIONS, lr=1e-3, gamma=0.99, device=device, dueling=True)
    # warm from abstract causal ckpt if present
    ckpt = OUT / "ddqn_causal_dueling.pt"
    if ckpt.is_file():
        agent.load(ckpt)
        print(f"loaded {ckpt}", flush=True)

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
            logits = agent.q(o)
            logits = logits.masked_fill(m < 0.5, -1e9)
            loss = F.cross_entropy(logits, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            with torch.no_grad():
                pred = agent.q(torch.as_tensor(obs, device=device))
                pred = pred.masked_fill(torch.as_tensor(masks, device=device) < 0.5, -1e9)
                acc = (pred.argmax(-1).cpu().numpy() == acts).mean()
            print(f"epoch {ep+1}/{epochs} loss={total/max(steps,1):.4f} acc={acc:.3f}", flush=True)

    OUT.mkdir(exist_ok=True)
    pt = OUT / "ddqn_wilab_causal.pt"
    wj = OUT / "ddqn_wilab_causal.weights.json"
    agent.save(pt)
    export_weights(agent, wj)
    print(f"wrote {pt}")
    print(f"wrote {wj}")


if __name__ == "__main__":
    main()
