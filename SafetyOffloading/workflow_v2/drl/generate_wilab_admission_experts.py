#!/usr/bin/env python3
"""Generate admission specialists from cumulative corrected WiLab replay."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

import train_wilab_convergence as wilab
from bc_wilab_droplow import export_weights
from ddqn_agent import DDQNAgent
from train_wilab_outcome_a_fixed import imitate_high_return_dagger

OUT = wilab.WF2 / "outputs" / "wilab_admission_experts"
WEIGHTS = wilab.OUT / "ddqn_wilab_admission_work.weights.json"
BASE_PT = wilab.OUT / "ddqn_wilab_outcome_a_fixed_best.pt"


def main() -> int:
    np.random.seed(20261124)
    torch.manual_seed(20261124)
    OUT.mkdir(parents=True, exist_ok=True)
    wilab.TRAIN = OUT
    wilab.WEIGHTS = WEIGHTS
    agent = DDQNAgent(16, 5, lr=1e-4, gamma=0.99,
                      device="cuda" if torch.cuda.is_available() else "cpu", dueling=True)
    agent.load(BASE_PT)
    imitate_high_return_dagger(agent)
    for ep in range(1, 9):
        beta = max(0.05, 0.35 * (0.88 ** (ep - 1)))
        # Alternate the hard case with nearby independent training worlds.
        seed = [302, 305, 302, 306, 302, 307, 302, 308][ep - 1]
        export_weights(agent, WEIGHTS)
        result = wilab.run_wilab(OUT / f"train_ep{ep:03d}_seed{seed}",
                                 OUT / f"train_ep{ep:03d}_cfg.json",
                                 seed, 0.0, True, 240, beta)
        tr = result["transitions"]
        value = float(result["summary"]["total_value"])
        logged = float(np.sum(np.asarray(tr["reward"], np.float32)))
        n, ret = wilab.push(agent, tr, terminal_bonus=value - logged)
        loss = 0.0
        for _ in range(max(32, min(96, n))):
            loss = float(agent.train_step(64))
        agent.sync_target()
        expert_w = wilab.OUT / f"ddqn_wilab_admission_ep{ep:02d}.weights.json"
        expert_pt = wilab.OUT / f"ddqn_wilab_admission_ep{ep:02d}.pt"
        export_weights(agent, expert_w)
        agent.save(expert_pt)
        print(json.dumps({"episode": ep, "seed": seed, "value": value,
                          "return": ret, "n": n, "loss": loss,
                          "weights": str(expert_w)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
