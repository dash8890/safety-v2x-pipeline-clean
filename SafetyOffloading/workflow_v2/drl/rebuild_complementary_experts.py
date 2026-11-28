#!/usr/bin/env python3
"""Rebuild the complementary checkpoints observed in the first corrected run."""
from __future__ import annotations

import numpy as np
import torch

import train_wilab_convergence as w
from bc_wilab_droplow import export_weights
from ddqn_agent import DDQNAgent
from train_wilab_outcome_a_fixed import imitate_high_return_dagger

OUT = w.WF2 / "outputs" / "wilab_complementary_experts"
WORK = w.OUT / "ddqn_wilab_complementary_work.weights.json"
BASE_PT = w.OUT / "ddqn_wilab_outcome_a_fixed_best.pt"


def main() -> int:
    np.random.seed(20261124)
    torch.manual_seed(20261124)
    OUT.mkdir(parents=True, exist_ok=True)
    w.TRAIN, w.WEIGHTS = OUT, WORK
    # CPU avoids a Windows CUDA teardown fault when MATLAB is spawned repeatedly.
    agent = DDQNAgent(16, 5, lr=1e-4, gamma=0.99, device="cpu", dueling=True)
    agent.load(BASE_PT)
    imitate_high_return_dagger(agent)
    for ep, seed in enumerate(range(305, 313), start=1):
        beta = max(0.05, 0.35 * (0.88 ** (ep - 1)))
        export_weights(agent, WORK)
        result = w.run_wilab(OUT / f"ep{ep:02d}_seed{seed}", OUT / f"ep{ep:02d}_cfg.json",
                             seed, 0.0, True, 240, beta)
        n, ret = w.push(agent, result["transitions"], terminal_bonus=0.0)
        loss = 0.0
        for _ in range(max(24, min(80, n))):
            loss = float(agent.train_step(64))
        agent.sync_target()
        export_weights(agent, w.OUT / f"ddqn_wilab_complementary_ep{ep:02d}.weights.json")
        print(f"ep={ep} seed={seed} V={result['summary']['total_value']} ret={ret:.1f} loss={loss:.3f}", flush=True)
        if ep == 4:
            # The full-five gate rejected ep4 in the original run.
            agent.load(BASE_PT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
