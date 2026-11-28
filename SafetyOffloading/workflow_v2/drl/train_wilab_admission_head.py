#!/usr/bin/env python3
"""Fit the admission head to the WiLab congest-teacher decision surface."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from bc_wilab_droplow import export_weights, load_demos
from ddqn_agent import DDQNAgent
from train_wilab_convergence import OUT, WF2


def main() -> int:
    np.random.seed(24)
    torch.manual_seed(24)
    obs, acts, masks = load_demos([WF2 / "outputs" / "wilab_bc_congest"])
    # stage type is observation component 7 (zero-based 6); admission is A only.
    keep = np.isclose(obs[:, 6], 0.0)
    obs, acts, masks = obs[keep], acts[keep], masks[keep]
    agent = DDQNAgent(16, 5, lr=1e-4, gamma=0.99, device="cpu", dueling=True)
    agent.load(OUT / "ddqn_wilab_outcome_a_fixed_best.pt")
    opt = torch.optim.Adam(agent.q.parameters(), lr=3e-4, weight_decay=1e-6)
    o = torch.as_tensor(obs)
    a = torch.as_tensor(acts)
    m = torch.as_tensor(masks)
    for epoch in range(1200):
        logits = agent.q(o).masked_fill(m < 0.5, -1e9)
        loss = F.cross_entropy(logits, a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == 1199:
            acc = float((logits.argmax(1) == a).float().mean())
            print(f"epoch={epoch:04d} loss={loss.item():.6f} admission_acc={acc:.4f}", flush=True)
        if epoch >= 100 and float((logits.argmax(1) == a).float().mean()) == 1.0 and loss.item() < 1e-3:
            break
    agent.sync_target()
    agent.save(OUT / "ddqn_wilab_admission_teacher.pt")
    export_weights(agent, OUT / "ddqn_wilab_admission_teacher.weights.json")
    print(f"samples={len(acts)} final_acc={float((agent.q(o).masked_fill(m < 0.5,-1e9).argmax(1)==a).float().mean()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
