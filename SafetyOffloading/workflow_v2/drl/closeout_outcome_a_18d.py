#!/usr/bin/env python3
"""Close Outcome A with a Markov-complete 18-D learned admission state."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train_wilab_convergence as w
from bc_wilab_droplow import export_weights, load_demos
from ddqn_agent import DDQNAgent

OUT = w.WF2 / "outputs" / "wilab_outcome_a_18d"
BASE16_PT = w.OUT / "ddqn_wilab_outcome_a_fixed_best.pt"
PLACE18 = w.OUT / "ddqn_wilab_placement_18d.weights.json"
ADMIT18 = w.OUT / "ddqn_wilab_admission_18d.weights.json"
ADMIT18_PT = w.OUT / "ddqn_wilab_admission_18d.pt"
SEEDS = [300, 301, 302, 303, 304]


def expanded_agent() -> DDQNAgent:
    old = DDQNAgent(16, 5, device="cpu", dueling=True)
    old.load(BASE16_PT)
    new = DDQNAgent(18, 5, device="cpu", dueling=True)
    old_sd, new_sd = old.q.state_dict(), new.q.state_dict()
    for key in new_sd:
        if key == "fc1.weight":
            new_sd[key].zero_()
            new_sd[key][:, :16] = old_sd[key]
        else:
            new_sd[key].copy_(old_sd[key])
    new.q.load_state_dict(new_sd)
    new.sync_target()
    return new


def collect_teacher(seed: int):
    case = OUT / "teacher" / f"seed{seed}"
    cfg = OUT / "teacher" / f"seed{seed}_cfg.json"
    result = w.run_wilab(case, cfg, seed, 0.0, False, 240, 0.0,
                         {"collect_demos": True, "ddqn_teacher_admission": True,
                          "ddqn_admission_weights_json": str(PLACE18).replace("\\", "/"),
                          "ddqn_placement_weights_json": str(PLACE18).replace("\\", "/")},
                         policy="workflow_causal_ddqn")
    return seed, float(result["summary"]["total_value"])


def fit_admission(agent: DDQNAgent, roots: list[Path]) -> tuple[int, float, float]:
    obs, acts, masks = load_demos(roots)
    if obs.shape[1] != 18:
        raise RuntimeError(f"expected fresh 18-D demos, got {obs.shape}")
    keep = np.isclose(obs[:, 6], 0.0)
    obs, acts, masks = obs[keep], acts[keep], masks[keep]
    o, a, m = torch.as_tensor(obs), torch.as_tensor(acts), torch.as_tensor(masks)
    opt = torch.optim.Adam(agent.q.parameters(), lr=2e-4, weight_decay=1e-6)
    for epoch in range(1200):
        logits = agent.q(o).masked_fill(m < 0.5, -1e9)
        loss = F.cross_entropy(logits, a)
        opt.zero_grad(); loss.backward(); opt.step()
        acc = float((logits.argmax(1) == a).float().mean())
        if epoch % 100 == 0:
            print(f"fit epoch={epoch} loss={loss.item():.6f} acc={acc:.4f}", flush=True)
        if epoch >= 150 and acc == 1.0 and loss.item() < 5e-4:
            break
    agent.sync_target()
    return len(acts), acc, float(loss.item())


def main() -> int:
    np.random.seed(18); torch.manual_seed(18)
    (OUT / "teacher").mkdir(parents=True, exist_ok=True)
    placement = expanded_agent()
    export_weights(placement, PLACE18)
    data = json.loads(PLACE18.read_text()); data["obs_dim"] = 18
    PLACE18.write_text(json.dumps(data), encoding="utf-8")
    w.WEIGHTS = PLACE18
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        teacher = dict(pool.map(collect_teacher, SEEDS))
    print(f"teacher={teacher}", flush=True)

    admission = expanded_agent()
    n, acc, loss = fit_admission(admission, [OUT / "teacher"])
    admission.save(ADMIT18_PT)
    export_weights(admission, ADMIT18)
    data = json.loads(ADMIT18.read_text()); data["obs_dim"] = 18
    ADMIT18.write_text(json.dumps(data), encoding="utf-8")

    w.TRAIN = OUT / "eval"
    w.TRAIN.mkdir(parents=True, exist_ok=True)
    w.WEIGHTS = PLACE18
    w.ADMISSION_WEIGHTS = ADMIT18
    w.PLACEMENT_WEIGHTS = PLACE18
    per = w.eval_parallel(SEEDS)
    score = float(np.mean(list(per.values())))
    report = {"passed": score >= 62.4 - 1e-9, "teacher": teacher,
              "samples": n, "accuracy": acc, "loss": loss,
              "per_seed": per, "full5": score,
              "admission_weights": str(ADMIT18), "placement_weights": str(PLACE18)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
