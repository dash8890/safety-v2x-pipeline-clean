#!/usr/bin/env python3
"""Train DDQN on value-aware diamond-4 offloading MDP; evaluate vs heuristics.

Primary metric: total_value of deadline-valid workflows (safety-priority utility).
Secondary: n_success count.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ddqn_agent import DDQNAgent
from workflow_mdp import (
    N_ACTIONS,
    WorkflowMDP,
    policy_b0,
    policy_b2,
    policy_b3,
    policy_b4,
    policy_value_greedy,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def eval_policy(name, policy_fn, seeds):
    rows = []
    for s in seeds:
        env = WorkflowMDP(seed=s)
        info = env.run_policy(policy_fn, seed=s)
        rows.append(info)
    succ = np.array([r["n_success"] for r in rows], dtype=float)
    val = np.array([r["total_value"] for r in rows], dtype=float)
    return {
        "policy": name,
        "mean_success": float(succ.mean()),
        "std_success": float(succ.std()),
        "mean_value": float(val.mean()),
        "std_value": float(val.std()),
        "per_seed_success": succ.tolist(),
        "per_seed_value": val.tolist(),
        "mean_remote": float(np.mean([r["n_remote"] for r in rows])),
    }


def make_agent_policy(agent: DDQNAgent):
    def fn(obs, mask, env):
        return agent.act(obs, mask, explore=False)

    return fn


def train(n_episodes: int = 1000, eval_every: int = 50, device: str = "cpu"):
    env = WorkflowMDP(seed=0)
    agent = DDQNAgent(env.obs_dim, N_ACTIONS, lr=8e-4, gamma=0.99, device=device)
    hist = []
    best_mean = -1.0
    best_path = OUT / "ddqn_best.pt"
    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        seed = 1000 + ep
        obs = env.reset(seed)
        done = False
        ep_r = 0.0
        info = {"n_success": 0, "total_value": 0.0}
        while not done:
            mask = env.action_mask()
            a = agent.act(obs, mask, explore=True)
            nobs, r, done, info = env.step(a)
            nmask = env.action_mask() if not done else np.ones(N_ACTIONS, dtype=bool)
            agent.push(obs, a, r, nobs, float(done), mask, nmask)
            agent.train_step(96)
            obs = nobs
            ep_r += r
        agent.decay_eps()
        if ep % 8 == 0:
            agent.sync_target()
        if ep % eval_every == 0 or ep == n_episodes:
            ev = eval_policy("ddqn", make_agent_policy(agent), list(range(1, 16)))
            hist.append(
                {
                    "episode": ep,
                    "eps": agent.eps,
                    "ep_reward": ep_r,
                    "mean_value": ev["mean_value"],
                    "mean_success": ev["mean_success"],
                }
            )
            print(
                f"ep={ep:4d} eps={agent.eps:.3f} "
                f"trainV={info['total_value']:.0f} "
                f"evalV={ev['mean_value']:.2f} evalN={ev['mean_success']:.2f} "
                f"[{time.time()-t0:.0f}s]",
                flush=True,
            )
            if ev["mean_value"] > best_mean:
                best_mean = ev["mean_value"]
                agent.save(str(best_path))
    agent.load(str(best_path))
    return agent, hist


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    base_seeds = list(range(1, 41))
    # Primary baselines: non-value-aware (established B0–B4 line)
    baselines = [
        eval_policy("B0_local", policy_b0, base_seeds),
        eval_policy("B2_queue", policy_b2, base_seeds),
        eval_policy("B3_mobility", policy_b3, base_seeds),
        eval_policy("B4_joint", policy_b4, base_seeds),
    ]
    ref = eval_policy("ValueGreedy_ref", policy_value_greedy, base_seeds)
    print("BASELINES (40 seeds):", flush=True)
    for b in baselines:
        print(
            f"  {b['policy']:12s} value={b['mean_value']:.2f}±{b['std_value']:.2f} "
            f"n={b['mean_success']:.2f}±{b['std_success']:.2f}",
            flush=True,
        )
    print(
        f"  {ref['policy']:12s} value={ref['mean_value']:.2f}±{ref['std_value']:.2f} "
        f"n={ref['mean_success']:.2f} (reference heuristic, not win-criterion)",
        flush=True,
    )

    agent, hist = train(n_episodes=900, eval_every=30, device=device)

    final_seeds = list(range(1, 41))
    drl = eval_policy("DDQN", make_agent_policy(agent), final_seeds)
    holdout = eval_policy("DDQN_holdout", make_agent_policy(agent), list(range(300, 340)))

    print("DDQN FINAL (40 seeds):", flush=True)
    print(
        f"  value={drl['mean_value']:.2f}±{drl['std_value']:.2f} "
        f"n={drl['mean_success']:.2f}±{drl['std_success']:.2f}",
        flush=True,
    )
    print(
        f"DDQN HOLDOUT: value={holdout['mean_value']:.2f}±{holdout['std_value']:.2f} "
        f"n={holdout['mean_success']:.2f}",
        flush=True,
    )

    best_base = max(baselines, key=lambda x: x["mean_value"])
    delta_v = drl["mean_value"] - best_base["mean_value"]
    delta_h = holdout["mean_value"] - best_base["mean_value"]
    # Substantial: >=25% relative value gain on holdout vs best B0–B4
    rel = delta_h / max(best_base["mean_value"], 1e-6)
    win = bool(delta_h >= 3.0 and rel >= 0.25)

    result = {
        "baselines": baselines,
        "ddqn": drl,
        "ddqn_holdout": holdout,
        "best_baseline": best_base["policy"],
        "value_greedy_reference": ref,
        "delta_value_vs_best": delta_v,
        "delta_holdout_vs_best": delta_h,
        "relative_holdout_gain": rel,
        "substantial_win": win,
        "n_workflows": int(len(WorkflowMDP().cfg["arrivals"])),
        "train_history": hist,
        "mdp": {
            "family": "deadline-aware prioritized DAG offloading (diamond-4)",
            "algorithm": "Double DQN + legal action mask",
            "reward": "workflow value on deadline-valid completion; expire penalty",
            "values": "high=5 (p=0.35) low=1",
            "contact": "helper-2 exit 0.11s return 0.58s with stranding",
            "literature": [
                "Farimani et al., ESWA 2024 — DQN deadline-aware vehicular offloading",
                "Pang et al., ComCom 2024 — DDQN dependency-aware VEC",
                "VTC 2024 Prioritized DQN — priority-class deadline completion",
            ],
        },
    }
    out_json = OUT / "drl_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(OUT / "drl_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "mean_value", "std_value", "mean_success", "std_success"])
        for b in baselines + [ref, drl, holdout]:
            w.writerow(
                [
                    b["policy"],
                    b["mean_value"],
                    b["std_value"],
                    b["mean_success"],
                    b["std_success"],
                ]
            )

    print(f"WROTE {out_json}", flush=True)
    print(
        f"VERDICT substantial_win={win} holdout_delta={delta_h:.2f} rel={rel:.1%} vs {best_base['policy']}",
        flush=True,
    )
    return 0 if win else 2


if __name__ == "__main__":
    raise SystemExit(main())
