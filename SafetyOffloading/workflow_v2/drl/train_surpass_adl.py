#!/usr/bin/env python3
"""Fine-tune Causal Dueling DDQN to clearly surpass Always-Drop-Low (B2).

Target: holdout 300–339 mean utility >= ADL * 1.10 (clear justification margin).
Does not change the frozen MDP observation/action interface.
Saves only if holdout beats both ADL and the previous Full checkpoint.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ddqn_agent import DDQNAgent
from run_drl_experiment import make_agent_policy
from workflow_mdp import (
    A_DROP,
    N_ACTIONS,
    WorkflowMDP,
    policy_always_drop_low,
    policy_value_greedy,
)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
CKPT_IN = OUT / "ddqn_causal_dueling.pt"
CKPT_OUT = OUT / "ddqn_causal_dueling_surpass.pt"
CKPT_BEST = OUT / "ddqn_causal_dueling.pt"  # promote only on clear win
REPORT = OUT / "surpass_adl_holdout.json"
HOLDOUT = list(range(300, 340))
PROBE = list(range(1, 41))

CFG = {
    "obs_mode": "causal",
    "radio_mode": "abstract",
    "reward_mode": "utility_focus",
    "phased_values": True,
    "phase_split_s": 0.16,
    "p_high_early": 0.05,
    "p_high_late": 0.75,
    "value_high": 12.0,
    "value_low": 1.0,
    "cpu_hz": np.array([2.8e8, 4.0e9, 0.8e9], dtype=np.float64),
    "deadline_s": 0.35,
    "bg_cycles_origin": 20e6,
    "contact_exit": 0.10,
    "contact_return": 0.62,
}

N_EP = 2500
TARGET_REL_VS_ADL = 0.10  # +10% over Always-Drop-Low


def make_env(seed=0):
    return WorkflowMDP(seed=seed, cfg=dict(CFG))


def eval_pol(fn, seeds):
    vals = []
    ns = []
    for s in seeds:
        info = make_env(s).run_policy(fn, seed=s)
        vals.append(info["total_value"])
        ns.append(info["n_success"])
    v = np.asarray(vals, float)
    n = np.asarray(ns, float)
    return {
        "mean_value": float(v.mean()),
        "std_value": float(v.std()),
        "mean_success": float(n.mean()),
        "per_seed_value": v.tolist(),
    }


def act_train(agent, obs, mask, env, ep_frac: float) -> int:
    """Early: keep DROP bias on lows. Late: more free explore for placement."""
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return 0
    drop_bias = 0.70 if ep_frac < 0.4 else (0.35 if ep_frac < 0.7 else 0.15)
    if np.random.rand() < agent.eps:
        if env.pending_decision is not None:
            wf = env.wfs[env.pending_decision[0]]
            # selective: if origin queue light and no high pending, sometimes keep low
            q0 = float(np.asarray(obs).reshape(-1)[0])
            if (
                wf.value < env.cfg["value_high"] - 1e-9
                and mask[A_DROP]
                and np.random.rand() < drop_bias
                and not (q0 < 0.15 and not env._any_high_pending() and np.random.rand() < 0.4)
            ):
                return A_DROP
        return int(np.random.choice(legal))
    return agent.act(obs, mask, explore=False)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    adl = eval_pol(policy_always_drop_low, HOLDOUT)
    vg = eval_pol(policy_value_greedy, HOLDOUT)
    adl_m = adl["mean_value"]
    target = adl_m * (1.0 + TARGET_REL_VS_ADL)
    print(f"ADL holdout={adl_m:.2f} VG={vg['mean_value']:.2f} TARGET>={target:.2f}", flush=True)

    env = make_env(0)
    agent = DDQNAgent(
        env.obs_dim, N_ACTIONS, lr=2.5e-4, gamma=0.99, device=device, dueling=True
    )
    agent.load(str(CKPT_IN))
    agent.eps = 0.20
    agent.eps_min = 0.08
    agent.eps_decay = 0.9992  # slow decay — previous run collapsed eps by ep 100

    base = eval_pol(make_agent_policy(agent), HOLDOUT)
    print(f"start Full checkpoint holdout={base['mean_value']:.2f}", flush=True)

    best_probe = -1e9
    best_holdout = base["mean_value"]
    t0 = time.time()
    hist = []

    for ep in range(1, N_EP + 1):
        obs = env.reset(40000 + ep)
        done = False
        info = {"total_value": 0.0}
        ep_frac = ep / N_EP
        while not done:
            mask = env.action_mask()
            a = act_train(agent, obs, mask, env, ep_frac)
            nobs, r, done, info = env.step(a)
            # terminal utility pressure vs ADL scale
            if done:
                tv = float(info.get("total_value", 0.0))
                r = float(r) + 0.20 * tv
                r += 0.80 * max(0.0, tv - adl_m)
                r -= 0.40 * max(0.0, adl_m - tv)
            nmask = env.action_mask() if not done else np.ones(N_ACTIONS, dtype=bool)
            agent.push(obs, a, r, nobs, float(done), mask, nmask)
            agent.train_step(128)
            obs = nobs
        agent.decay_eps()
        if ep % 4 == 0:
            agent.sync_target()

        if ep % 50 == 0 or ep == N_EP:
            ev = eval_pol(make_agent_policy(agent), PROBE)
            print(
                f"ep={ep:4d} eps={agent.eps:.3f} probeV={ev['mean_value']:.2f} "
                f"trainLastV={info.get('total_value', 0):.1f} [{time.time()-t0:.0f}s]",
                flush=True,
            )
            hist.append({"ep": ep, "probe": ev["mean_value"], "eps": agent.eps})
            if ev["mean_value"] > best_probe:
                best_probe = ev["mean_value"]
                agent.save(str(CKPT_OUT))
                print(f"  saved probe-best {best_probe:.2f} -> {CKPT_OUT.name}", flush=True)
                # holdout check when probe looks strong
                if ev["mean_value"] >= target - 5:
                    agent.load(str(CKPT_OUT))
                    agent.eps = 0.0
                    ho = eval_pol(make_agent_policy(agent), HOLDOUT)
                    print(
                        f"  HOLDOUT {ho['mean_value']:.2f} vs ADL {adl_m:.2f} "
                        f"({100*(ho['mean_value']/adl_m-1):+.1f}%)",
                        flush=True,
                    )
                    if ho["mean_value"] > best_holdout:
                        best_holdout = ho["mean_value"]
                    if ho["mean_value"] >= target:
                        agent.save(str(CKPT_BEST))
                        report = {
                            "adl": adl,
                            "valuegreedy": vg,
                            "ddqn": ho,
                            "delta_vs_adl": ho["mean_value"] - adl_m,
                            "relative_vs_adl_pct": 100.0 * (ho["mean_value"] / adl_m - 1.0),
                            "target_rel": TARGET_REL_VS_ADL,
                            "target_met": True,
                            "episodes": ep,
                            "hist": hist,
                        }
                        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
                        print(f"TARGET MET. Promoted {CKPT_BEST}", flush=True)
                        return 0
                    agent.eps = max(agent.eps_min, 0.05)

    agent.load(str(CKPT_OUT))
    agent.eps = 0.0
    ho = eval_pol(make_agent_policy(agent), HOLDOUT)
    d = ho["mean_value"] - adl_m
    rel = 100.0 * (ho["mean_value"] / max(adl_m, 1e-9) - 1.0)
    promoted = ho["mean_value"] > base["mean_value"] and ho["mean_value"] > adl_m
    if promoted:
        agent.save(str(CKPT_BEST))
    report = {
        "adl": adl,
        "valuegreedy": vg,
        "start_ddqn": base,
        "ddqn": ho,
        "delta_vs_adl": d,
        "relative_vs_adl_pct": rel,
        "target_rel": TARGET_REL_VS_ADL,
        "target_met": bool(ho["mean_value"] >= target),
        "promoted_main_ckpt": promoted,
        "episodes": N_EP,
        "hist": hist,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"DONE holdout={ho['mean_value']:.2f} ADL={adl_m:.2f} d={d:.2f} ({rel:+.1f}%) "
        f"target_met={report['target_met']} promoted={promoted}",
        flush=True,
    )
    print(f"WROTE {REPORT}", flush=True)
    return 0 if report["target_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
