#!/usr/bin/env python3
"""Train causal-observation Dueling DDQN for prioritized workflow utility.

PDP (Priority-Based Demonstration Policy = Always-Drop-Low) is the BC teacher only.
It is NOT an optimization oracle and NOT a principal literature baseline.

Pipeline:
  PDP teacher --> Behavioral cloning (BC-only) --> Dueling DDQN RL fine-tune (Full)

Principal comparator: ValueGreedy.
Learning-contribution analysis: PDP vs BC-only vs Full Causal Dueling DDQN.

No privileged contact leakage. Does not overwrite ddqn_best.pt.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import torch
import torch.nn.functional as F

from ddqn_agent import DDQNAgent
from run_drl_experiment import make_agent_policy
from workflow_mdp import (
    A_DROP,
    A_LOCAL,
    N_ACTIONS,
    WorkflowMDP,
    policy_b3,
    policy_b4,
    policy_value_greedy,
)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
CKPT = OUT / "ddqn_causal_dueling.pt"
HOLDOUT = list(range(300, 340))

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


def make_env(seed=0):
    return WorkflowMDP(seed=seed, cfg=dict(CFG))


def policy_pdp(obs, mask, env: WorkflowMDP) -> int:
    """Priority-Based Demonstration Policy (Always-Drop-Low). BC teacher only."""
    if env.pending_decision is None:
        return A_LOCAL
    wf_id, sid = env.pending_decision
    wf = env.wfs[wf_id]
    if wf.value < env.cfg["value_high"] - 1e-9 and mask[A_DROP]:
        return A_DROP
    if sid in ("A", "D"):
        return A_LOCAL
    return policy_b3(obs, mask, env)


# Back-compat alias (do not use in new code)
policy_oracle_drop_low = policy_pdp


def eval_pol(name, fn, seeds):
    rows = []
    for s in seeds:
        info = make_env(s).run_policy(fn, seed=s)
        rows.append(info)
    v = np.array([r["total_value"] for r in rows], float)
    n = np.array([r["n_success"] for r in rows], float)
    return {
        "policy": name,
        "mean_value": float(v.mean()),
        "std_value": float(v.std()),
        "mean_success": float(n.mean()),
        "std_success": float(n.std()),
        "per_seed_value": v.tolist(),
        "per_seed_success": n.tolist(),
    }


def collect_pdp_batch(n_episodes: int = 80):
    """(obs, action, mask) demonstrations from the PDP teacher."""
    obs_l, act_l, mask_l = [], [], []
    for ep in range(n_episodes):
        env = make_env(5000 + ep)
        obs = env.reset(5000 + ep)
        done = False
        while not done:
            mask = env.action_mask()
            a = policy_pdp(obs, mask, env)
            obs_l.append(obs.copy())
            act_l.append(a)
            mask_l.append(mask.astype(np.float32))
            obs, _, done, _ = env.step(a)
    return (
        np.stack(obs_l).astype(np.float32),
        np.array(act_l, dtype=np.int64),
        np.stack(mask_l).astype(np.float32),
    )


def behavioral_clone(agent: DDQNAgent, n_episodes: int = 80, epochs: int = 12, bs: int = 128):
    """Supervise Q-net toward PDP actions (cross-entropy on masked logits) → BC-only."""
    print(f"BC warm-start from PDP teacher ({n_episodes} eps, {epochs} epochs)", flush=True)
    obs, acts, masks = collect_pdp_batch(n_episodes)
    ov = eval_pol("PDP", policy_pdp, list(range(1, 21)))
    print(f"  PDP teacher probe V={ov['mean_value']:.2f} N={ov['mean_success']:.2f}", flush=True)
    opt = torch.optim.Adam(agent.q.parameters(), lr=1e-3)
    n = len(acts)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        total = 0.0
        steps = 0
        for i in range(0, n, bs):
            j = idx[i : i + bs]
            o = torch.as_tensor(obs[j], device=agent.device)
            a = torch.as_tensor(acts[j], device=agent.device)
            m = torch.as_tensor(masks[j], device=agent.device)
            logits = agent.q(o)
            logits = logits.masked_fill(m < 0.5, -1e9)
            loss = F.cross_entropy(logits, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        print(f"  BC epoch {ep+1}/{epochs} loss={total/max(steps,1):.4f}", flush=True)
    agent.tgt.load_state_dict(agent.q.state_dict())
    agent.eps = 0.35  # start RL with moderate explore, not blank slate


def act_with_drop_bias(agent: DDQNAgent, obs, mask, env) -> int:
    """During explore: if low-value and DROP legal, take DROP with high probability."""
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return 0
    if np.random.rand() < agent.eps:
        if env.pending_decision is not None:
            wf = env.wfs[env.pending_decision[0]]
            if (
                wf.value < env.cfg["value_high"] - 1e-9
                and mask[A_DROP]
                and np.random.rand() < 0.70
            ):
                return A_DROP
        return int(np.random.choice(legal))
    return agent.act(obs, mask, explore=False)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    vg = eval_pol("ValueGreedy", policy_value_greedy, HOLDOUT)
    b4 = eval_pol("B4_joint", policy_b4, HOLDOUT)
    pdp = eval_pol("PDP", policy_pdp, HOLDOUT)
    print(
        f"BASE VG V={vg['mean_value']:.2f}  PDP teacher V={pdp['mean_value']:.2f} "
        f"(teacher−VG)={pdp['mean_value']-vg['mean_value']:.2f}",
        flush=True,
    )

    env = make_env(0)
    agent = DDQNAgent(
        env.obs_dim, N_ACTIONS, lr=4e-4, gamma=0.99, device=device, dueling=True
    )
    agent.eps_decay = 0.996
    agent.eps_min = 0.03

    behavioral_clone(agent, n_episodes=100, epochs=15)
    bc_eval = eval_pol("BC_only", make_agent_policy(agent), list(range(1, 21)))
    print(
        f"BC-only: V={bc_eval['mean_value']:.2f} N={bc_eval['mean_success']:.2f}",
        flush=True,
    )
    agent.save(str(CKPT))

    best = bc_eval["mean_value"]
    t0 = time.time()
    n_episodes = 1000
    for ep in range(1, n_episodes + 1):
        obs = env.reset(3000 + ep)
        done = False
        info = {"total_value": 0.0}
        while not done:
            mask = env.action_mask()
            a = act_with_drop_bias(agent, obs, mask, env)
            nobs, r, done, info = env.step(a)
            nmask = env.action_mask() if not done else np.ones(N_ACTIONS, dtype=bool)
            agent.push(obs, a, r, nobs, float(done), mask, nmask)
            agent.train_step(128)
            obs = nobs
        agent.decay_eps()
        if ep % 5 == 0:
            agent.sync_target()
        if ep % 50 == 0 or ep == n_episodes:
            ev = eval_pol("ddqn", make_agent_policy(agent), list(range(1, 26)))
            print(
                f"ep={ep:4d} eps={agent.eps:.3f} "
                f"evalV={ev['mean_value']:.2f} evalN={ev['mean_success']:.2f} "
                f"[{time.time()-t0:.0f}s]",
                flush=True,
            )
            if ev["mean_value"] > best:
                best = ev["mean_value"]
                agent.save(str(CKPT))
                print(f"  saved best V={best:.2f}", flush=True)

    agent.load(str(CKPT))
    agent.eps = 0.0
    dd = eval_pol("Full_Causal_Dueling_DDQN", make_agent_policy(agent), HOLDOUT)

    dV = dd["mean_value"] - vg["mean_value"]
    rel = 100.0 * dV / max(vg["mean_value"], 1e-9)
    dN = dd["mean_success"] - vg["mean_success"]
    dV_pdp = dd["mean_value"] - pdp["mean_value"]
    v_dd = np.array(dd["per_seed_value"])
    v_vg = np.array(vg["per_seed_value"])
    wtl = [
        int((v_dd > v_vg).sum()),
        int((v_dd == v_vg).sum()),
        int((v_dd < v_vg).sum()),
    ]
    ambitious_target_met = bool(rel >= 15.0 and dV >= 3.0)
    utility_superior = bool(dV > 0.0)

    cfg_out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in CFG.items()}
    out = {
        "cfg": cfg_out,
        "checkpoint": str(CKPT.resolve()),
        "holdout": HOLDOUT,
        "method": [
            "PDP (Priority-Based Demonstration Policy) BC warm-start",
            "early-phase and high-pending DROP reward bonuses",
            "exploration bias toward DROP on low-value stages",
            "Dueling DDQN RL fine-tune",
        ],
        "pdp_role": "BC_teacher_only_not_oracle_not_literature_baseline",
        "valuegreedy": vg,
        "b4": b4,
        "pdp_teacher": pdp,
        "bc_only_probe": bc_eval,
        "ddqn": dd,
        "delta_value_vs_vg": dV,
        "relative_value_pct_vs_vg": rel,
        "delta_value_vs_pdp": dV_pdp,
        "relative_value_pct_vs_pdp": 100.0 * dV_pdp / max(pdp["mean_value"], 1e-9),
        "delta_success_vs_vg": dN,
        "paired_value_wtl_vs_vg": wtl,
        "ambitious_target_pct": 15.0,
        "ambitious_target_met": ambitious_target_met,
        "utility_superior_vs_ValueGreedy": utility_superior,
        "no_privileged_obs": True,
        # back-compat keys for older readers
        "oracle_drop_low": pdp,
        "delta_value_vs_oracle": dV_pdp,
    }
    path = OUT / "causal_dueling_holdout.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"HOLDOUT FullDDQN V={dd['mean_value']:.2f} VG V={vg['mean_value']:.2f} "
        f"PDP V={pdp['mean_value']:.2f} dV_vg={dV:.2f} ({rel:.1f}%) "
        f"dV_pdp={dV_pdp:.2f} superior_vs_VG={utility_superior} "
        f"ambitious_15pct={ambitious_target_met} wtl={wtl}",
        flush=True,
    )
    print(f"WROTE {path}", flush=True)
    return 0 if utility_superior else 2


if __name__ == "__main__":
    raise SystemExit(main())
