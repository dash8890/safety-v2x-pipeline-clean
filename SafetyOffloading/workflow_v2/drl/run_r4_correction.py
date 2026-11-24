#!/usr/bin/env python3
"""R4 corrections - preserve ddqn_best.pt; train/eval causal obs (+ optional mode2_event).

Eval matrix (partial until Mode-2 train completes):
  R4 reference:     obs=privileged, radio=abstract, checkpoint=ddqn_best.pt
  Corrected obs:    obs=causal,     radio=abstract, checkpoint=ddqn_causal_obs.pt
  Corrected radio:  obs=causal,     radio=mode2_event (train ddqn_mode2.pt)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ddqn_agent import DDQNAgent
from run_drl_experiment import eval_policy, make_agent_policy
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
HOLDOUT = list(range(300, 340))
REF_CKPT = OUT / "ddqn_best.pt"  # never overwrite
CAUSAL_CKPT = OUT / "ddqn_causal_obs.pt"
MODE2_CKPT = OUT / "ddqn_mode2.pt"


def make_env(obs_mode: str, radio_mode: str, seed: int = 0) -> WorkflowMDP:
    return WorkflowMDP(
        seed=seed,
        cfg={"obs_mode": obs_mode, "radio_mode": radio_mode},
    )


def eval_cfg(name, policy_fn, seeds, obs_mode, radio_mode):
    rows = []
    for s in seeds:
        env = make_env(obs_mode, radio_mode, seed=s)
        info = env.run_policy(policy_fn, seed=s)
        rows.append(info)
    succ = np.array([r["n_success"] for r in rows], dtype=float)
    val = np.array([r["total_value"] for r in rows], dtype=float)
    return {
        "policy": name,
        "obs_mode": obs_mode,
        "radio_mode": radio_mode,
        "mean_success": float(succ.mean()),
        "std_success": float(succ.std()),
        "mean_value": float(val.mean()),
        "std_value": float(val.std()),
        "per_seed_success": succ.tolist(),
        "per_seed_value": val.tolist(),
    }


def train_checkpoint(
    out_path: Path,
    obs_mode: str,
    radio_mode: str,
    n_episodes: int = 600,
    eval_every: int = 40,
    device: str = "cpu",
):
    env = make_env(obs_mode, radio_mode, seed=0)
    agent = DDQNAgent(env.obs_dim, N_ACTIONS, lr=8e-4, gamma=0.99, device=device)
    best = -1.0
    t0 = time.time()
    for ep in range(1, n_episodes + 1):
        obs = env.reset(1000 + ep)
        done = False
        info = {"total_value": 0.0, "n_success": 0}
        while not done:
            mask = env.action_mask()
            a = agent.act(obs, mask, explore=True)
            nobs, r, done, info = env.step(a)
            nmask = env.action_mask() if not done else np.ones(N_ACTIONS, dtype=bool)
            agent.push(obs, a, r, nobs, float(done), mask, nmask)
            agent.train_step(96)
            obs = nobs
        agent.decay_eps()
        if ep % 8 == 0:
            agent.sync_target()
        if ep % eval_every == 0 or ep == n_episodes:
            ev = eval_cfg(
                "ddqn",
                make_agent_policy(agent),
                list(range(1, 16)),
                obs_mode,
                radio_mode,
            )
            print(
                f"[{obs_mode}/{radio_mode}] ep={ep:4d} "
                f"V={ev['mean_value']:.2f} N={ev['mean_success']:.2f} "
                f"[{time.time()-t0:.0f}s]",
                flush=True,
            )
            if ev["mean_value"] > best:
                best = ev["mean_value"]
                agent.save(str(out_path))
    agent.load(str(out_path))
    return agent


def load_agent(path: Path, device: str) -> DDQNAgent:
    env = WorkflowMDP(seed=0)
    agent = DDQNAgent(env.obs_dim, N_ACTIONS, device=device)
    agent.load(str(path))
    agent.eps = 0.0
    return agent


def paired(dd, vg):
    v_dd = np.array(dd["per_seed_value"], dtype=float)
    v_vg = np.array(vg["per_seed_value"], dtype=float)
    n_dd = np.array(dd["per_seed_success"], dtype=float)
    n_vg = np.array(vg["per_seed_success"], dtype=float)
    return {
        "delta_mean_value": float(v_dd.mean() - v_vg.mean()),
        "relative_value_pct": float(100 * (v_dd.mean() - v_vg.mean()) / max(v_vg.mean(), 1e-9)),
        "delta_mean_success": float(n_dd.mean() - n_vg.mean()),
        "relative_success_pct": float(100 * (n_dd.mean() - n_vg.mean()) / max(n_vg.mean(), 1e-9)),
        "paired_value_wtl": [
            int((v_dd > v_vg).sum()),
            int((v_dd == v_vg).sum()),
            int((v_dd < v_vg).sum()),
        ],
        "paired_success_wtl": [
            int((n_dd > n_vg).sum()),
            int((n_dd == n_vg).sum()),
            int((n_dd < n_vg).sum()),
        ],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    assert REF_CKPT.is_file(), f"missing R4 reference {REF_CKPT}"

    # --- smoke baselines causal/abstract ---
    print("SMOKE causal/abstract baselines", flush=True)
    for name, fn in [("B0", policy_b0), ("B3", policy_b3), ("VG", policy_value_greedy)]:
        r = eval_cfg(name, fn, list(range(1, 6)), "causal", "abstract")
        print(f"  {name} V={r['mean_value']:.2f} N={r['mean_success']:.2f}", flush=True)

    # --- train causal/abstract (correction A) ---
    print("TRAIN causal/abstract ->", CAUSAL_CKPT, flush=True)
    causal_agent = train_checkpoint(
        CAUSAL_CKPT, "causal", "abstract", n_episodes=700, eval_every=35, device=device
    )

    # --- holdout matrix row: corrected observation ---
    print("EVAL holdout corrected observation", flush=True)
    vg_c = eval_cfg("ValueGreedy", policy_value_greedy, HOLDOUT, "causal", "abstract")
    b4_c = eval_cfg("B4_joint", policy_b4, HOLDOUT, "causal", "abstract")
    dd_c = eval_cfg("DDQN_causal", make_agent_policy(causal_agent), HOLDOUT, "causal", "abstract")

    # R4 reference transfer diagnostic (old ckpt on privileged still)
    ref_agent = load_agent(REF_CKPT, device)
    dd_ref = eval_cfg(
        "DDQN_R4_ref", make_agent_policy(ref_agent), HOLDOUT, "privileged", "abstract"
    )
    vg_ref = eval_cfg("ValueGreedy_R4obs", policy_value_greedy, HOLDOUT, "privileged", "abstract")

    # Transfer diagnostic: frozen R4 ckpt on causal obs (expected degradation)
    dd_xfer = eval_cfg(
        "DDQN_R4_on_causal", make_agent_policy(ref_agent), HOLDOUT, "causal", "abstract"
    )

    print("TRAIN causal/mode2_event ->", MODE2_CKPT, flush=True)
    mode2_agent = train_checkpoint(
        MODE2_CKPT, "causal", "mode2_event", n_episodes=500, eval_every=50, device=device
    )
    print("EVAL holdout corrected radio", flush=True)
    vg_m = eval_cfg("ValueGreedy", policy_value_greedy, HOLDOUT, "causal", "mode2_event")
    b4_m = eval_cfg("B4_joint", policy_b4, HOLDOUT, "causal", "mode2_event")
    dd_m = eval_cfg("DDQN_mode2", make_agent_policy(mode2_agent), HOLDOUT, "causal", "mode2_event")

    result = {
        "preserved_r4_checkpoint": str(REF_CKPT.resolve()),
        "causal_checkpoint": str(CAUSAL_CKPT.resolve()),
        "mode2_checkpoint": str(MODE2_CKPT.resolve()),
        "holdout_seeds": HOLDOUT,
        "R4_reference_privileged_abstract": {
            "ddqn": dd_ref,
            "valuegreedy": vg_ref,
            "vs_vg": paired(dd_ref, vg_ref),
        },
        "corrected_observation_causal_abstract": {
            "ddqn": dd_c,
            "valuegreedy": vg_c,
            "b4": b4_c,
            "vs_vg": paired(dd_c, vg_c),
            "r4_ckpt_transfer_on_causal": dd_xfer,
        },
        "corrected_radio_causal_mode2_event": {
            "ddqn": dd_m,
            "valuegreedy": vg_m,
            "b4": b4_m,
            "vs_vg": paired(dd_m, vg_m),
            "note": "mode2_event = COMMIT->ACK->frag input->CPU->frag output; not full WiLab PHY",
        },
    }
    out = OUT / "r4_correction_results.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"WROTE {out}", flush=True)
    print("CORRECTED OBS vs VG:", result["corrected_observation_causal_abstract"]["vs_vg"], flush=True)
    print("CORRECTED RADIO vs VG:", result["corrected_radio_causal_mode2_event"]["vs_vg"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
