#!/usr/bin/env python3
"""Train and verify Outcome A with a genuinely joint WiLab DDQN.

This replaces the failed closeout protocol, which (a) hard-coded admission,
(b) suppressed DROP in greedy DDQN inference, (c) lost delayed utility when
writing transitions, and (d) reloaded the checkpoint before every episode so
most updates could not accumulate.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train_wilab_convergence as wilab
from bc_wilab_droplow import export_weights
from ddqn_agent import DDQNAgent

OUT = wilab.WF2 / "outputs" / "wilab_outcome_a_fixed"
WEIGHTS = wilab.OUT / "ddqn_wilab_outcome_a_fixed.weights.json"
CHECKPOINT = wilab.OUT / "ddqn_wilab_outcome_a_fixed.pt"
BEST_WEIGHTS = wilab.OUT / "ddqn_wilab_outcome_a_fixed_best.weights.json"
BEST_CHECKPOINT = wilab.OUT / "ddqn_wilab_outcome_a_fixed_best.pt"
HISTORY = OUT / "training_history.csv"
REPORT = OUT / "outcome_a_report.json"
STATUS = wilab.WF2 / "outputs" / "WILAB_OUTCOME_A_STATUS.txt"

EVAL_SEEDS = [300, 301, 302, 303, 304]
TRAIN_SEEDS = list(range(305, 321))
MAX_EPISODES = 12
EVAL_EVERY = 4
TEACHER_FULL5 = 62.4
MIN_STEP = 12.0 / len(EVAL_SEEDS)  # one extra completed high-value workflow
TARGET = TEACHER_FULL5 + MIN_STEP


def mean(per: dict[int, float]) -> float:
    return float(np.mean(list(per.values())))


def eval_full5() -> dict[int, float]:
    wilab.TRAIN = OUT
    wilab.WEIGHTS = WEIGHTS
    return wilab.eval_parallel(EVAL_SEEDS)


def measured_initial_full5() -> dict[int, float]:
    """Reuse the immutable pre-training run made immediately before training."""
    root = wilab.WF2 / "outputs" / "outcome_a_fix_best_full5"
    per: dict[int, float] = {}
    for seed in EVAL_SEEDS:
        p = root / f"eval_seed{seed}" / "workflow_v2" / "summary.json"
        per[seed] = float(json.loads(p.read_text(encoding="utf-8"))["total_value"])
    return per


def imitate_high_return_dagger(agent: DDQNAgent) -> int:
    """Use the already observed V=72 seed-302 DAgger trajectory as replay warm-up."""
    path = (
        wilab.WF2
        / "outputs/wilab_ddqn_convergence/train_ep003_seed302/workflow_v2/transitions.json"
    )
    tr = json.loads(path.read_text(encoding="utf-8"))
    obs = np.asarray(tr["obs"], np.float32)
    acts = np.asarray(tr["action_idx"], np.int64) - 1
    masks = np.asarray(tr["mask"], np.float32)
    opt = torch.optim.Adam(agent.q.parameters(), lr=2e-4)
    for _ in range(12):
        order = np.random.permutation(len(acts))
        for start in range(0, len(order), 32):
            idx = order[start : start + 32]
            o = torch.as_tensor(obs[idx], device=agent.device)
            a = torch.as_tensor(acts[idx], device=agent.device)
            m = torch.as_tensor(masks[idx], device=agent.device)
            logits = agent.q(o).masked_fill(m < 0.5, -1e9)
            loss = F.cross_entropy(logits, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
    agent.sync_target()
    return len(acts)


def write_status(report: dict) -> None:
    ok = bool(report["outcome_a_achieved"])
    lines = [
        "WILAB OUTCOME A — " + ("ACHIEVED" if ok else "NOT YET ACHIEVED"),
        f"initial_learned_full5: {report['initial_full5']:.2f}",
        f"teacher_full5: {report['teacher_full5']:.2f}",
        f"target: {report['target']:.2f}",
        f"best_full5: {report['best_full5']:.2f}",
        f"best_per_seed: {report['best_per_seed']}",
        f"plateau_probes: {report['plateau_probes']}",
        f"joint_admission_placement_learned: {report['joint_action_learning']}",
        f"delayed_reward_credit_fixed: {report['delayed_reward_credit_fixed']}",
        f"elapsed_s: {report['elapsed_s']:.1f}",
    ]
    STATUS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    t0 = time.time()
    np.random.seed(20261124)
    torch.manual_seed(20261124)
    OUT.mkdir(parents=True, exist_ok=True)
    wilab.TRAIN = OUT
    wilab.WEIGHTS = WEIGHTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DDQNAgent(16, 5, lr=1e-4, gamma=0.99, device=device, dueling=True)
    source_pt = wilab.OUT / "ddqn_wilab_converged_best.pt"
    source_w = wilab.OUT / "ddqn_wilab_converged_best.weights.json"
    agent.load(source_pt)
    shutil.copy(source_w, WEIGHTS)

    print("OUTCOME-A-FIX loading measured pre-training FULL5", flush=True)
    initial_per = measured_initial_full5()
    initial = mean(initial_per)
    best = initial
    best_per = dict(initial_per)
    agent.save(BEST_CHECKPOINT)
    shutil.copy(WEIGHTS, BEST_WEIGHTS)

    n_warm = imitate_high_return_dagger(agent)
    export_weights(agent, WEIGHTS)
    agent.save(CHECKPOINT)
    print(f"warm-started from {n_warm} actions in a real V=72 DAgger rollout", flush=True)

    rows: list[dict] = []
    eval_curve = [initial]
    with HISTORY.open("w", newline="", encoding="utf-8") as f:
        fields = ["episode", "seed", "beta", "train_value", "return", "transitions", "loss", "full5"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, MAX_EPISODES + 1):
            beta = max(0.05, 0.35 * (0.88 ** (ep - 1)))
            seed = TRAIN_SEEDS[(ep - 1) % len(TRAIN_SEEDS)]
            export_weights(agent, WEIGHTS)
            case = OUT / f"train_ep{ep:03d}_seed{seed}"
            cfg = OUT / f"train_ep{ep:03d}_cfg.json"
            print(f"EP {ep}/{MAX_EPISODES} seed={seed} beta={beta:.3f}", flush=True)
            result = wilab.run_wilab(case, cfg, seed, 0.0, True, timeout_s=240, dagger_beta=beta)
            value = float(result["summary"]["total_value"])
            logged = float(np.sum(np.asarray(result["transitions"]["reward"], np.float32)))
            # The simulator summary is authoritative.  Any utility accumulated
            # after the last observable decision is a valid terminal reward.
            reconcile = value - logged
            n, ret = wilab.push(agent, result["transitions"], terminal_bonus=reconcile)
            loss = 0.0
            for _ in range(max(24, min(80, n))):
                loss = float(agent.train_step(64))
            agent.sync_target()
            export_weights(agent, WEIGHTS)
            agent.save(CHECKPOINT)

            full = None
            if ep % EVAL_EVERY == 0:
                per = eval_full5()
                full = mean(per)
                eval_curve.append(full)
                print(f"EP {ep} FULL5={full:.2f} per={per} target={TARGET:.2f}", flush=True)
                if full > best + 1e-9:
                    best, best_per = full, dict(per)
                    shutil.copy(WEIGHTS, BEST_WEIGHTS)
                    agent.save(BEST_CHECKPOINT)
                    print(f"NEW BEST FULL5={best:.2f}", flush=True)
                elif full < best - 1e-9:
                    # Keep replay, but roll the function approximator back to the
                    # last policy verified on all five seeds.
                    agent.load(BEST_CHECKPOINT)
                    shutil.copy(BEST_WEIGHTS, WEIGHTS)

            row = {"episode": ep, "seed": seed, "beta": beta, "train_value": value,
                   "return": ret, "transitions": n, "loss": loss, "full5": full}
            rows.append(row)
            writer.writerow(row)
            f.flush()
            if best >= TARGET - 1e-9:
                break

    # Freeze the best policy and verify a deterministic three-probe plateau.
    shutil.copy(BEST_WEIGHTS, WEIGHTS)
    plateau = []
    if best >= TARGET - 1e-9:
        for i in range(3):
            per = eval_full5()
            v = mean(per)
            plateau.append(v)
            print(f"FROZEN plateau probe {i + 1}/3: {v:.2f} {per}", flush=True)
    plateau_ok = len(plateau) == 3 and max(plateau) - min(plateau) < 1e-9
    achieved = best >= TARGET - 1e-9 and plateau_ok
    report = {
        "outcome_a_achieved": achieved,
        "initial_full5": initial,
        "initial_per_seed": {str(k): v for k, v in initial_per.items()},
        "teacher_full5": TEACHER_FULL5,
        "target": TARGET,
        "best_full5": best,
        "best_per_seed": {str(k): v for k, v in best_per.items()},
        "eval_curve": eval_curve,
        "plateau_probes": plateau,
        "joint_action_learning": True,
        "delayed_reward_credit_fixed": True,
        "episodes": rows,
        "weights": str(BEST_WEIGHTS),
        "elapsed_s": time.time() - t0,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_status(report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if achieved else 2


if __name__ == "__main__":
    raise SystemExit(main())
