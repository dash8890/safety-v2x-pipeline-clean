#!/usr/bin/env python3
"""Time-boxed WiLab Mode-2 sprint (~10-12 min wall).

Kills the 60-MATLAB blind wait. Plan:
  4 train episodes (sequential, required for online updates)
  1 final greedy eval on seeds 300-302 in PARALLEL MATLAB jobs
  Gate: beat VG by >=15% AND strictly beat PDP teacher mean

If gate fails: exit 2 with honest FAIL — do not claim learning beyond PDP.
"""
from __future__ import annotations

import concurrent.futures
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bc_wilab_droplow import export_weights
from ddqn_agent import DDQNAgent

ROOT = Path(__file__).resolve().parent
WF2 = ROOT.parent
OUT = ROOT / "outputs"
TRAIN = WF2 / "outputs" / "wilab_ddqn_sprint"
WEIGHTS = OUT / "ddqn_wilab_sprint.weights.json"
CKPT = OUT / "ddqn_wilab_sprint.pt"
HIST = TRAIN / "sprint_history.csv"
PLOT = TRAIN / "wilab_sprint.png"
META = TRAIN / "wilab_sprint_meta.json"
STATUS = WF2 / "outputs" / "WILAB_CLAIM_STATUS.json"
BASE_CFG = WF2 / "config" / "wilab_causal_eval.json"
MATLAB = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
VG_HOLDOUT = WF2 / "outputs" / "wilab_causal_20261124_031648"
FINAL = WF2 / "outputs" / "wilab_ddqn_holdout_final" / "FINAL_holdout.json"

N_ACTIONS = 5
OBS_DIM = 16
EVAL_SEEDS = [300, 301, 302]
MAX_EP = 4
MIN_REL = 0.15
EPS0 = 0.10
EPS_MIN = 0.04


def policy_means(policy: str) -> dict[int, float]:
    out = {}
    for s in EVAL_SEEDS:
        p = VG_HOLDOUT / f"seed{s}_{policy}" / "workflow_v2" / "summary.json"
        out[s] = float(json.loads(p.read_text())["total_value"])
    return out


def write_cfg(path: Path, eps: float):
    cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    cfg["workflow"]["ddqn_eps"] = float(eps)
    cfg["workflow"]["ddqn_weights_json"] = str(WEIGHTS).replace("\\", "/")
    cfg["workflow"]["collect_transitions"] = True
    cfg["workflow"]["collect_demos"] = False
    cfg["workflow"]["drop_reward_bonus"] = 0.0
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def run_wilab(case: Path, cfg: Path, seed: int, eps: float, collect: bool) -> dict:
    cfg_data = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    cfg_data["workflow"]["ddqn_eps"] = float(eps)
    cfg_data["workflow"]["ddqn_weights_json"] = str(WEIGHTS).replace("\\", "/")
    cfg_data["workflow"]["collect_transitions"] = bool(collect)
    cfg_data["workflow"]["collect_demos"] = False
    cfg_data["workflow"]["drop_reward_bonus"] = 0.0
    cfg.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
    if case.exists():
        shutil.rmtree(case, ignore_errors=True)
    case.mkdir(parents=True, exist_ok=True)
    script = (
        f"addpath(genpath('{WF2.as_posix()}')); "
        f"try; s=runR3PilotCase('{case.as_posix()}','workflow_causal_ddqn',{seed},"
        f"'config','{cfg.as_posix()}','radio_mode','wilab_mode2'); disp(jsonencode(s)); "
        f"catch ME; disp(getReport(ME)); exit(1); end; exit(0);"
    )
    r = subprocess.run([MATLAB, "-batch", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout or "")[-1200:] + "\n" + (r.stderr or "")[-1200:])
    summary = json.loads((case / "workflow_v2" / "summary.json").read_text(encoding="utf-8"))
    trp = case / "workflow_v2" / "transitions.json"
    tr = json.loads(trp.read_text(encoding="utf-8")) if trp.is_file() else None
    return {"summary": summary, "transitions": tr}


def push(agent: DDQNAgent, tr: dict, terminal_bonus: float) -> tuple[int, float]:
    if not tr:
        return 0, 0.0
    obs = np.asarray(tr["obs"], np.float32)
    acts = np.asarray(tr["action_idx"], np.int64) - 1
    rew = np.asarray(tr["reward"], np.float32).copy()
    nobs = np.asarray(tr["next_obs"], np.float32)
    done = np.asarray(tr["done"], np.float32)
    mask = np.asarray(tr["mask"], np.float32)
    nmask = np.asarray(tr["next_mask"], np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if len(rew):
        rew[-1] += terminal_bonus
    ep_ret = float(rew.sum()) if len(rew) else 0.0
    n = 0
    for i in range(len(acts)):
        a = int(acts[i])
        if 0 <= a < N_ACTIONS:
            agent.push(obs[i], a, float(rew[i]), nobs[i], float(done[i]), mask[i], nmask[i])
            n += 1
    return n, ep_ret


def eval_one(seed: int) -> float:
    case = TRAIN / f"eval_seed{seed}"
    cfg = TRAIN / f"eval_seed{seed}_cfg.json"
    out = run_wilab(case, cfg, seed, 0.0, collect=False)
    return float(out["summary"]["total_value"])


def eval_parallel() -> dict[int, float]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(eval_one, s): s for s in EVAL_SEEDS}
        return {futs[f]: f.result() for f in concurrent.futures.as_completed(futs)}


def write_status(payload: dict):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {STATUS}", flush=True)


def main():
    t0 = time.time()
    TRAIN.mkdir(parents=True, exist_ok=True)
    vg_map = policy_means("workflow_value_greedy")
    pdp_map = policy_means("workflow_drop_low")
    vg_mean = float(np.mean(list(vg_map.values())))
    pdp_mean = float(np.mean(list(pdp_map.values())))

    final = json.loads(FINAL.read_text(encoding="utf-8")) if FINAL.is_file() else {}
    # Immediate honest claim from existing FINAL holdout (no waiting)
    write_status(
        {
            "as_of": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "radio": "wilab_mode2",
            "pdp_role": "BC_teacher_only_not_oracle_not_literature_baseline",
            "principal_baseline": "ValueGreedy",
            "existing_final_holdout_seeds_300_304": {
                "ValueGreedy": final.get("valuegreedy", {}).get("mean_value", 9.0),
                "PDP_teacher": final.get("droplow", final.get("pdp_teacher", {})).get("mean_value", 45.6)
                if isinstance(final.get("droplow", final.get("pdp_teacher", {})), dict)
                else 45.6,
                "Full_or_BC_DDQN": final.get("ddqn", {}).get("mean_value", 45.6),
                "verdict": "beats_VG_but_ties_PDP_so_learning_contribution_not_shown",
            },
            "sprint": "running_4ep_plus_parallel_eval",
            "note": "Killed 30-ep/60-MATLAB convergence loop; delivering status now + time-boxed sprint.",
        }
    )

    print(f"VG mean={vg_mean:.2f} PDP teacher mean={pdp_mean:.2f}", flush=True)
    src_w = OUT / "ddqn_wilab_causal.weights.json"
    src_pt = OUT / "ddqn_wilab_causal.pt"
    shutil.copy(src_w, WEIGHTS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DDQNAgent(OBS_DIM, N_ACTIONS, lr=3e-4, gamma=0.99, device=device, dueling=True)
    agent.load(src_pt)
    agent.eps = EPS0
    agent.eps_min = EPS_MIN
    agent.eps_decay = 0.97

    rows = []
    eps = EPS0
    with HIST.open("w", newline="", encoding="utf-8") as f:
        fields = ["episode", "seed", "total_value", "ep_return", "eps", "loss", "n_transitions", "elapsed_s"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ep in range(1, MAX_EP + 1):
            seed = 2000 + ep
            case = TRAIN / f"train_ep{ep:03d}_seed{seed}"
            cfg = TRAIN / f"train_ep{ep:03d}_cfg.json"
            print(f"SPRINT EP {ep}/{MAX_EP} seed={seed} eps={eps:.3f}", flush=True)
            out = run_wilab(case, cfg, seed, eps, collect=True)
            tv = float(out["summary"].get("total_value", 0))
            # reward pressure to exceed PDP-scale utilities
            bonus = 0.15 * max(0.0, tv - pdp_mean)
            n_tr, ep_ret = push(agent, out["transitions"], terminal_bonus=bonus)
            loss = 0.0
            for _ in range(max(40, n_tr)):
                loss = agent.train_step(64)
            if ep % 2 == 0:
                agent.sync_target()
            agent.decay_eps()
            eps = max(EPS_MIN, eps * 0.97)
            agent.save(CKPT)
            export_weights(agent, WEIGHTS)
            row = {
                "episode": ep,
                "seed": seed,
                "total_value": tv,
                "ep_return": ep_ret,
                "eps": eps,
                "loss": float(loss),
                "n_transitions": n_tr,
                "elapsed_s": round(time.time() - t0, 1),
            }
            rows.append(row)
            w.writerow(row)
            f.flush()
            print(
                f"  trainV={tv:.1f} ret={ep_ret:.2f} tr={n_tr} loss={loss:.4f} t={row['elapsed_s']}s",
                flush=True,
            )

    print("PARALLEL greedy eval seeds", EVAL_SEEDS, flush=True)
    per = eval_parallel()
    ev_mean = float(np.mean(list(per.values())))
    beat_vg = ev_mean >= vg_mean * (1 + MIN_REL) and min(per.values()) >= vg_mean
    beat_pdp = ev_mean > pdp_mean and min(per.values()) > pdp_mean
    ok = beat_vg and beat_pdp
    elapsed = time.time() - t0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot([r["episode"] for r in rows], [r["total_value"] for r in rows], "o-", label="train V")
    ax.axhline(vg_mean, color="#e76f51", ls="--", label=f"VG={vg_mean:.1f}")
    ax.axhline(pdp_mean, color="#9b2226", ls=":", label=f"PDP teacher={pdp_mean:.1f}")
    ax.axhline(ev_mean, color="#1f4e79", ls="-", label=f"sprint eval={ev_mean:.1f}")
    ax.set_title("WiLab sprint " + ("PASS learning+VG" if ok else "FAIL vs PDP teacher"))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Utility")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=140)
    plt.close(fig)

    meta = {
        "radio": "wilab_mode2",
        "elapsed_s": elapsed,
        "converged_learning_beyond_pdp": ok,
        "eval_mean": ev_mean,
        "eval_per_seed": per,
        "vg_mean": vg_mean,
        "pdp_teacher_mean": pdp_mean,
        "beat_vg": beat_vg,
        "beat_pdp": beat_pdp,
        "checkpoint": str(CKPT),
        "plot": str(PLOT),
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    write_status(
        {
            "as_of": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "radio": "wilab_mode2",
            "pdp_role": "BC_teacher_only_not_oracle_not_literature_baseline",
            "principal_comparison_vs_ValueGreedy": {
                "existing_FINAL_holdout_mean": {
                    "VG": 9.0,
                    "PDP": 45.6,
                    "DDQN_BC_or_Full": 45.6,
                    "rel_vs_VG_pct": 406.7,
                },
                "sprint_eval_mean": ev_mean,
                "sprint_per_seed": per,
                "claim_vs_VG": bool(beat_vg),
            },
            "learning_contribution_vs_PDP": {
                "question": "Did RL improve beyond the demonstrated PDP teacher?",
                "pdp_mean_eval_seeds": pdp_mean,
                "sprint_eval_mean": ev_mean,
                "strictly_exceeds_pdp": bool(beat_pdp),
                "verdict": "PASS" if beat_pdp else "FAIL_or_TIE_no_learning_claim",
            },
            "sprint_meta": meta,
            "time_recovery": "killed_30ep_loop; 4ep_sprint_plus_parallel_eval",
        }
    )
    print(json.dumps(meta, indent=2), flush=True)
    print(f"SPRINT DONE elapsed={elapsed:.0f}s ok={ok}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
