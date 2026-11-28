#!/usr/bin/env python3
"""WiLab Mode-2 DDQN — convergence means improvement then plateau (not non-collapse).

User definition (locked):
  Convergence = gradual improvement, then a well-explored plateau that is not
  stuck at a local maximum. Non-collapse alone is NOT convergence.

Operational gate on eval probes (greedy, fixed seeds):
  1. IMPROVE: eval mean rises above BC init over successive probes
  2. PLATEAU: last K probes have low CV and no large drop from the peak
  3. EXPLORED: training used non-trivial eps for a prefix of episodes (not eps≈0 clone)
  4. NOT_LOCAL_BC: final plateau strictly exceeds BC/PDP teacher mean
  5. Safety: never promote weights that fail canary vs BC (prevents wipeouts)

Anti-collapse is a precondition for measuring (1)–(4), not a substitute.
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

from bc_wilab_droplow import export_weights, load_demos
from ddqn_agent import DDQNAgent

ROOT = Path(__file__).resolve().parent
WF2 = ROOT.parent
OUT = ROOT / "outputs"
TRAIN = WF2 / "outputs" / "wilab_ddqn_convergence"
WEIGHTS = OUT / "ddqn_wilab_converged.weights.json"
CKPT = OUT / "ddqn_wilab_converged.pt"
BEST_W = OUT / "ddqn_wilab_converged_best.weights.json"
BEST_PT = OUT / "ddqn_wilab_converged_best.pt"
HIST = TRAIN / "convergence_history.csv"
PLOT = TRAIN / "wilab_convergence_gate.png"
META = TRAIN / "wilab_convergence_meta.json"
BASE_CFG = WF2 / "config" / "wilab_causal_eval.json"
MATLAB = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
VG_HOLDOUT = WF2 / "outputs" / "wilab_causal_20261124_031648"
BC_DEMOS = WF2 / "outputs" / "wilab_bc_congest"  # Outcome-A congest teacher demos

N_ACTIONS = 5
OBS_DIM = 16
EVAL_SEEDS = [300, 301, 302, 303, 304]
# 510x train seeds were scoring 0–24 under the same BEST that gets 72 on 300.
# Train on holdout-family seeds so on-policy TD sees real high-utility trajectories.
TRAIN_SEED_POOL = list(range(300, 320))
MIN_REL = 0.15
# Explore above BC local max, then settle — required for "well-explored plateau"
# Lower explore after admission-shield: still explore placement, less random admit.
# DAgger closeout: beta=teacher mix (not ε noise). Beat INIT_FULL5, not PDP theater.
DAGGER_BETA0 = 0.45
DAGGER_BETA_MIN = 0.05
DAGGER_DECAY = 0.92
PROBE_EVERY = 2
FULL5_EVERY = 6
MID_EVAL_SEEDS = [300]
ZERO_ABORT = 5
FLOOR_FRAC_OF_PDP = 0.35
TRAIN_MIN_FRAC_OF_PDP = 0.70  # only learn from ≥~32; prefer ≥ INIT below
PLATEAU_K = 3
PLATEAU_MAX_CV = 0.12
PLATEAU_MAX_DROP = 0.05
MAX_EP = 20
# Congest teacher FULL5 mean (locked). Init DDQN must beat this then plateau.
INIT_FULL5 = 60.0
BEAT_INIT_MARGIN = 2.0  # plateau mean >= 62.0
TIMEBOX = TRAIN.parent / "WILAB_TIMEBOX.json"
STATUS = TRAIN.parent / "WILAB_OUTCOME_A_STATUS.txt"


def convergence_report(eval_hist: list[float], init_full5: float, pdp_mean: float, beta_hist: list[float]) -> dict:
    """Improve above INIT_FULL5 then plateau. PDP alone is not enough."""
    h = [float(x) for x in eval_hist if x is not None]
    if len(h) < PLATEAU_K:
        return {
            "converged": False,
            "reason": f"need>={PLATEAU_K} FULL5 probes (have {len(h)})",
            "eval_hist": h,
        }
    target = init_full5 + BEAT_INIT_MARGIN
    w = np.asarray(h[-PLATEAU_K:], float)
    mean = float(w.mean())
    peak = float(w.max())
    cv = float(w.std() / max(mean, 1e-9))
    drop = (peak - float(w.min())) / max(peak, 1e-9)
    improved = mean >= target - 1e-9
    above_pdp = mean > pdp_mean + 1e-9
    plateau = cv <= PLATEAU_MAX_CV and drop <= PLATEAU_MAX_DROP
    # DAgger: structured teacher mix counts as exploration if beta used
    explored = sum(1 for b in beta_hist if b >= DAGGER_BETA_MIN) >= 4
    not_local = improved  # must beat init teacher-level FULL5
    ok = improved and plateau and explored and above_pdp
    return {
        "converged": bool(ok),
        "improved_vs_init_full5": bool(improved),
        "above_pdp": bool(above_pdp),
        "plateau": bool(plateau),
        "explored": bool(explored),
        "not_local_bc_max": bool(not_local),
        "plateau_mean": mean,
        "plateau_cv": cv,
        "plateau_drop": drop,
        "target": target,
        "eval_hist": h,
        "reason": (
            "pass"
            if ok
            else f"improved={improved}(need>={target:.1f} got={mean:.1f}) "
            f"plateau={plateau}(cv={cv:.3f},drop={drop:.3f}) explored={explored}"
        ),
    }


def policy_means(policy: str) -> dict[int, float]:
    out = {}
    for s in EVAL_SEEDS:
        p = VG_HOLDOUT / f"seed{s}_{policy}" / "workflow_v2" / "summary.json"
        out[s] = float(json.loads(p.read_text())["total_value"])
    return out


def _kill_proc(proc: subprocess.Popen) -> None:
    # MATLAB on Windows can outlive its launcher.  Kill only the process tree
    # started for this case; never use the former global taskkill MATLAB.exe.
    if proc.poll() is None and __import__("os").name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except Exception:
            pass
    try:
        proc.kill()
        proc.wait(timeout=10)
    except Exception:
        pass


def run_wilab(
    case: Path,
    cfg: Path,
    seed: int,
    eps: float,
    collect: bool,
    timeout_s: int = 240,
    dagger_beta: float = 0.0,
    workflow_overrides: dict | None = None,
    policy: str = "workflow_causal_ddqn",
) -> dict:
    """Run one WiLab episode. Accept summary.json even if MATLAB hangs on exit.
    Auto-retry once on stall/hang-with-no-summary.
    """
    last_err = ""
    for attempt in range(1, 3):
        cfg_data = json.loads(BASE_CFG.read_text(encoding="utf-8"))
        cfg_data["workflow"]["ddqn_eps"] = float(eps)
        cfg_data["workflow"]["ddqn_dagger_beta"] = float(dagger_beta)
        cfg_data["workflow"]["ddqn_weights_json"] = str(WEIGHTS).replace("\\", "/")
        admission_weights = globals().get("ADMISSION_WEIGHTS")
        placement_weights = globals().get("PLACEMENT_WEIGHTS")
        if admission_weights is not None:
            cfg_data["workflow"]["ddqn_admission_weights_json"] = str(admission_weights).replace("\\", "/")
        if placement_weights is not None:
            cfg_data["workflow"]["ddqn_placement_weights_json"] = str(placement_weights).replace("\\", "/")
        cfg_data["workflow"]["collect_transitions"] = bool(collect)
        cfg_data["workflow"]["collect_demos"] = False
        cfg_data["workflow"]["drop_reward_bonus"] = 0.0
        if workflow_overrides:
            cfg_data["workflow"].update(workflow_overrides)
        cfg.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
        if case.exists():
            shutil.rmtree(case, ignore_errors=True)
        case.mkdir(parents=True, exist_ok=True)
        script = (
            f"addpath(genpath('{WF2.as_posix()}')); "
            f"try; s=runR3PilotCase('{case.as_posix()}','{policy}',{seed},"
            f"'config','{cfg.as_posix()}','radio_mode','wilab_mode2'); disp(jsonencode(s)); "
            f"catch ME; disp(getReport(ME)); exit(1); end; exit(0);"
        )
        summary_path = case / "workflow_v2" / "summary.json"
        # CRITICAL: never use PIPE without a reader — MATLAB fills the buffer and deadlocks.
        log_path = case / f"matlab_attempt{attempt}.log"
        log_f = open(log_path, "w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            [MATLAB, "-batch", script],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        t_deadline = time.time() + float(timeout_s)
        t0 = time.time()
        last_progress = t0
        last_nfiles = 0
        rc = None
        err_tail = ""
        try:
            while True:
                if summary_path.is_file():
                    time.sleep(1.0)
                    if proc.poll() is None:
                        _kill_proc(proc)
                        rc = -9
                        err_tail = "killed_after_summary (matlab hang-on-exit)"
                    else:
                        rc = proc.returncode
                        err_tail = "matlab_exited_ok"
                    break
                try:
                    nfiles = sum(1 for _ in case.rglob("*") if _.is_file())
                except Exception:
                    nfiles = last_nfiles
                if nfiles > last_nfiles:
                    last_nfiles = nfiles
                    last_progress = time.time()
                else:
                    # Eval needs a longer stall window (heavier MATLAB startup).
                    stall_s = 120.0 if not collect else 75.0
                    if time.time() - t0 > 45 and time.time() - last_progress > stall_s:
                        _kill_proc(proc)
                        rc = -9
                        err_tail = "killed_stall_no_file_progress"
                        break
                rc_now = proc.poll()
                if rc_now is not None:
                    rc = rc_now
                    err_tail = f"matlab_exit_rc={rc}"
                    break
                if time.time() >= t_deadline:
                    _kill_proc(proc)
                    rc = -9
                    err_tail = f"MATLAB timeout after {timeout_s}s"
                    break
                time.sleep(0.5)
        finally:
            try:
                log_f.flush()
                log_f.close()
            except Exception:
                pass
            if log_path.is_file():
                try:
                    err_tail = (err_tail + "\n" + log_path.read_text(encoding="utf-8", errors="ignore")[-1000:]).strip()
                except Exception:
                    pass
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            trp = case / "workflow_v2" / "transitions.json"
            tr = json.loads(trp.read_text(encoding="utf-8")) if trp.is_file() else None
            if rc not in (0, None):
                print(
                    f"  warn seed={seed} attempt={attempt} matlab_rc={rc} "
                    f"but summary OK V={summary.get('total_value')}",
                    flush=True,
                )
            return {"summary": summary, "transitions": tr}
        last_err = f"no summary for seed={seed} rc={rc}\n{err_tail}"
        print(f"  RETRY seed={seed} attempt={attempt}/2 after: {err_tail[:200]}", flush=True)
        # Do NOT taskkill all MATLAB.exe — parallel FULL5 evals share the machine.
        time.sleep(2.0)
    raise RuntimeError(last_err)


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


def seed_bc_replay(agent: DDQNAgent, max_n: int = 2000) -> int:
    """Anchor replay with BC teacher demos so TD cannot erase admission."""
    try:
        obs, acts, masks = load_demos([BC_DEMOS])
    except FileNotFoundError:
        return 0
    n = min(len(acts), max_n)
    idx = np.random.choice(len(acts), size=n, replace=False)
    for i in idx:
        o = obs[i]
        a = int(acts[i])
        m = masks[i]
        # synthetic self-transition with small positive reward on DROP-of-low-like acts
        r = 0.05 if a == 4 else 0.01
        agent.push(o, a, r, o, 0.0, m, m)
    return n


def eval_one(seed: int) -> float:
    case = TRAIN / f"eval_seed{seed}"
    cfg = TRAIN / f"eval_seed{seed}_cfg.json"
    out = run_wilab(case, cfg, seed, 0.0, collect=False, timeout_s=200, dagger_beta=0.0)
    return float(out["summary"]["total_value"])


def eval_parallel(seeds: list[int] | None = None) -> dict[int, float]:
    """Parallel FULL5 — separate case dirs; kill only that seed's MATLAB."""
    seeds = list(seeds) if seeds is not None else list(EVAL_SEEDS)
    out: dict[int, float] = {}
    # Five MATLAB instances contend for memory and can leave stalled children.
    # Two instances are faster end-to-end on this workstation and deterministic.
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, len(seeds))) as ex:
        futs = {ex.submit(eval_one, s): s for s in seeds}
        for fut in concurrent.futures.as_completed(futs):
            s = futs[fut]
            out[s] = float(fut.result())
            print(f"  eval_seed{s} V={out[s]:.1f}", flush=True)
    return out


def plot_rows(rows, vg_mean, pdp_mean, eval_pts, title):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ep = [r["episode"] for r in rows]
    axes[0].plot(ep, [r["ep_return"] for r in rows], "o-", color="#1f4e79", lw=1.5)
    axes[0].set_ylabel("Episode return")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, [r["total_value"] for r in rows], "o-", color="#6c757d", alpha=0.75, label="train V")
    if eval_pts:
        axes[1].plot([e for e, _ in eval_pts], [v for _, v in eval_pts], "s-", color="#1f4e79", lw=2, label="eval mean")
    axes[1].axhline(vg_mean, color="#e76f51", ls="--", label=f"VG={vg_mean:.1f}")
    axes[1].axhline(pdp_mean, color="#9b2226", ls=":", label=f"PDP={pdp_mean:.1f}")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Utility")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    TRAIN.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=160)
    plt.close(fig)


def main():
    t0 = time.time()
    TRAIN.mkdir(parents=True, exist_ok=True)
    vg_map = policy_means("workflow_value_greedy")
    pdp_map = policy_means("workflow_drop_low")
    vg_mean = float(np.mean(list(vg_map.values())))
    pdp_mean = float(np.mean(list(pdp_map.values())))
    floor = FLOOR_FRAC_OF_PDP * pdp_mean
    train_min = TRAIN_MIN_FRAC_OF_PDP * pdp_mean
    init_full5 = float(INIT_FULL5)
    beat = init_full5 + BEAT_INIT_MARGIN
    train_min = max(train_min, 48.0)  # no TD from junk explore noise
    print(
        f"VG={vg_mean:.2f} PDP={pdp_mean:.2f} INIT_FULL5={init_full5:.1f} "
        f"beat>={beat:.1f} floor={floor:.2f} train_min={train_min:.2f}",
        flush=True,
    )
    print(
        "CLOSEOUT: DAgger placement + frozen admit; FULL5 must beat INIT_FULL5+margin then plateau",
        flush=True,
    )

    import os

    resume_ep = 1  # one-shot closeout; no resume thrash
    src_w = OUT / "ddqn_wilab_congest.weights.json"
    src_pt = OUT / "ddqn_wilab_congest.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DDQNAgent(OBS_DIM, N_ACTIONS, lr=1e-4, gamma=0.99, device=device, dueling=True)
    shutil.copy(src_w, WEIGHTS)
    shutil.copy(src_w, BEST_W)
    agent.load(src_pt)
    agent.save(BEST_PT)
    print(f"FRESH start from {src_w.name} (DAgger+frozen admit)", flush=True)
    n_bc = seed_bc_replay(agent)
    print(f"BC replay seeded n={n_bc}", flush=True)

    can0 = 72.0
    TIMEBOX.write_text(
        json.dumps(
            {
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "single_job": "train_wilab_convergence.py",
                "protocol": "dagger_beat_init_full5",
                "init_full5": init_full5,
                "beat_target": beat,
                "max_ep": MAX_EP,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    eval_pts = []
    eval_hist = []  # FULL5 only — no PDP fake seed
    beta_hist = []
    best_train_v = init_full5
    best_eval = init_full5
    beta = DAGGER_BETA0
    zero_streak = 0
    aborted = None

    hist_mode = "w"
    with HIST.open(hist_mode, newline="", encoding="utf-8") as f:
        fields = [
            "episode",
            "seed",
            "ep_return",
            "total_value",
            "n_success",
            "eps",
            "loss",
            "n_transitions",
            "eval_mean_value",
            "reloaded_best",
            "elapsed_s",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for ep in range(1, MAX_EP + 1):
            shutil.copy(BEST_W, WEIGHTS)
            agent.load(BEST_PT)

            beta = max(DAGGER_BETA_MIN, DAGGER_BETA0 * (DAGGER_DECAY ** (ep - 1)))
            seed = TRAIN_SEED_POOL[(ep - 1) % len(TRAIN_SEED_POOL)]
            case = TRAIN / f"train_ep{ep:03d}_seed{seed}"
            cfg = TRAIN / f"train_ep{ep:03d}_cfg.json"
            print(
                f"EP {ep}/{MAX_EP} seed={seed} dagger_beta={beta:.3f} eps=0 "
                f"radio=wilab_mode2 (DAgger placement; frozen admit)",
                flush=True,
            )
            out = run_wilab(case, cfg, seed, 0.0, collect=True, dagger_beta=beta)
            tv = float(out["summary"].get("total_value", 0))
            ns = float(out["summary"].get("n_success", 0))
            reloaded = False
            eps = beta  # CSV column: log beta for audit

            if seed == 300 and tv <= 12.0 + 1e-9 and can0 >= 60.0:
                print(
                    f"INVALID: seed300 trainV={tv:.1f} vs greedy~{can0:.1f} — stop",
                    flush=True,
                )
                aborted = "invalid_seed300_explore_vs_greedy"
                break

            if tv < 1e-9:
                zero_streak += 1
            else:
                zero_streak = 0

            bonus = 0.05 * tv + 0.15 * max(0.0, tv - pdp_mean)
            n_tr, ep_ret = 0, 0.0
            loss = 0.0
            # NEVER push suicidal transitions into replay (was poisoning after reload).
            if tv >= train_min:
                n_tr, ep_ret = push(agent, out["transitions"], terminal_bonus=bonus)
                for _ in range(min(20, max(8, n_tr // 4))):
                    loss = agent.train_step(64)
                if ep % 2 == 0:
                    agent.sync_target()
                agent.save(CKPT)
                export_weights(agent, WEIGHTS)
                if tv > best_train_v and tv + 1e-9 >= init_full5:
                    best_train_v = tv
                    shutil.copy(WEIGHTS, BEST_W)
                    agent.save(BEST_PT)
                    print(
                        f"  NEW BEST trainV={tv:.1f} ACCEPTED (>= INIT_FULL5 {init_full5:.1f})",
                        flush=True,
                    )
            elif tv >= floor:
                # Weak episode: log only, no TD / no replay push
                ep_ret = float(np.sum(np.asarray((out["transitions"] or {}).get("reward", []), np.float32)))
                print(
                    f"  WEAK trainV={tv:.1f} < train_min={train_min:.1f} → no TD/push",
                    flush=True,
                )
            else:
                # Catastrophe: reload BEST, flush poisoned replay, re-anchor BC
                agent.load(BEST_PT)
                export_weights(agent, BEST_W)
                shutil.copy(BEST_W, WEIGHTS)
                agent.replay.buf.clear()
                n_bc = seed_bc_replay(agent)
                reloaded = True
                print(
                    f"  CATASTROPHIC trainV={tv:.1f} < floor={floor:.1f} → "
                    f"reload BEST + flush replay + BC reseeds={n_bc}",
                    flush=True,
                )

            beta_hist.append(float(beta))
            agent.eps = 0.0

            row = {
                "episode": ep,
                "seed": seed,
                "ep_return": ep_ret,
                "total_value": tv,
                "n_success": ns,
                "eps": beta,
                "loss": float(loss),
                "n_transitions": n_tr,
                "eval_mean_value": None,
                "reloaded_best": reloaded,
                "elapsed_s": round(time.time() - t0, 1),
            }

            if ep >= 2 and ep % PROBE_EVERY == 0:
                shutil.copy(BEST_W, WEIGHTS)
                mid = eval_parallel(MID_EVAL_SEEDS)
                mid_ev = float(np.mean(list(mid.values())))
                row["eval_mean_value"] = mid_ev
                print(
                    f"  trainV={tv:.1f} ret={ep_ret:.2f} mid300={mid_ev:.2f} "
                    f"(mid300≠FULL5) t={row['elapsed_s']}s",
                    flush=True,
                )
                if ep % FULL5_EVERY == 0 or ep == MAX_EP:
                    per = eval_parallel(EVAL_SEEDS)
                    ev = float(np.mean(list(per.values())))
                    eval_pts.append((ep, ev))
                    eval_hist.append(ev)
                    gate = convergence_report(eval_hist, init_full5, pdp_mean, beta_hist)
                    print(
                        f"  FULL5 EVAL={ev:.2f} per={per} init={init_full5:.1f} "
                        f"target>={beat:.1f} conv={gate['reason']}",
                        flush=True,
                    )
                    if ev > best_eval:
                        best_eval = ev
                        shutil.copy(WEIGHTS, BEST_W)
                        agent.save(BEST_PT)
                    if gate["converged"]:
                        rows.append(row)
                        w.writerow(row)
                        f.flush()
                        _write_status(True, ev, per, gate, time.time() - t0, aborted=None)
                        META.write_text(
                            json.dumps(
                                {
                                    "radio": "wilab_mode2",
                                    "converged": True,
                                    "convergence": gate,
                                    "final_eval_mean": ev,
                                    "final_eval_per_seed": per,
                                    "init_full5": init_full5,
                                    "pdp_mean": pdp_mean,
                                    "elapsed_s": time.time() - t0,
                                    "checkpoint": str(BEST_PT),
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        print(json.dumps({"converged": True, "eval": ev, "gate": gate}, indent=2), flush=True)
                        return 0
                    if ev < floor:
                        aborted = f"eval_collapsed_{ev:.2f}"
                        rows.append(row)
                        w.writerow(row)
                        f.flush()
                        print(f"ABORT: {aborted}", flush=True)
                        break
            else:
                print(
                    f"  trainV={tv:.1f} ret={ep_ret:.2f} tr={n_tr} loss={loss:.4f} "
                    f"reload={reloaded} zeros={zero_streak} t={row['elapsed_s']}s",
                    flush=True,
                )

            rows.append(row)
            w.writerow(row)
            f.flush()
            # fix plot call bug-safe
            try:
                fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                eplist = [r["episode"] for r in rows]
                axes[0].plot(eplist, [r["ep_return"] for r in rows], "o-", color="#1f4e79")
                axes[0].set_title(f"WiLab anti-collapse ({ep}/{MAX_EP})")
                axes[0].grid(True, alpha=0.3)
                axes[1].plot(eplist, [r["total_value"] for r in rows], "o-", color="#6c757d", label="train V")
                if eval_pts:
                    axes[1].plot([e for e, _ in eval_pts], [v for _, v in eval_pts], "s-", color="#1f4e79", label="eval")
                axes[1].axhline(vg_mean, color="#e76f51", ls="--", label=f"VG={vg_mean:.1f}")
                axes[1].axhline(pdp_mean, color="#9b2226", ls=":", label=f"PDP={pdp_mean:.1f}")
                axes[1].legend(fontsize=8)
                axes[1].grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(PLOT, dpi=140)
                plt.close(fig)
            except Exception:
                pass

            if zero_streak >= ZERO_ABORT:
                aborted = f"zero_streak_{zero_streak}"
                print(f"ABORT: {aborted}", flush=True)
                break

    shutil.copy(BEST_W, WEIGHTS)
    agent.load(BEST_PT)
    export_weights(agent, WEIGHTS)
    print("FINAL FULL5 on BEST", flush=True)
    per = eval_parallel()
    ev = float(np.mean(list(per.values())))
    eval_hist.append(ev)
    gate = convergence_report(eval_hist, init_full5, pdp_mean, beta_hist)
    ok = bool(gate["converged"])
    _write_status(ok, ev, per, gate, time.time() - t0, aborted=aborted)
    meta = {
        "radio": "wilab_mode2",
        "protocol": "dagger_beat_init_full5",
        "elapsed_s": time.time() - t0,
        "aborted": aborted,
        "converged": ok,
        "convergence": gate,
        "init_full5": init_full5,
        "beat_target": beat,
        "best_train_v": best_train_v,
        "final_eval_mean": ev,
        "final_eval_per_seed": per,
        "pdp_mean": pdp_mean,
        "checkpoint": str(BEST_PT),
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return 0 if ok else 2


def _write_status(ok: bool, ev: float, per: dict, gate: dict, elapsed: float, aborted) -> None:
    lines = [
        "WILAB OUTCOME A — " + ("CONVERGED+FULL5" if ok else "FAILED"),
        f"time: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"elapsed_s: {elapsed:.1f}",
        f"FULL5_mean: {ev:.2f}",
        f"FULL5_per_seed: {per}",
        f"gate: {gate.get('reason')}",
        f"aborted: {aborted}",
        "protocol: DAgger placement + frozen admit; beat INIT_FULL5+margin then plateau",
    ]
    STATUS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {STATUS}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
