# DRL P4 Training — Handover Document

**Date:** 2026-11-28  
**Commit:** 6156e21  
**Branch:** main → `github.com/dash8890/safety-v2x-pipeline-clean`  
**Status:** Algorithmic plateau reached. Seed search exhausted. Next phase requires structural change.

---

## 1. Project Objective

Train a Dueling Double DQN (DDQN) agent to outperform the Behavior Cloning (BC) baseline on the P4 V2X workflow offloading task.

**Target:** DRL terminal utility (TU) > BC TU = 24/30 on held-out validation seeds 1301 and 1302, measured as a mean across both seeds at the final training milestone (block 12).

**Definition of success:** Mean TU > 24 (strictly). A tie (mean = 24) or any score below has no value and does not constitute improvement over BC.

**Metric:** Terminal utility = number of workflows completing before their deadline out of 30 per episode. BC achieves TU = 24 on seeds 1301 (TU=24) and 1302 (TU=24) under identical evaluation conditions.

---

## 2. Project Roadmap (as executed)

| Phase | Goal | Status |
|---|---|---|
| P1–P3 | BC baseline, feature engineering, replay pipeline | Complete |
| P4-early (runs 01–11) | Architecture iteration: DQN → Dueling DDQN, PER, λ schedule | Complete |
| P4-mid (runs 12–17) | Stabilise training; run17 achieves B12=23.5 (best result) | Complete |
| P4-late (runs 18–21) | Extend to 16 blocks, online RL, seed search | Complete — all failed |
| **Next** | Structural change: offline RL (CQL) or floor-guarded accumulation | Pending |

---

## 3. What Has Been Done

### Training infrastructure
- **`run_p4_online60_training.ps1`** — full 60-rollout / 12-block orchestrator:
  - Persistent MATLAB worker process (no per-rollout startup cost)
  - Fisher-Yates shuffled seed pool 83001–83060, keyed by `RunId × 7919`
  - Startup stagger bypass (`stagger_done.flag`)
  - Resume detection (block + rollout level)
  - Validation at blocks 4, 8, 12 on seeds 1301 and 1302
  - `RunId` range 1–99 via `-RunId N`

- **`run_p4_online80_training.ps1`** — 80-rollout / 16-block variant (deprecated; kept for reference)

- **`drl/train_fresh_p4_ddqn.py`** — Dueling DDQN trainer:
  - Architecture: encoder 16→96→ReLU, value head 96→64→1, advantage head 192→64→1
  - PER with guaranteed rare-transition share (25% of batch)
  - Batch cap: `min(2000, max(512, len(data)//2))`
  - Step-λ BC-retention schedule: flat 0.01 for global updates 0–239, linear decay to 0 for 240–479
  - `--bc-weight 0.0` (BC regularisation disabled; BC used only as weight initialiser)
  - `--total-updates 480`, `--global-update-start (block-1)*40`

- **`drl/evaluate_dueling_convergence.py`** — generalised convergence checker:
  - Supports arbitrary N milestones via `--milestones`
  - Three-stage test: progressive learning (M2 ≥ 1.2×M1), policy refinement (Mn ≥ 0.95×Mn-1), stable convergence (CI lower bound, action agreement ≥ 0.90)

- **MATLAB policy inference** — `WorkflowPolicyCausalDDQN.m`, `WorkflowFreshCandidateFeatures.m`: reads `policy.json`, scores candidates, dispatches decisions to WiLab

- **WiLab integration** — persistent worker bridge (`run_p4_matlab_worker.ps1`), observation builders, safety recording, feasibility and scheduling helpers

### Runs executed (21 total)

| Run | B4 | B8 | B12 | B16 | Notes |
|---|---|---|---|---|---|
| 01 | 21.0 | 12.5 | 8.0 | 12.0 | collapse at B8 |
| 04 | 16.0 | 9.0 | 8.0 | — | collapse at B8 |
| 08 | 0.0 | 0.0 | 0.0 | — | initialisation failure |
| 09 | 11.5 | 19.0 | 10.0 | — | peak at B8 then collapse |
| 10 | 11.5 | 20.0 | 11.0 | — | same pattern |
| 11 | 22.0 | 19.5 | 22.5 | — | stable, non-monotonic |
| 12 | 12.5 | 8.0 | 8.0 | — | early collapse |
| 13 | 21.5 | 23.0 | 8.0 | — | collapse exactly at B12 |
| 14 | 20.0 | 11.5 | 11.5 | — | collapse at B8 |
| 15 | 22.5 | 20.0 | 11.5 | — | collapse at B12 |
| 16 | 20.0 | 13.5 | 20.0 | — | non-monotonic, partial recovery |
| **17** | **21.0** | **22.5** | **23.5** | — | **best result, monotonic** |
| 18 | 11.5 | 11.5 | 11.5 | 16.0 | flat (λ schedule bug: totalUpdates=640) |
| 19 | 22.5 | 22.0 | 20.5 | 11.5 | promising then collapse at B16 |
| 20 | 21.5 | 19.0 | 16.0 | 24.0 | online RL divergence; B16 recovery artefact |
| 21 | 8.0 | 15.5 | 12.0 | — | bad seed draw; confirms plateau |

---

## 4. What Has Been Achieved

- **Best result:** Run 17, B12 mean = 23.5 (seed1301 = 24, seed1302 = 23). This is 0.5 TU below target.
- A stable, reproducible 60-rollout / 12-block training pipeline that consistently reaches B4 ≈ 21–22 on good seed draws.
- Full MATLAB/WiLab integration for DRL-driven rollouts and validation.
- Identified the precise architectural and training constraints that prevent exceeding BC (see Section 6).
- Run17's `update12/policy.json` and `update12/replay.npz` (5,792 transitions, 60 diverse BC rollouts) are preserved as the best training artefact.

---

## 5. Limitations and Failures

### 5.1 Target not reached
No run achieved mean TU > 24. The closest is run17 at 23.5, which is 0.5 short.

### 5.2 High variance, no upward trend
B12 scores across runs: 8, 8, 0, 10, 11, 22.5, 8, 8, 11.5, 11.5, 20, **23.5**, 11.5, 20.5, 16, 12. No monotonic improvement. Run17 is an outlier, not evidence of algorithmic progress.

### 5.3 80-rollout / 16-block extension failed (runs 18, 19, 20)
All three attempts with 16 blocks collapsed at or after B12. Causes: larger replay buffer diversity (more conflicting gradients), λ schedule misconfiguration (run18), online RL instability (run20), late-seed degradation amplified by buffer size (run19).

### 5.4 Online RL failed (run 20)
Enabling `$script:weights = $policy` (true sequential policy propagation) caused immediate divergence. Without a performance floor, early DRL policies weaker than BC polluted the replay buffer, triggering a negative feedback loop. B12 degraded to 16.0.

### 5.5 Statistically insufficient validation
Two validation seeds (1301, 1302) cannot distinguish signal from noise. TU=23.5 vs TU=24.0 at n=2 is not statistically meaningful. The target TU>24 may itself be a two-seed artefact.

---

## 6. Root Causes — Confirmed and Possible

### Confirmed

**C1. BC-always training data (OOD problem)**  
Every transition in the replay buffer comes from BC rollouts. The DRL policy is trained on BC's decision distribution and cannot safely deviate from it: Q-values for out-of-distribution actions are unconstrained bootstrap estimates with no ground truth. This is the textbook offline RL problem. The DRL can find small perturbations of BC but cannot explore BC's policy space boundary.

*Evidence:* Run17 improved +2.5 TU over 480 total gradient steps starting from BC. The improvement saturated — no further gains from additional blocks or seed search.

**C2. No block-to-block accumulation**  
Each optimizer block re-initialises from BC weights (`$weights` in function scope, never updated). The 480 total gradient steps are 12 independent 40-step fine-tunes from BC, not one 480-step trajectory through policy space. Effective fine-tuning depth: 40 epochs, not 480.

*Evidence:* `td_loss_first` decreases B1→B12 (0.303→0.169) solely because the replay buffer grows and BC fits a larger dataset better — not because the policy accumulated anything across blocks.

**C3. Seed variance dominates outcome**  
Run17's B12=23.5 resulted from two independent lucky draws: (a) the RunId=17 seed shuffle placed high-TU scenarios (seed pool members with favorable channel conditions) in the first 20 rollouts, front-loading the replay buffer with high-quality transitions; (b) seeds 1301 and 1302 happen to yield 24 and 23 for the marginally-better-than-BC policy. Neither is replicable by design.

*Evidence:* Run21, identical config (RunId=21), B12=12.0. Rollout TU=5 appeared at rollout 7, poisoning the early buffer.

**C4. Degenerate collapse attractor**  
Most runs collapse to TU≈11–12 at some block. This corresponds to the network's advantage head outputting near-zero (Q values collapse to V(s)), making action selection arbitrary. High λ during late training drives the advantage mean toward zero; without a ratchet, the policy cannot recover.

*Evidence:* The 11.5 failure mode appears in runs 09, 10, 14, 15, 18, 19 at different blocks. `action_agree` in run17's diagnostics swings from 0.617 to 1.0 within adjacent blocks, showing the policy is oscillating near this attractor.

### Possible (not confirmed)

**P1. BC may be near-optimal for seeds 1301/1302**  
If the 6 failed workflows per episode (TU=24/30) are caused by channel conditions no policy can overcome (unavoidable packet loss windows), then TU=24 is the physical ceiling, not a BC ceiling. DRL cannot exceed a physical limit.

*How to test:* Run BC on 10+ seeds. If BC scores vary between 20 and 27, the ceiling is learnable. If BC consistently scores 23–25 across seeds, TU≈24 may be near-optimal.

**P2. 16-feature state representation may be insufficient**  
BC uses the same 16 features. If BC is near-optimal for this feature set, DRL cannot discover better strategies — the information needed to do so isn't in the state. Additional features (buffer occupancy trends, peer task deadlines, historical channel quality) might open new policy space.

**P3. Sparse terminal reward prevents accurate Q-estimation**  
Reward = 1 only when a workflow completes (terminal). A workflow expiring at task 29 gets zero reward — same as one expiring at task 1. The DRL cannot attribute reward to specific actions, making Q-value estimation very noisy and limiting convergence speed.

---

## 7. Future Path Forward

### Immediate (before committing to more runs)

**Step 0 — Multi-seed validation (cost: ~2 hours, 20 MATLAB calls)**  
Run BC and run17's B12 policy on seeds 1303–1312. This establishes: (a) whether BC's TU=24 on seeds 1301/1302 is representative or an outlier, (b) whether run17's policy is actually better than BC in expectation, (c) the size of the gap worth pursuing. If BC scores 23–25 and run17 scores 21–24 across 10 seeds, the gap is real and small. If they're interleaved, the framing needs rethinking.

Script: instantiate `Invoke-Validation`-style evaluation with seeds 1303–1312 for both `fresh_candidate_bc.json` and `outputs/p4_corrected_run17/training/update12/policy.json`.

### Algorithmic changes (ranked by ROI)

**Step 1 — Offline RL (CQL) on run17's existing data (no new MATLAB rollouts)**

Apply Conservative Q-Learning to `outputs/p4_corrected_run17/training/update12/replay.npz` (5,792 transitions). CQL explicitly penalises Q-value overestimation for out-of-distribution actions (root cause C1), making it safe to train on a fixed offline dataset. If CQL on run17's data cannot exceed BC on seeds 1301/1302, that proves the data distribution itself is the ceiling. If it can, a better algorithm has been found without running a single new simulation.

Implementation: modify `train_fresh_p4_ddqn.py` to add a CQL penalty term:
```python
# After computing qa and y:
cql_penalty = -torch.logsumexp(q(cur), dim=0).mean() + qa.mean()
loss = F.smooth_l1_loss(qa, y) + lam * _adv.mean()**2 + alpha * cql_penalty
```
Start with `alpha=0.1`, validate on seeds 1301/1302 after training.

**Step 2 — Floor-guarded block accumulation (structural fix for C1 + C2)**

Change `run_p4_online60_training.ps1`:
1. Remove BC reinit: `$weights` starts as BC, then propagates the previous block's `$policy` if it passes the floor check.
2. Add a floor check after each optimizer block: run a quick 2-seed validation (seeds 1301, 1302). If `mean_TU >= BC_TU - margin` (margin = 2), accept `$policy` as the new `$weights`. Otherwise revert to BC for that block.

This creates a ratchet: the policy can only improve, never catastrophically degrade. Cost: 2 extra MATLAB calls per block (24 extra calls per run, ~1.5h overhead).

**Step 3 — Dense reward shaping (requires MATLAB changes)**

Augment the per-step reward with intermediate signals:
- Partial completion bonus: +r_partial for each task successfully offloaded before deadline
- Deadline proximity penalty: -r_deadline × (remaining_deadline / max_deadline) for tasks at risk
- Channel utilization bonus: +r_channel for offloading during favorable channel windows

This makes Q-value estimation more accurate and addresses possible root cause P3. Requires modification of `spv2xWorkflowBridge.m` and the replay builder.

**Step 4 — Richer state features (if Steps 1–3 fail)**

Add features to `WorkflowFreshCandidateFeatures.m`:
- Current channel congestion trend (CBR delta over last 3 steps)
- Peer task deadline proximity distribution
- Rolling offload success rate over last N tasks

If BC is near-optimal for the current 16-feature state, adding features that BC doesn't use gives DRL a genuine informational advantage.

---

## 8. Clear Plan Towards Convergence

```
Week 1
  Day 1: Multi-seed validation (Steps 0)
         → if BC=run17 interleaved: reconsider target definition
         → if gap confirmed: proceed

  Day 2: CQL offline RL on run17/update12/replay.npz (Step 1)
         → evaluate on seeds 1301, 1302 (and 1303-1312)
         → if CQL > BC: found the lever, run full CQL-driven training
         → if CQL ≤ BC: data distribution is the ceiling, must fix rollout policy

  Day 3-4: Floor-guarded block accumulation (Step 2)
           → run22 with floor guard enabled, RunId=22
           → target: B4→B8→B12 monotonic, B12 mean > 24

Week 2 (if floor guard run doesn't converge)
  → Combine CQL loss + floor-guarded accumulation
  → Consider adding one dense reward signal (partial task completion)
  → Run validation on 10+ seeds to assess statistical significance

Week 3+ (if still below target)
  → Investigate whether TU=24 is physically optimal (root cause P1)
  → If yes: reframe — DRL value is in generalisation across seeds, not on 1301/1302
  → Add features (Step 4)
  → Consider policy gradient (PPO) as alternative to Q-learning
```

---

## 9. Key File Locations

| Item | Path |
|---|---|
| Best DRL policy | `outputs/p4_corrected_run17/training/update12/policy.json` |
| Best replay buffer | `outputs/p4_corrected_run17/training/update12/replay.npz` |
| BC policy | `drl/outputs/fresh_candidate_bc.json` |
| 60-rollout training script | `run_p4_online60_training.ps1` |
| DDQN trainer | `drl/train_fresh_p4_ddqn.py` |
| Convergence evaluator | `drl/evaluate_dueling_convergence.py` |
| Replay builder | `drl/build_fresh_p4_replay.py` |
| MATLAB DRL policy | `policy/WorkflowPolicyCausalDDQN.m` |
| Run17 diagnostics | `outputs/p4_corrected_run17/convergence_diagnostics.json` |

---

*Document generated 2026-11-28 by Claude Sonnet 4.6 at end of runs 01–21.*
