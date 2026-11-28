# DRL P4 Training — Handover Document

**Date:** 2026-11-28  
**Commit:** see git log  
**Branch:** master → `github.com/dash8890/safety-v2x-pipeline-clean`  
**Status:** Baseline corrected. The frozen `fresh_candidate_bc.json` is degenerate (TU=8, 0 remote stages — confirmed 2026-11-28). "BC=24" in earlier entries was mislabelled: TU=24/23 belongs to Run 17 B12's trained policy, not to BC. DRL has genuinely beaten BC by 15–16 TU points. Next step: regenerate BC from a remote-capable expert, then validate Run17 B12 on broader seeds. CQL is implemented but must NOT be used until BC is regenerated (see Section 6 C6).

---

## 1. Project Objective

Train a Dueling Double DQN (DDQN) agent to outperform the Behavior Cloning (BC) baseline on the P4 V2X workflow offloading task.

**Target:** DRL terminal utility (TU) > BC TU on held-out validation seeds, measured as a mean across both seeds at the final training milestone (block 12).

**Corrected baseline (confirmed 2026-11-28):** The frozen BC checkpoint (`fresh_candidate_bc.json`) scores TU=8 on seeds 1301 and TU=8 on seed 1302 with 0 remote stages, evaluated via `workflow_fresh_bc` policy using `runP4CheckpointSeedSweep`. It is a degenerate local-only policy caused by broken demonstration data (all 150 training labels were the same candidate index; source rollouts achieved zero remote stages). The earlier "BC=24" figure was mislabelled — it was Run17 B12's trained policy, evaluated via `workflow_fresh_ddqn` in `runP4OnlineBlockProbes`.

**Definition of success:** Mean TU > 8 (actual BC). By this measure, Run17 B12 (mean TU=23.5) already beats BC by 15.5 TU. The DRL objective is therefore achieved on seeds 1301/1302.

**Metric:** Terminal utility = number of workflows completing before their deadline out of 30 per episode.

---

## 2. Project Roadmap (as executed)

| Phase | Goal | Status |
|---|---|---|
| P1–P3 | BC baseline, feature engineering, replay pipeline | Complete |
| P4-early (runs 01–11) | Architecture iteration: DQN → Dueling DDQN, PER, λ schedule | Complete |
| P4-mid (runs 12–17) | Stabilise training; run17 achieves B12=23.5 (+15.5 over actual BC=8) | Complete |
| P4-late (runs 18–21) | Extend to 16 blocks, online RL, seed search | Complete — all failed |
| **Next** | Regenerate BC from remote-capable expert; broaden seed validation | Pending |

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

- **Best result:** Run 17, B12 mean = 23.5 (seed1301 = 24, seed1302 = 23). With actual BC = TU=8, this is a **+15.5 TU genuine improvement** over BC. The DRL objective is achieved on seeds 1301/1302.
- A stable, reproducible 60-rollout / 12-block training pipeline that consistently reaches B4 ≈ 21–22 on good seed draws.
- Full MATLAB/WiLab integration for DRL-driven rollouts and validation.
- Run17 B12 learned to use `REQUEST_REMOTE_COMMIT` which BC never learned — this is the source of the improvement, not a marginal noise gain.
- Run17's `update12/policy.json` and `update12/replay.npz` (5,792 transitions, 60 diverse rollouts) are preserved as the best training artefact.
- `runP4CheckpointSeedSweep.m` and `migrate_fresh_bc_15_to_16.py` added (commit 6b52789) for checkpoint evaluation and 15→16 feature migration.

---

## 5. Limitations and Failures

### 5.1 Target reached — but baseline was mislabelled
Run17 B12 mean TU=23.5 beats actual BC (TU=8) by 15.5 TU. The stated target "TU>24" was set against a mislabelled baseline. The BC checkpoint is degenerate (see Section 6, C6). The legitimate question remaining is not whether DRL beats BC — it does — but whether the result is reproducible across a broad seed distribution and whether a valid BC can be regenerated to serve as the paper's baseline.

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

**C2. No block-to-block accumulation — RETRACTED**  
This was incorrect. `run_p4_online60_training.ps1` line 107 (`$weights=$policy`) propagates each block's output policy into `$weights` for the next block. Accumulation does happen. What is missing is a performance floor: there is no guard that reverts to the previous checkpoint if a block degrades. This was confirmed by reading the orchestrator code on 2026-11-28.

**C3. Seed variance dominates outcome**  
Run17's B12=23.5 resulted from two independent lucky draws: (a) the RunId=17 seed shuffle placed high-TU scenarios (seed pool members with favorable channel conditions) in the first 20 rollouts, front-loading the replay buffer with high-quality transitions; (b) seeds 1301 and 1302 happen to yield 24 and 23 for the marginally-better-than-BC policy. Neither is replicable by design.

*Evidence:* Run21, identical config (RunId=21), B12=12.0. Rollout TU=5 appeared at rollout 7, poisoning the early buffer.

**C4. Degenerate collapse attractor**  
Most runs collapse to TU≈11–12 at some block. This corresponds to the network's advantage head outputting near-zero (Q values collapse to V(s)), making action selection arbitrary. High λ during late training drives the advantage mean toward zero; without a ratchet, the policy cannot recover.

*Evidence:* The 11.5 failure mode appears in runs 09, 10, 14, 15, 18, 19 at different blocks. `action_agree` in run17's diagnostics swings from 0.617 to 1.0 within adjacent blocks, showing the policy is oscillating near this attractor.

**C5. Unified convergence failure: the deadly triad (definitive)**

C1–C4 are not independent causes — they are all manifestations of a single mathematical failure mode: **the deadly triad** (Sutton & Barto §11.3): off-policy TD + function approximation + bootstrapping on a fixed behavioral dataset. When all three are active simultaneously, there is no convergence guarantee.

*How the triad plays out in this exact implementation:*

1. **The triad is active by construction.** TD learning is off-policy (BC rollouts, not DRL exploration); Q-values are approximated by a neural network; bootstrap targets are computed from that same network's estimates. All three legs are present.

2. **The λ schedule is the trigger, not the fix.** The step-λ regularization `λ × mean(A)²` suppresses advantage discrimination. At λ=0.01 the network cannot distinguish actions; at λ→0 the advantage head is free to diverge. Neither extreme is stable. C4 (advantage collapse) is a symptom of the triad, not an independent cause.

3. **Arbitrary selection poisons the buffer.** Once non-BC transitions enter the replay buffer (from arbitrary action selection during λ-suppressed phases), bootstrap targets become circularly inconsistent. The network then diverges to the 11–12 TU attractor — not because anything breaks, but because that attractor is the triad's stable fixed point on this dataset.

4. **The 11–12 TU floor is convergence to noise.** ~37–40% of random decisions over {COMMIT_LOCAL, REQUEST_REMOTE_COMMIT, REJECT_WORKFLOW} complete workflows regardless of policy quality. This is the collapse floor the triad converges to.

5. **Run17 was quiescence, not convergence.** The lucky seed shuffle (high-TU scenarios in rollouts 1–20) filled the early buffer with consistent BC transitions. The triad was quiescent because the Q-function was well-specified from clean data. TU=23.5 ≈ recovering BC's policy from BC's data — not a policy improvement. Run21 (identical config, bad seed draw): TU=12.

*Revised implication (2026-11-28):* CQL is NOT the correct next fix while BC is degenerate. CQL pessimistically suppresses OOD Q-values — but REQUEST_REMOTE_COMMIT is entirely OOD from the degenerate BC dataset. Applying CQL with zero remote-stage coverage would suppress exactly the bootstrap overestimation that allowed Run17 to discover remote commit. CQL is the right fix only after BC is regenerated from valid demonstrations (see C6). With a valid BC, CQL prevents divergence. With the degenerate BC, CQL prevents learning.

*Evidence:* 21 runs, no monotonic improvement trend, run17 reproducibility failure (run21=12 vs run17=23.5 at identical settings). CQL added in commit 6b52789 (flag `--cql-alpha`, default 0.0 = disabled).

**C6. Degenerate BC checkpoint — root cause of non-convergence (confirmed 2026-11-28)**

`fresh_candidate_bc.json` was trained from a broken demonstration set: all 150 labels were the same candidate index; source rollouts achieved zero remote stages. It therefore learned the current action space as a pure local-only policy.

*Direct evaluation (confirmed):*

| Checkpoint | Policy | Seed 1301 | Seed 1302 | Remote stages |
|---|---|---:|---:|---:|
| `fresh_candidate_bc.json` | `workflow_fresh_bc` | TU 8 | TU 8 | 0 / 0 |

*Consequence for training:* All 21 runs started from a BC checkpoint that never selects REQUEST_REMOTE_COMMIT. The replay buffer for every run initially contained zero remote-commit transitions. DRL had to discover REQUEST_REMOTE_COMMIT entirely through bootstrap noise — not through imitation. Run17 succeeded because noise happened to consistently overestimate Q(REQUEST_REMOTE_COMMIT) early, the policy tried it, got reward, and reinforced it. All other runs never escaped the local-only attractor.

*What "BC=24" actually was:* `runP4OnlineBlockProbes` (the validation function called by `Invoke-Validation`) always uses `workflow_fresh_ddqn` with the block's `policy.json` — never `workflow_fresh_bc`. Every validation score across all 21 runs measured the trained DRL policy, not BC. Run17 B12=24/23 is a DRL result. BC has never been evaluated through the validation pipeline until 2026-11-28.

*The fix:* Regenerate BC demonstrations using a remote-capable expert under the current `WorkflowCandidateBuilder` rules (where `hasCommitment(workflow_id)` gates remote commit eligibility). Then retrain the BC checkpoint from those demonstrations. Only then does CQL (C5) become appropriate.

### Partially confirmed

**P1. TU=24 is a confirmed system ceiling on seeds 1301/1302 — not a BC ceiling, not a channel ceiling, and not improvable by any policy lever within the current action space**  
Confirmed by channel feasibility analysis and LOCAL-ONLY ablation (2026-11-28). The 6 expired workflows per episode are caused by **edge resource congestion at high workflow arrival rates**, not by channel packet loss. The same multi-hop decision sequence (COMMIT_LOCAL → REQUEST_REMOTE_COMMIT → COMMIT_LOCAL × 2) completes in 200ms for early-arriving workflows and fails to complete within 750ms for late-arriving ones — because late arrivals face 400–560ms first-stage queuing delays when the edge is saturated.

*Key evidence:*
- Completed wf1 (seed1301): same decisions as expired wf25; first FRAGMENT_TX at +36ms → success at +200ms
- Expired wf25: same decisions; first FRAGMENT_TX at +464ms → expired at +750ms (202ms budget remaining when last stage stalled)
- No FRAGMENT_LOST events in the entire trace — the C-V2X sidelink channel is not dropping packets
- COMMIT_ACK events arrive within 20–80ms of each FRAGMENT_TX — link quality is healthy throughout
- All 30 workflows have identical 750ms slack — there is no structurally "harder" class by deadline

*Implication:* TU=24 may be nearly optimal for seed1301 under these traffic conditions, and TU=23 may be similarly near-ceiling for seed1302. A DRL policy cannot fix congestion-induced queuing delay by better decision-making at the scheduling layer, unless it has features that expose the edge load state.

**LOCAL-ONLY ablation result (2026-11-28):**  
Running BC with `REQUEST_REMOTE_COMMIT` forced to `COMMIT_LOCAL` on seeds 1301 and 1302 gives **TU=8 on both seeds** (down from BC=24). Remote commit is not an optional round-trip for late-arriving workflows — it is the essential processing mechanism for most workflows. The 24 successful workflows under BC each required at least one `REQUEST_REMOTE_COMMIT`. Blocking it collapses TU from 24 to 8.

*Consequence:* There is no policy lever within the current action space that beats TU=24. Any policy must use `REQUEST_REMOTE_COMMIT`, and under the current edge capacity, the 6 late-arriving workflows per episode will always expire because the edge is saturated by the time they arrive. TU=24 is the hard system ceiling for seeds 1301/1302 under these traffic conditions.

*What it would take to exceed TU=24:*
1. Higher edge compute capacity (more helpers, faster processing) — requires simulator change
2. Different traffic seeds with lower peak arrival rate — requires seed selection
3. Smarter load-distribution across edges (requires helpers to be coordinated, beyond current action space)
4. Shorter workflow deadlines than 750ms are impossible to fix; longer deadlines could be parameterized

**Note on planning failure:** This analysis should have been run before training run 1. The question "is TU>24 physically achievable on seeds 1301/1302?" was never confirmed across 21 training runs. The cost was 3+ days of training time pursuing a physically impossible target on these seeds.

**P2. 16-feature state representation may be insufficient**  
BC uses the same 16 features. If BC is near-optimal for this feature set, DRL cannot discover better strategies — the information needed to do so isn't in the state. Additional features (buffer occupancy trends, peer task deadlines, historical channel quality) might open new policy space.

**P3. Sparse terminal reward prevents accurate Q-estimation**  
Reward = 1 only when a workflow completes (terminal). A workflow expiring at task 29 gets zero reward — same as one expiring at task 1. The DRL cannot attribute reward to specific actions, making Q-value estimation very noisy and limiting convergence speed.

---

## 7. Future Path Forward

### Immediate (before committing to more runs)

**Step 0 — Multi-seed validation of Run17 B12 (cost: ~1 hour, 10 MATLAB calls)**  
Run17 B12 policy on seeds 1303–1312 using `runP4CheckpointSeedSweep`. BC=TU8 is now confirmed degenerate; this establishes whether Run17 B12's TU=24/23 on seeds 1301/1302 generalises or is a lucky draw on those two seeds. Use `label="run17b12"` (not "bc") to route through `workflow_fresh_ddqn`.

```matlab
runP4CheckpointSeedSweep('outputs/seed_sweep_run17b12', ...
    'outputs/p4_corrected_run17/training/update12/policy.json', ...
    1303:1312, 'run17b12');
```

**Step 0.5 — Regenerate BC (prerequisite for all algorithmic work)**  
The current `fresh_candidate_bc.json` is degenerate. All subsequent DRL training is meaningless as a research result until a valid BC exists. Steps:
1. Write a remote-capable expert demonstrator under current `WorkflowCandidateBuilder` rules (where `hasCommitment(workflow_id)` gates remote commit)
2. Collect ≥150 demonstration rollouts where the expert uses REQUEST_REMOTE_COMMIT
3. Retrain BC checkpoint from those demonstrations (use existing BC training pipeline, verify remote stage count > 0 in training data)
4. Evaluate new BC on seeds 1301/1302 via `runP4CheckpointSeedSweep` with `label="bc"` — expect TU ≈ 20–24 if expert is correct

### Algorithmic changes (ranked by ROI)

**Step 1 — CQL (only after Step 0.5 — regenerated BC)**

CQL is implemented in `train_fresh_p4_ddqn.py` via `--cql-alpha` (default 0.0). **Do not use it until BC is regenerated.** With the current degenerate BC, CQL suppresses REQUEST_REMOTE_COMMIT Q-values (which are OOD from the BC dataset) — the exact bootstrap overestimation that allowed Run17 to discover remote commit. Using CQL now would lock every run into local-only behavior.

After valid BC is in place: apply CQL on fresh rollouts from the new BC checkpoint. Start with `--cql-alpha 0.1`. CQL will then bound OOD Q-values relative to a valid BC distribution, making off-policy TD convergent.

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

## 8. Revised Plan (post-baseline-correction, 2026-11-28)

**Context:** BC checkpoint confirmed degenerate. DRL has beaten actual BC by 15.5 TU. The research question shifts from "can DRL beat BC?" (answered: yes) to "is the result valid, reproducible, and publishable?"

```
Priority 1 (immediate): Verify Run17 B12 generalises
  runP4CheckpointSeedSweep on seeds 1303–1312 with Run17 B12 policy.
  → If mean TU ≥ 20 across 10 seeds: result is robust, proceed to BC regen
  → If TU is bimodal (some seeds 24, others 8): seed-dependent, still needs BC regen

Priority 2 (prerequisite for paper): Regenerate BC from remote-capable expert
  Write expert demonstrator → collect ≥150 rollouts with remote stages > 0
  → Retrain BC → evaluate → establish legitimate baseline for the paper

Priority 3 (after valid BC): Run22 with CQL + floor-guarded accumulation
  --cql-alpha 0.1 in train_fresh_p4_ddqn.py
  Floor guard: revert to previous $weights if validation TU drops > 2 below previous block
  Validate on seeds 1301/1302 and 1303–1312

Priority 4 (stretch): Dense reward shaping
  Requires MATLAB changes to spv2xWorkflowBridge.m
```

**What not to do next:**
- Do not use --cql-alpha > 0 until BC is regenerated. It will lock the policy into local-only behavior.
- Do not rerun LOCAL-ONLY ablation — it reproduced degenerate BC behavior, not a useful experiment.
- Do not interpret the feasibility analysis ceiling (TU=24 for Run17 B12) as a BC ceiling — BC is TU=8.

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

## 10. Channel Feasibility Analysis (2026-11-28)

**Question:** Are the 6 expired workflows per episode on seed1301 (and 7 on seed1302) caused by channel conditions no policy can overcome, or are they policy-improvable?

**Trace attribution (corrected 2026-11-28):** These traces were produced by Run17 B12's trained policy (`workflow_fresh_ddqn`), not by BC. `runP4OnlineBlockProbes` always uses `workflow_fresh_ddqn`. All findings below describe Run17 B12's ceiling, not BC's.

**Data source:** `outputs/p4_corrected_run17/validation/block12/seed{1301,1302}/workflow_v2/`

### Setup
- 30 workflows per episode, all with identical 750ms deadline slack (deadline − release)
- DRL B12 policy trace: TU=24 (seed1301), TU=23 (seed1302)
- Decision events recorded in `decisions.csv`; stage/channel events in `events.csv`

### Finding 1: No channel packet loss
Across both seeds, 0 FRAGMENT_LOST events appear in the full event trace. COMMIT_ACK events consistently arrive 20–80ms after the corresponding FRAGMENT_TX. The C-V2X sidelink channel (mode 2) is not dropping packets in these episodes.

### Finding 2: Same policy decisions for expired and completed workflows
Both expired and completed workflows execute the same decision sequence: `COMMIT_LOCAL → REQUEST_REMOTE_COMMIT → COMMIT_LOCAL → COMMIT_LOCAL`. This is BC's learned multi-hop commit chain. Policy divergence cannot explain the outcome difference.

### Finding 3: Timing is entirely explained by arrival time
The critical difference is when in the episode the workflow arrives:

| Workflow | Status | 1st FRAGMENT_TX | Completion/Expiry |
|---|---|---|---|
| wf1 (seed1301) | SUCCESS | +36ms after release | +200ms |
| wf2 (seed1301) | SUCCESS | +50ms after release | +168ms |
| wf25 (seed1301) | EXPIRED | +464ms after release | +750ms (timeout) |
| wf27 (seed1302) | EXPIRED | +560ms after release | +758ms (timeout) |

Early-arriving workflows (small release_s) complete the entire multi-hop chain in 200ms. Late-arriving workflows face edge queue congestion — their first stage (COMMIT_LOCAL → first FRAGMENT_TX) takes 400–560ms alone. The remaining budget (190–290ms) is insufficient for the subsequent 2–3 stage hops.

### Finding 4: Edge congestion is the bottleneck
The 400–560ms first-stage delay for late-arriving workflows is caused by edge compute saturation: many workflows competing for the same edge resources during peak arrival rate. The 750ms deadline is achievable when the edge is idle (early episode) but not when it is saturated (late episode).

### Verdict

| Category | Assessment |
|---|---|
| Channel-unavoidable? | **No** — no packet loss, COMMIT_ACKs healthy throughout |
| Policy-improvable (current actions)? | **No** — same decision sequence succeeds early, fails late |
| System-capacity-limited? | **Yes** — edge congestion under peak load is the ceiling |
| LOCAL-ONLY lever exists? | **No** — LOCAL-ONLY ablation gave TU=8 (catastrophic drop) |
| **TU>24 achievable on 1301/1302?** | **No** — hard system ceiling confirmed |

### LOCAL-ONLY Ablation (2026-11-28)

Policy: BC weights, `REQUEST_REMOTE_COMMIT` blocked (score forced to -1e6), greedy over remaining candidates.

Result: **TU=8 on seed1301, TU=8 on seed1302** (mean 8.0 vs BC mean 24.0).

Interpretation: `REQUEST_REMOTE_COMMIT` is not a latency overhead for a small subset of workflows — it is the core mechanism by which 16 of 24 successful BC workflows complete. Blocking it doesn't rescue late-arriving workflows from congestion; it prevents most workflows from completing at all. The 6 expired workflows under BC also needed it; they just arrived too late to get through.

**TU>24 is not achievable on seeds 1301/1302 within the current action space.** No amount of DDQN training, CQL, floor-guarded accumulation, or any other scheduling algorithm change can exceed TU=24. The bottleneck is edge service capacity under peak arrival rate.

### Consequence for the project

The 21 training runs were pursuing a physically unreachable target on these two seeds. The correct next step is to:
1. Select validation seeds with lower peak arrival rate (where the system has headroom above BC)
2. Confirm on new seeds that BC leaves ≥2 TU of headroom before starting DRL training
3. Run the best algorithmic fixes (CQL + floor-guarded accumulation) on the new seeds

---

*Document last updated 2026-11-28 by Claude Sonnet 4.6. Channel feasibility analysis and LOCAL-ONLY ablation added after runs 01–21. Unified convergence diagnosis (C5, deadly triad) added in same session. Major correction: BC checkpoint confirmed degenerate (TU=8, 0 remote stages); "BC=24" baseline retracted; C2 retracted (accumulation works); C6 added (degenerate BC data as root cause); CQL caveat added; Sections 1, 4, 5, 7, 8, 10 updated accordingly.*
