# Session Starter — DRL P4 V2X Project

**Purpose:** Self-contained briefing for a new agent/session. No prior context needed.  
**Written:** 2026-11-28  
**Repo state at handoff:** commit `6b52789`, branch `master`, remote `https://github.com/dash8890/safety-v2x-pipeline-clean`

---

## 1. What This Project Is

A research project to train a **Dueling Double DQN (DDQN)** agent to outperform a **Behavior Cloning (BC)** baseline on a **C-V2X workflow offloading** task in a 5G NR-V2X vehicular network simulation (WiLab V2X simulator, MATLAB). The goal is to include this result in a research paper.

**The metric:** Terminal Utility (TU) = number of workflows (out of 30 per episode) that complete before their 750ms deadline.

**Working directory:** `C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline`  
**GitHub remote:** `https://github.com/dash8890/safety-v2x-pipeline-clean` (branch: `master`)  
**Key subdirectory for all DRL work:** `SafetyOffloading/workflow_v2/`

---

## 2. Confirmed State (as of 2026-11-28)

### 2.1 The BC checkpoint is degenerate

`drl/outputs/fresh_candidate_bc.json` was trained from a broken demonstration set: all 150 labels were the same candidate index; source rollouts achieved zero remote stages. It is a pure local-only policy.

**Direct evaluation result:**

| Checkpoint | Policy | Seed 1301 | Seed 1302 | Remote stages |
|---|---|---:|---:|---:|
| `fresh_candidate_bc.json` | `workflow_fresh_bc` | TU 8 | TU 8 | 0 / 0 |

Evaluated via `runP4CheckpointSeedSweep.m` (committed 6b52789).

### 2.2 "BC=24" was mislabelled — DRL has beaten BC

The validation function `runP4OnlineBlockProbes.m` (line 16) always uses `workflow_fresh_ddqn` with the block's `policy.json`. It never evaluates `fresh_candidate_bc.json`. Every "BC=24" figure in earlier documentation was actually Run17 B12's trained DRL policy.

**Actual picture:**
- Real BC = TU=8 (degenerate, local-only)
- Run17 B12 = TU=24 (seed1301), TU=23 (seed1302) — a genuine DRL result
- DRL has beaten BC by **+15 to +16 TU**. The research objective is achieved on seeds 1301/1302.

### 2.3 Why DRL doesn't converge reliably (unified answer)

BC has zero coverage of `REQUEST_REMOTE_COMMIT` (zero remote stages in training data). DRL starts from a degenerate initialization where Q(REQUEST_REMOTE_COMMIT) is pure bootstrap noise.

- Run17 succeeded because noise happened to consistently overestimate Q(REQUEST_REMOTE_COMMIT) early → policy tried it → got reward → reinforced it (lucky positive feedback loop)
- All other runs: noise was insufficient → never escaped local-only attractor → TU≈11–12
- This is the deadly triad (Sutton & Barto §11.3: off-policy TD + function approximation + bootstrapping) operating on a dataset with zero coverage of the high-value action

**CQL is counterproductive until BC is regenerated.** CQL pessimistically suppresses OOD Q-values — which means it suppresses REQUEST_REMOTE_COMMIT Q-values (entirely OOD from degenerate BC). Using `--cql-alpha > 0` now would lock every run into local-only behavior. `train_fresh_p4_ddqn.py` has CQL implemented via `--cql-alpha` (default 0.0 = disabled) — leave it disabled.

### 2.4 Feasibility analysis — traces are Run17 B12, not BC

The channel feasibility analysis (TU=24 ceiling on seeds 1301/1302 due to edge congestion) was run on Run17 B12 traces. The finding is still valid: Run17 B12 hits a wall at TU=24 because 6 late-arriving workflows face 400–560ms queuing delay that exhausts the 750ms deadline. This is a ceiling for Run17 B12's policy, not BC's ceiling.

### 2.5 Block accumulation works

`run_p4_online60_training.ps1` line 107 (`$weights=$policy`) propagates each block's output policy into `$weights` for the next block. The handover's earlier claim "no accumulation" was wrong. What's missing is a performance floor guard — no revert to previous checkpoint if a block degrades.

---

## 3. Correct Next Steps (in order)

### Step 1 — Verify Run17 B12 generalises across seeds

```matlab
runP4CheckpointSeedSweep('outputs/seed_sweep_run17b12', ...
    'outputs/p4_corrected_run17/training/update12/policy.json', ...
    1303:1312, 'run17b12');
```

If mean TU ≥ 20 across seeds 1303–1312: Run17 B12 is robust. If bimodal (some seeds 24, others 8): seed-dependent, but still worth publishing with the BC regen.

### Step 2 — Regenerate BC from a remote-capable expert (prerequisite for paper)

The paper needs a legitimate BC baseline. Steps:
1. Write a remote-capable expert demonstrator under current `WorkflowCandidateBuilder` rules — specifically, the expert must select `REQUEST_REMOTE_COMMIT` after `hasCommitment(workflow_id)` is true
2. Collect ≥150 demonstration rollouts where remote stage count > 0
3. Retrain BC from those demonstrations using the existing BC training pipeline
4. Verify: `runP4CheckpointSeedSweep` with `label="bc"` on seeds 1301/1302 — expect TU ≈ 20–24
5. The 15→16 feature migration tool (`drl/migrate_fresh_bc_15_to_16.py`) is available if the expert only produces a 15-feature checkpoint

### Step 3 — Run22 with CQL + floor-guarded accumulation (after Step 2)

Only after valid BC exists:
- Enable CQL: `--cql-alpha 0.1` in `run_p4_online60_training.ps1`'s training call
- Add floor guard: after each block, quick 2-seed validation; revert `$weights` to previous block if TU drops > 2 below previous
- Use `-RunId 22` (runs 1–21 exhausted)

---

## 4. What NOT to Do

- **Do not use `--cql-alpha > 0`** until BC is regenerated. It will lock the policy into local-only behavior.
- **Do not use `runP4OnlineBlockProbes` to evaluate BC.** It routes through `workflow_fresh_ddqn`, not `workflow_fresh_bc`. Use `runP4CheckpointSeedSweep` with `label="bc"` for BC evaluation.
- **Do not resume paused runs 2 or 3.** They used the degenerate BC initialization. Any result from them is against a broken baseline.
- **Do not interpret Run17 B12=23.5 as "near BC."** Actual BC is TU=8. Run17 B12 is +15.5 TU above BC.
- **Do not rerun the LOCAL-ONLY ablation.** It reproduced degenerate BC behavior (TU=8), not a useful experiment.

---

## 5. Key File Locations

| Item | Path (relative to `SafetyOffloading/workflow_v2/`) |
|---|---|
| **HANDOVER document** (full diagnosis + run history) | `docs/HANDOVER_DRL_P4.md` |
| Best DRL policy (Run17 B12) | `outputs/p4_corrected_run17/training/update12/policy.json` |
| Best replay buffer (5792 transitions) | `outputs/p4_corrected_run17/training/update12/replay.npz` |
| BC checkpoint (degenerate — do not use as baseline) | `drl/outputs/fresh_candidate_bc.json` |
| DDQN trainer (CQL via --cql-alpha) | `drl/train_fresh_p4_ddqn.py` |
| 60-rollout training orchestrator | `run_p4_online60_training.ps1` |
| Checkpoint seed sweep evaluator | `runP4CheckpointSeedSweep.m` |
| Block validation function (uses workflow_fresh_ddqn) | `runP4OnlineBlockProbes.m` |
| 15→16 feature migration tool | `drl/migrate_fresh_bc_15_to_16.py` |
| Convergence evaluator | `drl/evaluate_dueling_convergence.py` |
| MATLAB DRL policy inference | `policy/WorkflowPolicyCausalDDQN.m` |
| MATLAB BC policy inference | `policy/WorkflowPolicyFreshDDQN.m` |
| WiLab bridge dispatcher | `wilab/spv2xWorkflowBridge.m` |
| Run17 diagnostics | `outputs/p4_corrected_run17/convergence_diagnostics.json` |

---

## 6. Architecture Reference

```
Input: 16-feature candidate vector
  → encoder: Linear(16→96) → ReLU
  → value head:     Linear(96→64) → ReLU → Linear(64→1)   → V(s)
  → advantage head: Linear(192→64) → ReLU → Linear(64→1)  → A(s,a)
  → Q = V + (A - mean(A))   [dueling combination]

Training:
  - PER replay with 25% guaranteed rare-transition share
  - Batch cap: min(2000, max(512, len(data)//2))
  - Step-λ: flat 0.01 for global_update 0–239, linear decay to 0 for 240–479
  - --bc-weight 0.0 (BC regularisation disabled; BC used only as weight init)
  - --total-updates 480, --global-update-start (block-1)*40
  - --cql-alpha 0.0 (disabled — do not enable until BC is regenerated)
```

---

## 7. Environment and Constraints

- **OS:** Windows 10, PowerShell primary shell
- **MATLAB:** R2026a at `C:\Program Files\MATLAB\R2026a\bin\matlab.exe`
- **Python:** used for training scripts, replay building, convergence evaluation
- **Git remote:** `https://github.com/dash8890/safety-v2x-pipeline-clean` (branch: `master` — never `main`)
- **Commit attribution:** End all commit messages with `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
- **Timezone:** ALL times must be in **UTC only**. Never use PKT or any other timezone.
- **Multi-line git commits in PowerShell:** write message to a `.txt` file, use `git commit -F <path>` — heredoc syntax does not work in PowerShell.
- **pycache:** After `git add SafetyOffloading/`, always run `git reset HEAD "SafetyOffloading/workflow_v2/drl/__pycache__/"` to unstage.

---

## 8. The One-Paragraph Summary

The frozen BC checkpoint is degenerate (TU=8, zero remote stages) — not the TU=24 baseline previously documented. Run17 B12 (TU=24/23) is a genuine DRL result that outperforms actual BC by +15 to +16 TU, achieved by learning to use REQUEST_REMOTE_COMMIT that BC never learned. DRL non-convergence across 21 runs is explained by the BC training data having zero coverage of REQUEST_REMOTE_COMMIT — DRL must discover it through bootstrap noise, which only Run17 did reliably. The correct next steps are: (1) verify Run17 B12 generalises across seeds 1303–1312, (2) regenerate BC from a remote-capable expert, (3) run Run22 with CQL + floor guard using the new BC. CQL is implemented but must stay disabled until valid BC exists. Everything needed is in `docs/HANDOVER_DRL_P4.md` and the files listed in Section 5.
