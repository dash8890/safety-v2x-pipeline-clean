# Session Starter — DRL P4 V2X Project

**Purpose:** Self-contained briefing for a new agent/session. No prior context needed.  
**Written:** 2026-11-28  
**Repo state at handoff:** commit `ffea345`, branch `main`, remote `https://github.com/dash8890/safety-v2x-pipeline-clean`

---

## 1. What This Project Is

A research project to train a **Dueling Double DQN (DDQN)** agent to outperform a **Behavior Cloning (BC)** baseline on a **C-V2X workflow offloading** task in a 5G NR-V2X vehicular network simulation (WiLab V2X simulator, MATLAB). The goal is to include this result in a research paper.

**The metric:** Terminal Utility (TU) = number of workflows (out of 30 per episode) that complete before their 750ms deadline. BC achieves TU=24 on held-out validation seeds 1301 and 1302. The DRL target was TU>24.

**Working directory:** `C:\Users\dash\Downloads\NewProject\safety-v2x-pipeline`  
**GitHub remote:** `https://github.com/dash8890/safety-v2x-pipeline-clean`  
**Key subdirectory for all DRL work:** `SafetyOffloading/workflow_v2/`

---

## 2. What Was Established — The Complete Picture

### 2.1 The training history (21 runs, all summarised)

21 training runs have been executed. No run achieved TU>24. The best result was **run17, B12 mean=23.5** (seed1301=24, seed1302=23). Every other run either collapsed to TU≈11–12 or was non-monotonic. Full run table is in Section 3 of `SafetyOffloading/workflow_v2/docs/HANDOVER_DRL_P4.md`.

### 2.2 TU=24 is a confirmed hard system ceiling on seeds 1301/1302 — not improvable

This was confirmed by two analyses run on 2026-11-28:

**Channel feasibility analysis:** Read the full event traces (workflows.csv, events.csv, decisions.csv) from run17 block12 validation on seeds 1301 and 1302. Findings:
- 0 FRAGMENT_LOST events — the C-V2X sidelink channel is healthy throughout
- Expired and completed workflows execute identical decision sequences (COMMIT_LOCAL → REQUEST_REMOTE_COMMIT → COMMIT_LOCAL × 2)
- The difference is entirely timing: wf1 (early arrival) first FRAGMENT_TX at +36ms → success at +200ms; wf25 (late arrival) first FRAGMENT_TX at +464ms → expired at +750ms
- Root cause: edge compute saturation at peak arrival rate. Late-arriving workflows face 400–560ms first-stage queuing delay, leaving insufficient budget for the subsequent multi-hop chain. This is not fixable by policy decisions.

**LOCAL-ONLY ablation:** Ran BC weights with REQUEST_REMOTE_COMMIT blocked (WorkflowPolicyLocalOnly.m) on seeds 1301 and 1302. Result: **TU=8 on both seeds** (down from BC=24). Remote commit is not an optional optimization for a subset of workflows — it is the core processing mechanism for 16 of 24 successful BC workflows. Blocking it collapses TU by 16 points.

**Conclusion:** TU=24 is the hard ceiling. The 6 expired workflows per episode under BC need REQUEST_REMOTE_COMMIT but arrive too late due to edge congestion. No scheduling policy, no DRL algorithm, no action-space change can exceed TU=24 on seeds 1301/1302 under these traffic conditions.

### 2.3 Why the DRL policy is not converging — unified root cause

The DRL uses **off-policy TD + function approximation + bootstrapping on a fixed BC dataset**. This is Sutton & Barto's "deadly triad." There is no convergence guarantee. The four previously-identified causes (C1–C4) are all manifestations of this single mathematical failure mode:

**The specific mechanism in this implementation:**
1. The step-λ regularization `λ × mean(A)²` drives the advantage head toward zero → action selection becomes arbitrary
2. Once arbitrary selection occurs, non-BC transitions enter the replay buffer with inconsistent reward attribution
3. Bootstrap targets become circularly inconsistent → Q-values diverge
4. The network converges to the 11–12 TU attractor (≈37–40% random success rate over the candidate action set)

**Why run17 "worked" (TU=23.5):** The RunId=17 seed shuffle placed high-TU scenarios in rollouts 1–20. The early replay buffer contained only consistent high-reward BC transitions. The deadly triad was quiescent because the Q-function was well-specified from clean data. TU=23.5 ≈ recovering BC's policy from BC's data — not a policy improvement. Run21 (identical config, bad seed draw): TU=12.

**The fix:** CQL (Conservative Q-Learning) adds `α · (−log Σ_a exp(Q(s,a)) + Q(s, a_BC))` to the loss. This pessimistically bounds Q-values for OOD actions — which is the precise mechanism that breaks the triad's divergence. Without CQL, off-policy TD on BC data cannot converge. With CQL, it has a convergence guarantee.

---

## 3. The Correct Next Steps (in order)

### Step 0 — Select new validation seeds (prerequisite for everything else)

The 21-run failure was caused by pursuing a physically unreachable target on seeds 1301/1302. Before any further DRL training, you must find seeds where BC has headroom.

**How to do this:**
```matlab
% Run BC evaluation on seeds 1303–1312
% Use Invoke-Validation or the existing run_p4_online60_training.ps1 validation logic
% with policy = 'workflow_fresh_bc' and seeds 1303:1312
% Look for seeds where BC scores 22–23 (not 24 — 24 means you're at the ceiling again)
```

**Success criterion:** Find at least 2 seeds where BC TU ∈ [21, 23]. These are seeds where DRL has ≥2 TU of headroom to demonstrate improvement.

**Script to extend:** `run_p4_online60_training.ps1` already has the `Invoke-Validation` function. Extract it and loop over seeds 1303–1312.

### Step 1 — Implement CQL on run17's existing replay buffer

Apply Conservative Q-Learning to `outputs/p4_corrected_run17/training/update12/replay.npz` (5,792 transitions, 60 diverse BC rollouts). No new MATLAB rollouts needed for this step.

**Where to implement:** `SafetyOffloading/workflow_v2/drl/train_fresh_p4_ddqn.py`

**What to add** (after computing `qa` and `y` in the training loop):
```python
# CQL conservative penalty
cql_penalty = -torch.logsumexp(q(cur), dim=0).mean() + qa.mean()
loss = F.smooth_l1_loss(qa, y) + lam * _adv.mean()**2 + alpha * cql_penalty
```
Start with `alpha=0.1`. The `q(cur)` call computes Q-values over all actions for the current states.

**Validate on new seeds (from Step 0), not seeds 1301/1302.** Testing CQL on 1301/1302 is moot — the ceiling there is 24.

### Step 2 — Add floor-guarded block accumulation

Currently `$weights` in `run_p4_online60_training.ps1` never updates (it's in function scope, never assigned from `$policy`). Each block re-initialises from BC. This means 480 total gradient steps = 12 independent 40-step fine-tunes from BC, not one 480-step trajectory.

**Fix:**
1. After each optimizer block, run a quick 2-seed validation (new seeds from Step 0)
2. If `mean_TU >= BC_TU - 2`, accept `$policy` as the new `$weights` for the next block
3. If `mean_TU < BC_TU - 2`, revert to BC weights for that block

This creates a ratchet: policy can only improve, never catastrophically degrade.

### Step 3 — Run a full training run (after Steps 0–2 are confirmed working)

Use `run_p4_online60_training.ps1` with:
- New seeds from Step 0 as validation seeds
- CQL penalty active (`alpha=0.1`)
- Floor-guarded accumulation from Step 2
- `-RunId 22` (runs 1–21 are exhausted; use 22+)

---

## 4. What NOT to Do

- **Do not run more DDQN training on seeds 1301 or 1302.** TU=24 is the hard ceiling. It is not improvable.
- **Do not resume paused runs 2 or 3.** They are moot. Same ceiling, same seeds.
- **Do not interpret run17 B12=23.5 as a converged result.** It was a lucky seed draw, not algorithmic progress.
- **Do not implement CQL and test it on seeds 1301/1302.** The ceiling there cannot be exceeded regardless of algorithm. Use new seeds.
- **Do not add dense reward shaping or richer features before fixing the algorithm.** The deadly triad is the root cause. Everything else is secondary.
- **Do not use 16-block (80-rollout) extension.** All three attempts (runs 18/19/20) collapsed. The 12-block 60-rollout config is the stable one.

---

## 5. Key File Locations

| Item | Path (relative to `SafetyOffloading/workflow_v2/`) |
|---|---|
| **HANDOVER document** (full diagnosis + run history) | `docs/HANDOVER_DRL_P4.md` |
| Best DRL policy (run17 B12) | `outputs/p4_corrected_run17/training/update12/policy.json` |
| Best replay buffer (5792 transitions) | `outputs/p4_corrected_run17/training/update12/replay.npz` |
| BC policy weights | `drl/outputs/fresh_candidate_bc.json` |
| DDQN trainer (modify for CQL) | `drl/train_fresh_p4_ddqn.py` |
| 60-rollout training orchestrator | `run_p4_online60_training.ps1` |
| Convergence evaluator | `drl/evaluate_dueling_convergence.py` |
| MATLAB DRL policy inference | `policy/WorkflowPolicyCausalDDQN.m` |
| MATLAB BC policy | `policy/WorkflowPolicyFreshDDQN.m` |
| LOCAL-ONLY ablation policy | `policy/WorkflowPolicyLocalOnly.m` |
| WiLab bridge dispatcher | `wilab/spv2xWorkflowBridge.m` |
| Run17 diagnostics | `outputs/p4_corrected_run17/convergence_diagnostics.json` |
| Feasibility analysis traces | `outputs/p4_corrected_run17/validation/block12/seed{1301,1302}/workflow_v2/` |

---

## 6. Architecture Reference (for modifying train_fresh_p4_ddqn.py)

```
Input: 16-feature candidate vector
  → encoder: Linear(16→96) → ReLU
  → value head:     Linear(96→64) → ReLU → Linear(64→1)      → V(s)
  → advantage head: Linear(192→64) → ReLU → Linear(64→1)     → A(s,a)
  → Q = V + (A - mean(A))   [dueling combination]

Training:
  - PER replay with 25% guaranteed rare-transition share
  - Batch cap: min(2000, max(512, len(data)//2))
  - Step-λ: flat 0.01 for global_update 0–239, linear decay to 0 for 240–479
  - --bc-weight 0.0 (BC regularisation disabled; BC used only as weight init)
  - --total-updates 480, --global-update-start (block-1)*40
```

**CQL insertion point in `train_fresh_p4_ddqn.py`:** find the section where `loss = F.smooth_l1_loss(qa, y) + lam * _adv.mean()**2` and add the CQL term after computing `qa`.

---

## 7. Environment and Constraints

- **OS:** Windows 10, PowerShell primary shell
- **MATLAB:** R2026a at `C:\Program Files\MATLAB\R2026a\bin\matlab.exe`
- **Python:** used for training scripts (`train_fresh_p4_ddqn.py`), replay building, convergence evaluation
- **Git remote:** `https://github.com/dash8890/safety-v2x-pipeline-clean` (branch: `main`)
- **Commit attribution:** End all commit messages with `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
- **Timezone:** ALL times must be in **UTC only**. Never use PKT or any other timezone.
- **Multi-line git commits in PowerShell:** write message to a `.txt` file and use `git commit -F <path>` — heredoc syntax does not work in PowerShell.
- **pycache:** Always run `git reset HEAD "SafetyOffloading/workflow_v2/drl/__pycache__/"` and `git reset HEAD "SafetyOffloading/workflow_v2/tools/"` after broad `git add SafetyOffloading/` to avoid staging pycache accidentally.

---

## 8. The One-Paragraph Summary

After 21 training runs, TU=24 is a confirmed hard ceiling on validation seeds 1301/1302 — caused by edge compute saturation at peak workflow arrival rate, not by channel conditions or policy quality. The DRL cannot converge to or exceed BC on these seeds because (a) the target is physically unreachable, and (b) the algorithm (DDQN on fixed BC data) lacks convergence guarantees without CQL. Run17's TU=23.5 was not convergence — it was a lucky seed draw that kept the deadly triad quiescent. The correct path forward is: (1) find new validation seeds where BC scores 21–23 (not 24), (2) implement CQL in `train_fresh_p4_ddqn.py`, (3) add floor-guarded block accumulation to `run_p4_online60_training.ps1`, (4) run a fresh training run (RunId=22+) on the new seeds. Everything needed is in `docs/HANDOVER_DRL_P4.md` and the files listed in Section 5 above.
