# R4-DRL implementation notes

What was built, how it differs from the R1–R3-M line, and why the new approach produced a positive result.

Related: results [`R4_DRL_EXIT.md`](R4_DRL_EXIT.md), freeze/evidence [`R4_DRL_FREEZE_ADDENDUM.md`](R4_DRL_FREEZE_ADDENDUM.md), Mode-2 gate [`R3M_EXIT.md`](R3M_EXIT.md).

**Freeze:** code, checkpoint, configs, and results are frozen. No further MDP redesign from this document.

---

## 1. What was implemented

### 1.1 Package layout

| Path | Role |
| --- | --- |
| `drl/workflow_mdp.py` | Discrete-event abstract diamond-4 MDP + baseline policies |
| `drl/ddqn_agent.py` | Double DQN (Q-network, target net, replay, masked actions) |
| `drl/run_drl_experiment.py` | Train → eval vs B0/B2/B3/B4 → holdout → JSON/CSV |
| `drl/outputs/ddqn_best.pt` | Best checkpoint by eval utility |
| `drl/outputs/drl_result.json` | Frozen metrics and win flag |

Stack: Python, NumPy, PyTorch (CUDA used for training).

### 1.2 Environment (MDP)

**Workflow object (unchanged topology):** diamond-4 DAG — stages A→{B,C}→D with the same cycle weights as `workflow_v2` (9 / 12 / 12 / 3 Mcycles).

**Vehicles:** origin (slow local CPU) + helper-2 (fast, volatile) + helper-3 (slower, always reachable).

**Decision epochs:** whenever at least one stage is `ELIGIBLE`. The env presents one stage at a time (soft-EDF with defer counts so `HOLD` can skip).

**Observation (16-D):** normalized CPU queues, helper contact flags, slack, stage type, cycles, active load, time, time-to-exit/return, remote/success fractions, **workflow value**, eligible count.  
**Note:** `time-to-exit/return` are computed from the **scripted future schedule** (privileged). See [`R4_DRL_FREEZE_ADDENDUM.md`](R4_DRL_FREEZE_ADDENDUM.md) §2. Not onboard mobility estimates.

**Actions (5, legally masked):**

| Action | Meaning |
| --- | --- |
| `LOCAL` | Commit stage to origin CPU |
| `H2` / `H3` | Offload to helper (blocked if no contact or helper banned after strand) |
| `HOLD` | Deprioritize this eligible stage so another can be chosen |
| `DROP` | Abort the whole workflow (admission reject) |

A/D are placement-forced to local (DAG anchors) but may still be `DROP`’d.

**Dynamics extras (vs pure queueing):**

- Abstract uplink delay + jitter before remote enqueue (not WiLab Mode-2 SPS).
- Scripted helper-2 outage; on exit, in-flight/queued H2 work is **stranded**, helper banned for that stage, re-eligibility delayed until return.
- Heterogeneous workflow **values**: high=5 with p=0.30, else 1.
- Reward: `+value` on deadline-valid completion; small penalties for illegal/drop; expire penalty.

**Primary metric:** `total_value` (sum of values of deadline-valid workflows). Secondary: `n_success`.

### 1.3 Learning algorithm

**Double DQN** with:

- MLP Q-network (obs → 128 → 128 → 5 actions)
- Experience replay + target network sync
- ε-greedy exploration decaying to 0.05
- **Action masking** at act and bootstrap (illegal Q set to −∞)

Literature alignment: deadline-aware DQN / DDQN for vehicular offloading and prioritized completion (e.g. Farimani ESWA 2024; dependency-aware DDQN; prioritized DQN at VTC).

### 1.4 Baselines in the same MDP

| Policy | Behavior |
| --- | --- |
| B0 | Always local |
| B2 | Shortest expected finish among reachable CPUs |
| B3 | B2 + avoid H2 if job would overrun contact exit |
| B4 | Joint heuristic (prefer distinct helpers for B/C) + B3 |
| ValueGreedy | Drop low-value when overloaded; else B3 |

Original train-time win flag compared DDQN to **B0–B4 only**. For research claim scope, **ValueGreedy is a principal comparator**; matched holdout numbers are in the freeze addendum.

### 1.5 Training / eval protocol

- 900 training episodes (seeds 1001…)
- Periodic eval on seeds 1–15; checkpoint best by mean value
- Final report: seeds 1–40; **holdout** seeds 300–339 (never used in training eval cadence)

---

## 2. How this differed from the previous implementation

### 2.1 Previous line (R1 → R3-M)

| Aspect | R1–R3-M |
| --- | --- |
| Runtime | MATLAB `workflow_v2` + WiLab NR-V2X Mode-2 |
| Radio | Real SPS / TB / sensing path (`wilab_mode2`) |
| Policies | Deterministic B0–B5 planners (`WorkflowPolicy*.m`) |
| Objective | Count of deadline-valid workflows (`n_success`) |
| Values | Homogeneous (every workflow worth 1) |
| Admission | No explicit reject; all arrivals admitted |
| Learning | None (R3-M exit: stop, no DRL on that gate) |
| R3-M outcome | **B0 ≫ B4** — local beat joint offload under Mode-2 tax |

R3-M deliberately stressed contention + scripted contact churn. Mechanisms were identifiable, but **remote execution was dominated by Mode-2 cost**, so joint placement could not beat local EDF.

### 2.2 What stayed the same

- Diamond-4 stage/edge structure and cycle sizes  
- Origin + two helpers  
- Deadline-valid completion as the success definition  
- Scripted helper-2 exit/return as the mobility stress  
- Same baseline *names* (B0/B2/B3/B4) for continuity of narrative  

### 2.3 What changed (authorized redesign)

| Axis | Previous | DRL implementation |
| --- | --- | --- |
| Radio model | WiLab Mode-2 | Abstract fixed delay + jitter |
| Capacity regime | Remote often worse than local | Local insufficient; helpers scarce but useful |
| Mobility effect | Contact gating | Contact gating **+ stranding + ban** |
| Objective | Unweighted completions | **Prioritized utility** (value 5 vs 1) |
| Action set | Commit / request / release edges | Place / hold / **drop** |
| Agent | Hand rules | Double DQN |
| Speed | ~minutes per WiLab seed | ~ms per episode → full train in ~25 min |

R3-M is **not invalidated**. It remains the Mode-2 fidelity gate. R4-DRL answers a different question: *given an abstract VEC model where offload and admission can help, can DRL beat value-blind heuristics?*

---

## 3. Why the new approach worked

### 3.1 The failure mode that R3-M exposed

Under Mode-2, the “joint scheduling” signal was swamped by radio delay. Completing stages locally avoided that tax, so B0 won. Pushing DRL onto that same WiLab regime would have optimized a landscape where the optimum is approximately “never offload.”

### 3.2 Three ingredients that create a learnable advantage

1. **Abstract radio with real helper speedup**  
   Remote compute is faster than origin by enough to offset a modest RTT. Offloading is *sometimes* correct — unlike R3-M.

2. **Scarce capacity**  
   ~20 overlapping workflows and a weak stable helper mean only a subset can finish. *Which* subset finishes matters.

3. **Heterogeneous value + DROP**  
   High-value workflows (safety-critical class) are worth 5× low-value ones. Value-blind B2/B3/B4 fill queues with low-value work and starve high-value completions.  
   Allowing `DROP` turns the problem into **admission + placement**, which is exactly where deadline/priority DRL is known to help in VEC literature.

Contact stranding reinforces contact-aware placement (don’t park work on H2 that will leave), but the large utility gap vs B4 is driven mainly by **learning to reject low-value load** while still using helpers aggressively for what remains — DDQN also raised raw completion count above the hand-coded ValueGreedy reference.

### 3.3 Why DDQN specifically was enough

- Discrete, small action set with hard constraints → masking removes most illegal exploration.  
- Reward is sparse but aligned with the reported metric (`+value` on success).  
- Double DQN is stable enough for this obs/action scale without needing PPO’s continuous control machinery.

### 3.4 Matched holdout vs ValueGreedy (frozen; no retrain)

| | ValueGreedy | DDQN frozen |
| --- | ---: | ---: |
| mean utility (seeds 300–339) | 27.63 | **30.78** (+11.4%) |
| mean n_success | 7.03 | **13.18** (+87.5%) |

Utility edge over the priority-aware rule is **moderate**; completion edge is **large**. This supports “useful admission+placement,” not a headline utility blowout vs strong priority-aware optimization.

### 3.5 One-line causal summary

**R3-M failed for DRL because Mode-2 made offloading irrational. R4 worked because the abstract MDP made admission and placement control scarce prioritized utility — and DDQN beat value-blind schedulers there (with privileged schedule features in the obs).**

---

## 4. How to read the two results together

| Claim | Supported by |
| --- | --- |
| Under NR-V2X Mode-2 WiLab as configured in R3-M, local can beat joint offload | `R3M_EXIT.md` |
| Under abstract prioritized DAG offloading with scarcity + churn, DDQN beats B0–B4 by large utility margin | `R4_DRL_EXIT.md` + this doc |

Closing Mode-2 radio into this DRL loop remains future work; it is a different engineering gate, not assumed by the R4 claim.

## 5. Reproduce

```bash
cd SafetyOffloading/workflow_v2/drl
python run_drl_experiment.py
```

Outputs overwrite `drl/outputs/drl_result.json`, `drl_summary.csv`, and refresh `ddqn_best.pt` when a new best appears.
