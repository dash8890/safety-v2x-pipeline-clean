# Causal Dueling DDQN — self-contained report

This document stands alone. It explains what was built, how it differs from earlier stages of the same project, what metrics were obtained, and why the method worked—without requiring other memos to be readable.

**Result in one line:** under causal (non-privileged) observations, a Dueling Double DQN initialized by behavioral cloning from a **Priority-Based Demonstration Policy (PDP)** and then RL-fine-tuned achieved **+29.8% mean deadline-valid utility** vs ValueGreedy on held-out seeds 300–339 (Python abstract radio). WiLab Mode-2 transport is a separate claim (`runWilabCausalEval`); abstract `mode2_event` is not that claim.

**PDP role (locked):** PDP = Always-Drop-Low is the **BC teacher only**. It is **not** an optimization oracle, **not** an optimality upper bound, and **not** a principal literature baseline. Principal comparison = ValueGreedy. Learning-contribution analysis = PDP vs BC-only vs Full Causal Dueling DDQN.

---

## 0. Project context (minimal background)

The broader project studies **safety-oriented workflow offloading** among vehicles: a requesting vehicle (origin) may run DAG compute stages locally or on neighbor “helpers,” subject to deadlines, CPU queues, and intermittent wireless contact.

**Diamond-4 workflow.** Each workflow is a small DAG:

```text
        B
       / \
Input → A   D → Result
       \ /
        C
```

Stage CPU work is fixed at 9 / 12 / 12 / 3 Mcycles for A / B / C / D. Parallel stages B and C can run on different machines. A workflow succeeds only if D finishes before the absolute deadline; **utility** is the sum of per-workflow values of successful workflows (high-priority workflows are worth more than low-priority ones).

**Earlier stages of this codebase (summary only):**

| Stage | What it was | Outcome relevant here |
| --- | --- | --- |
| R1–R2 | MATLAB workflow engine + Mode-2 adapters | Implementation validated |
| R3 / R3-M | WiLab NR-V2X Mode-2 pilots with hand-coded planners B0–B4 | Under Mode-2 radio cost, **local-only often beat remote offload**; that Mode-2-negative finding remains separate from this Python MDP result |
| R4 (first DRL) | Python MDP + DDQN | Showed learning can help, but the first strong numbers used **privileged** contact-timing features (exact time until a scripted helper exit/return)—not deployable as onboard estimates |
| Correction attempts | Remove privilege; optional event-driven radio | Without a clear admission gap and without cloning early DROP, utility vs ValueGreedy was roughly **flat** |

This document is about the **next** learning stack: causal observations + explicit exploitation of an admission-control gap.

**Glossary**

| Term | Meaning |
| --- | --- |
| Causal observation | Features computable from events the ego vehicle has already received by time *t* (e.g. current contact bit, age since last contact change, age of last helper advertisement). **Not** exact future schedule times |
| Privileged observation | Features that reveal the simulator’s private future contact schedule (e.g. seconds until a scripted helper exit) |
| ValueGreedy | Principal hand baseline: drop a low-value workflow only if at least 4 workflows are already active; otherwise mobility-aware placement |
| PDP (Priority-Based Demonstration Policy) | Always-Drop-Low: if low priority and DROP legal, reject; else mobility-aware placement. **BC teacher only** — handcrafted demonstration policy, not an oracle |
| BC-only | Neural policy after imitating PDP, before RL fine-tuning |
| Full Causal Dueling DDQN | BC-initialized then RL-fine-tuned policy (proposed method) |
| BC | Behavioral cloning from PDP demonstration trajectories |
| Dueling DQN | Q-network that splits state value and action advantages (Wang et al.), combined with Double DQN targets |
| Holdout | Seeds 300–339 never used as the training episode index stream for the final reported table |

---

## 1. What was implemented

### 1.1 Code and artifacts

| Path | Role |
| --- | --- |
| `drl/workflow_mdp.py` | Discrete-event diamond-4 simulator: CPU queues, contact, actions, rewards, causal/privileged obs modes |
| `drl/mode2_radio.py` | Optional event-driven radio (COMMIT→ACK→input fragments→CPU→output fragments). **Not** used for the headline metrics below (those use abstract delay) |
| `drl/ddqn_agent.py` | Replay buffer, Double DQN updates, action masking; optional Dueling head |
| `drl/train_causal_dueling_ddqn.py` | End-to-end: evaluate ValueGreedy → BC from PDP → DROP-biased RL fine-tune → holdout JSON |
| `drl/outputs/ddqn_causal_dueling.pt` | Final checkpoint |
| `drl/outputs/causal_dueling_holdout.json` | Full numeric dump of the holdout evaluation |
| `drl/outputs/ddqn_best.pt` | **Older** R4 checkpoint; intentionally **not** overwritten |

Stack: Python 3, NumPy, PyTorch (CUDA used for training).

### 1.2 Decision problem

**State / decision epochs.** Whenever at least one stage is eligible, the environment presents one stage (soft earliest-deadline order, with a defer counter so HOLD can skip). The agent must choose an action for that stage.

**Actions (five, legally masked):**

| Action | Meaning |
| --- | --- |
| `LOCAL` | Run the stage on the origin CPU |
| `H2` / `H3` | Offload to helper 2 or 3 (only if currently in contact and not banned after a strand) |
| `HOLD` | Deprioritize this eligible stage so another may be presented |
| `DROP` | Abort the entire workflow (admission reject) |

Stages A and D are placement-forced to local but may still be dropped.

**Vehicles.** Origin CPU is slow; helper 2 is fast but leaves contact for a scripted interval (work on it can be stranded); helper 3 is slower but always reachable. Contact flips are simulated from a private schedule; **causal mode does not expose that schedule in the observation.**

**Observation (16 dimensions, causal mode used here).** Normalized CPU queues; whether H2/H3 are currently reachable; deadline slack; stage type; normalized cycles; active-load fraction; normalized time; **contact age** (0 if down); **advertisement age** (how stale the last in-contact helper advertisement is); remote/success fractions; workflow value; eligible-count feature.  
Contrast: privileged mode would replace the two age features with exact time-to-exit / time-to-return from the script—**not used for this result.**

**Configured workload for this run:**

| Setting | Value |
| --- | --- |
| Radio | Abstract delay + jitter (not full WiLab PHY) |
| Workflow values | Low = 1; high = 12 |
| Phased highs | Before t = 0.16 s: P(high) = 0.05; after: P(high) = 0.75 |
| Deadline | 0.35 s relative |
| Capacity | Scarce: weak stable helper, background load on origin |
| Reward (`utility_focus`) | Extra credit for high-value success; penalty if a low-value workflow completes while highs are still active; **positive bonus for DROPping low-value in the early phase or while highs are pending**; large penalty for DROPping a high-value workflow |

**Primary metric:** mean **total_value** (sum of values of deadline-valid workflows) on holdout seeds. Secondary: mean completion count.

### 1.3 Comparators and PDP placement

| Policy | Role in reporting |
| --- | --- |
| ValueGreedy | **Principal baseline** (priority-aware hand rule) |
| B4 joint | Secondary hand heuristic |
| PDP (Always-Drop-Low) | **Method teacher** + teacher/BC/RL analysis — **not** a literature baseline |
| BC-only | Ablation: imitation of PDP without RL |
| Full Causal Dueling DDQN | Proposed method |

PDP rule: if current workflow is low priority and `DROP` is legal, reject; else mobility-aware placement. It does not solve an optimal program and provides **no optimality guarantee**.

### 1.4 Learning procedure

\\[ \\pi_{\\mathrm{PDP\\ teacher}} \\longrightarrow \\text{Behavioral cloning} \\longrightarrow \\text{Dueling DDQN fine-tuning}. \\]

| Step | Detail |
| --- | --- |
| 1. PDP rollouts | Collect (observation, action, legal mask) from the demonstration policy |
| 2. Behavioral cloning | Masked cross-entropy so argmax matches PDP actions → **BC-only** checkpoint |
| 3. Biased exploration | During ε-greedy explore, if low-value and DROP legal, choose DROP with probability 0.70 |
| 4. RL fine-tune | Double DQN with utility_focus rewards → **Full Causal Dueling DDQN** |
| 5. Selection | Checkpoint by periodic eval utility; final report on seeds **300–339** |

---

## 2. How this differed from previous implementations

### 2.1 Versus WiLab Mode-2 planners (R3 / R3-M)

Those experiments ran inside MATLAB/WiLab with hand-coded policies and counted deadline successes under **real Mode-2 transport cost**. There, local execution often dominated remote offload; that finding is about Mode-2 radio economics and is **not contradicted** by this Python abstract-radio result. This experiment asks a different question: given scarce compute and prioritized workflows, can a learned **admission + placement** policy beat a strong hand priority rule when observations are causal?

### 2.2 Versus the first DRL result (privileged R4)

The first DDQN result used observation channels equal to “time until scripted helper exit/return.” Those channels are exact future information. Baselines that also read the schedule can be compared fairly *inside* that toy, but the setup does **not** establish deployability. The present stack replaces those channels with contact/advertisement ages, keeps a separate checkpoint (`ddqn_causal_dueling.pt`), and leaves `ddqn_best.pt` untouched as the privileged-era reference.

### 2.3 Versus an intermediate causal run that tied ValueGreedy

Simply turning on causal observations on a workload where ValueGreedy’s `n_active ≥ 4` rule was already near-optimal produced **~0% utility difference**. That run also relied on generic ε-greedy discovery of DROP. The present run **changes the workload** so early lows conflict with a late high burst (PDP−VG gap ≈ +21 utility) and **changes the training method** so early DROP is cloned from PDP and then RL-refined—not merely hoped for.

| Aspect | Privileged R4 DDQN | Causal-but-flat attempt | This stack |
| --- | --- | --- | --- |
| Contact features | Future schedule times | Causal ages | Causal ages |
| High-value process | i.i.d. ~30% | Similar / weak gap | Phased 5% → 75% |
| How DROP enters policy | Exploration | Exploration + light shaping | **BC + early-DROP reward + explore bias** |
| Headline comparator | Often value-blind B4 | ValueGreedy | ValueGreedy |
| Holdout utility vs ValueGreedy | N/A (different framing) | ≈ 0% | **+29.8%** |

---

## 3. What was achieved (metrics)

**Protocol.** For each seed in 300–339, every policy sees the same arrival times, sampled values, and contact realization. Observations are causal. Training episode seeds are separate from this holdout list.

**Principal comparison (literature-style baseline):**

| Policy | Mean utility | Mean completions |
| --- | ---: | ---: |
| B4 joint | 3.68 | 1.48 |
| ValueGreedy | 78.40 | 7.45 |
| **Full Causal Dueling DDQN** | **101.80** | **8.58** |

| Metric vs ValueGreedy | Value |
| --- | ---: |
| Absolute utility gain | **+23.40** |
| Relative utility gain | **+29.8%** |
| Completion gain | **+1.13** mean workflows |
| Paired utility wins / ties / losses | **28 / 0 / 12** |

**Learning-contribution analysis (teacher vs learning — not a literature bake-off):**

| Policy | Mean utility | Role |
| --- | ---: | --- |
| PDP (teacher) | 99.30 | What human-designed admission already supplies |
| BC-only (after imitating PDP) | ~99 (probe) | What imitation alone installs |
| Full Causal Dueling DDQN | **101.80** | ≈ **+2.5%** over PDP — RL refinement beyond demonstrations |

Question answered here: *does RL improve scheduling beyond the behavior explicitly demonstrated during training?* The BC-only checkpoint is required to separate imitation from RL.

Raw numbers: `drl/outputs/causal_dueling_holdout.json`.

---

## 4. Why the new approach worked

### 4.1 A measurable structural gap

Under phased arrivals, most early workflows are low value and most late ones are high value. Capacity is insufficient to finish everything. ValueGreedy still **runs early lows** until four workflows are active; by then, scarce queues are polluted and late highs miss deadlines.

PDP (always dropping lows) frees that capacity and scores **99.30** vs ValueGreedy’s **78.40** (**+20.9**) on the same holdout — establishing what the **demonstration teacher** already captures before RL.

### 4.2 Training attacked that gap directly

1. **Phased value process** created the conflict between early lows and late highs without leaking future *contact* times into the observation.  
2. **Behavioral cloning from PDP** installed admission behavior so the agent did not depend on accidentally sampling DROP early.  
3. **Early-phase DROP reward** reinforced rejecting lows *before* highs appear—the timing PDP demonstrates.  
4. **Exploration bias** continued to sample DROP on low-value decisions during RL so fine-tuning would not erase admission.  
5. **Dueling Double DQN** refined action ranking after admission was cloned.  
6. **Causal observations** ensured the reported gain is not explained by reading the private mobility schedule.

### 4.3 What failure modes this avoids

| Failure mode | How this stack avoids it |
| --- | --- |
| No positive offload regime (Mode-2 tax dominates) | Different question: abstract radio + admission; Mode-2 left as a separate claim |
| Inflated win from future contact features | Causal obs only for headline metrics |
| “Train DDQN and hope” on a flat MDP | Workload chosen so PDP ≫ ValueGreedy; method clones PDP then RL-fine-tunes |
| Utility-flat “win” that only raises low-value completions | Primary metric is value-sum; reward punishes crowding out highs |

### 4.4 Summary

The method worked because **(a)** the environment made early admission control worth roughly twenty utility points against ValueGreedy, and **(b)** training **explicitly cloned PDP then RL-fine-tuned**, under observations that do not reveal future contact transitions. Most of the VG gap is already in the teacher (~99.3 vs 78.4); Full DDQN’s scientific job is to show whether anything beyond that demonstration is learned (~101.8).

---

## 5. Claim boundaries

| Claim | Status |
| --- | --- |
| On this causal, abstract-radio, phased-value MDP, DDQN+BC beats ValueGreedy by ~30% mean utility on holdout 300–339 | **Supported** |
| Observations include exact future helper exit/return times | **False for this result** |
| Result is a measured WiLab NR-V2X Mode-2 PHY gain | **Not claimed** |
| Pure RL discovered DROP with no demonstrations | **Not claimed** (BC from PDP was deliberate) |
| PDP is an optimization oracle / upper bound | **False** |
| PDP is a principal literature baseline | **False** (teacher + learning-contribution analysis only) |
| `ddqn_best.pt` was replaced | **False** (separate checkpoint) |

---

## 6. Reproduce

```bash
cd SafetyOffloading/workflow_v2/drl
python train_causal_dueling_ddqn.py
```

Expected outputs: `outputs/ddqn_causal_dueling.pt`, `outputs/causal_dueling_holdout.json`.

---

## Appendix — optional further files

These are **not** required to understand this report; they archive related experiments:

- `docs/R4_DRL_IMPLEMENTATION.md` — first privileged-observation DRL setup  
- `docs/R4_CORRECTION.md` — causal-obs / Mode-2-event correction matrix  
- `docs/R4_DRL_FREEZE_ADDENDUM.md` — matched holdout audit of the privileged R4 era  
- `docs/DRL_ADVANCED_Q_EXPLORATION.md` — notes on Rainbow-style Q components for future work  
- `docs/R3M_EXIT.md` — Mode-2 gate where local beat remote  
