# R4-DRL freeze addendum — comparison & observation validity

**Status:** R4 abstract-MDP experiment **ACCEPTED and FROZEN** (final classification).  
**No further action authorized:** no retraining, MDP redesign, architecture search, Mode-2 DRL integration, or further algorithm development.

### Final research classification

| | |
| --- | --- |
| **Demonstrated** | DDQN improves prioritized workflow admission and placement over the tested priority-aware heuristic (ValueGreedy) in an abstract, capacity-constrained vehicular computing environment **with known future contact transitions**. Matched holdout: +11.4% utility, +87.5% completions. |
| **Not demonstrated** | A novel, deployable NR-V2X Mode-2 algorithm that outperforms strong priority-aware optimization. |

| Research component | Status |
| --- | --- |
| R1–R2 implementation validation | Closed |
| R3 and R3-M Mode-2 feasibility | Closed |
| R4 abstract-MDP learning | **Accepted and frozen** |
| Matched holdout / observation audit | Complete |
| Mode-2 DRL integration | **Not authorized** |
| Further algorithm development | **Not authorized** |

Frozen artifacts:

| Item | Path |
| --- | --- |
| MDP + baselines | `drl/workflow_mdp.py` (defaults unchanged) |
| DDQN code | `drl/ddqn_agent.py` |
| Train script (historical) | `drl/run_drl_experiment.py` |
| Checkpoint | `drl/outputs/ddqn_best.pt` |
| Original train report | `drl/outputs/drl_result.json` |
| **Matched holdout** | `drl/outputs/r4_matched_holdout.json` |
| Matched eval runner | `drl/eval_matched_holdout.py` |

---

## Decision (accepted)

R4 is a **successful abstract-MDP learning experiment**. It is the first result in this sequence showing a substantial benefit from a learned scheduling policy versus **value-blind** schedulers (B0–B4).

It is **not** an NR-V2X Mode-2 result. R3-M and R4 remain separate claims.

---

## Evidence requirement 1 — matched holdout vs ValueGreedy

Frozen DDQN and ValueGreedy (plus B0–B4) evaluated on **identical** seeds `300–339`.

| Policy | Mean utility | Mean completions |
| --- | ---: | ---: |
| B0 local | 2.20 | 1.00 |
| B2 queue | 9.08 | 4.38 |
| B3 mobility | 9.30 | 4.40 |
| B4 joint | 9.10 | 4.30 |
| **ValueGreedy** | **27.63** | **7.03** |
| **DDQN (frozen)** | **30.78** | **13.18** |

**DDQN − ValueGreedy (matched):**

| Metric | Absolute | Relative |
| --- | ---: | ---: |
| Utility | +3.15 | **+11.4%** |
| Completions | +6.15 | **+87.5%** |

Paired over 40 seeds:

- Utility: **27 wins / 3 ties / 10 losses**
- Completions: **40 / 0 / 0**

### Interpretation (narrowed claim)

| Finding | Assessment |
| --- | --- |
| DDQN ≫ value-blind B0–B4 on holdout | **Supported** (utility ~30.8 vs ~9.1–9.3) |
| DDQN improves completions vs ValueGreedy | **Supported** (+87.5%, all seeds) |
| DDQN substantially improves **utility** vs ValueGreedy | **Weakly supported** (+11.4% mean; not a +203%-class gap) |
| Original `substantial_win=true` vs B4 | Valid only as **vs value-blind**; overstated if read as vs priority-aware |

The strongest DRL-specific signal on the matched set is the **completion lift at comparable-or-better utility**, consistent with learning a useful admission+placement mix rather than maximizing utility by rejecting almost everything.

Reproduce (no train):

```bash
cd SafetyOffloading/workflow_v2/drl
python eval_matched_holdout.py
```

---

## Evidence requirement 2 — observation validity (`time-to-exit/return`)

### What the code does

In `workflow_mdp.py` `_obs()`:

```text
to_exit = max(0, contact_exit - t) / horizon
to_ret  = max(0, contact_return - t) / horizon
```

`contact_exit` / `contact_return` are **fixed scripted schedule constants** in the MDP config (0.10 s / 0.62 s), not estimates from simulated sensors, GNSS traces, or neighbor discovery.

### Classification

| Question | Answer |
| --- | --- |
| Are these deployable local mobility estimates? | **No** (not as implemented) |
| Are they exact future contact transitions? | **Yes — privileged schedule features** |
| Does a real vehicle observe them directly? | **No**, unless an external oracle/schedule is assumed |

**Research classification of the frozen R4 result:**  
contact-timing channels in the observation are **privileged future information**. Contact-aware placement credit must be read under that assumption.

### Fairness note vs heuristics

`policy_b3` / ValueGreedy also read the same scripted `contact_exit` / `contact_return` when deciding whether H2 is safe. So the **matched DDQN vs ValueGreedy** comparison shares privileged schedule knowledge; it does not isolate “learned timing from onboard estimates.” It does isolate learned admission/placement given that shared schedule.

A deployable variant would replace these two features with estimates from locally available state only (e.g. current contact bit, recent contact duration, relative kinematics). That is **not** part of the frozen R4 claim and is not authorized here.

---

## Claim boundary (preserve)

| Claim | Status |
| --- | --- |
| Learned policy beats value-blind schedulers in this abstract capacity/churn/priority MDP | **Accepted** |
| Learned policy is a large utility win over strong priority-aware rules | **Not the headline; matched gap is +11.4% utility** |
| Result transfers to WiLab NR-V2X Mode-2 | **Not established** (R3-M boundary stands) |

Mode-2 integration remains a separate authorization.

---

## Bottom line

**Most important matched result:** DDQN improves **completions by +87.5%** against the priority-aware comparator ValueGreedy while also raising mean utility (+11.4%). That is more informative than the original +203% framing vs value-blind B4.

**Demonstrated:** prioritized admission/placement learning in an abstract capacity-constrained env **with known future contact transitions**.

**Not demonstrated:** deployable NR-V2X Mode-2 superiority over strong priority-aware optimization.

Preserve checkpoint + matched eval + privileged-obs classification together. Sequence end-state: `SEQUENCE_STATUS.md`.
