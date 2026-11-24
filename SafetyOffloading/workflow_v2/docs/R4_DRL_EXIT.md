# R4-DRL — definitive result (CLOSED)

**Status:** ACCEPTED and **FROZEN** as an abstract-MDP learning experiment.  
**Clock:** trained and evaluated before midnight local.  
**Code:** `workflow_v2/drl/`  
**Artifacts:** `workflow_v2/drl/outputs/drl_result.json`, `ddqn_best.pt`, `drl_summary.csv`  
**Implementation write-up:** [`R4_DRL_IMPLEMENTATION.md`](R4_DRL_IMPLEMENTATION.md)  
**Freeze / matched holdout / observation validity:** [`R4_DRL_FREEZE_ADDENDUM.md`](R4_DRL_FREEZE_ADDENDUM.md)

## What changed vs R3-M (required redesign)

R3-M (Mode-2 WiLab) showed **B0 ≫ remote** — radio tax erased joint-offload value. That gate remains valid for Mode-2 SPS cost.

For DRL we **changed the problem** (authorized) to the literature-standard VEC setting where the established *decision* still matters:

| Axis | Choice |
| --- | --- |
| System model | Abstract V2V delay + diamond-4 DAG (same workflow object) |
| Scarcity | Overloaded helpers, local CPU insufficient |
| Mobility | Helper-2 scripted exit/return with **stranding** |
| Objective | **Prioritized** deadline utility: high-value=5 (p=0.30), low=1 |
| MDP actions | `{LOCAL, H2, H3, HOLD, DROP}` with legal masks |
| Algorithm | **Double DQN** + action masking (Farimani/Pang/VTC prioritized DQN line) |

Primary metric: **total_value** of deadline-valid workflows (safety-priority utility). Secondary: completion count.

## Results (40 seeds; holdout seeds 300–339 never used in training eval)

| Policy | mean value | mean n_success |
| --- | ---: | ---: |
| B0 local | 2.10 | 1.00 |
| B2 queue | 9.55 | 4.35 |
| B3 mobility | 10.03 | 4.53 |
| B4 joint heuristic | **10.15** | 4.65 |
| ValueGreedy reference | 30.20 | 7.30 |
| **DDQN (eval 1–40)** | **31.40** | **13.00** |
| **DDQN holdout 300–339** | **30.78** | **13.18** |

**Holdout vs best B0–B4:** large utility gain (original report: +203% vs B4 on mismatched ValueGreedy framing — see addendum).  
Original `substantial_win=true` means **substantial vs value-blind B0–B4 only**.

### Corrected comparator (matched holdout 300–339)

See [`R4_DRL_FREEZE_ADDENDUM.md`](R4_DRL_FREEZE_ADDENDUM.md): DDQN utility **30.78** vs ValueGreedy **27.63** (**+11.4%**); completions **13.18** vs **7.03** (**+87.5%**).

## Interpretation (final)

1. **Headline matched finding:** vs priority-aware ValueGreedy, DDQN delivers **+87.5% completions** and **+11.4% utility** (seeds 300–339). That is the research-relevant comparator.
2. Vs value-blind B0–B4, utility gains are large; do not conflate that with the ValueGreedy result.
3. `time-to-exit/return` are **privileged scripted schedule features** (addendum §2) — not deployable estimates. Shared with ValueGreedy → fair inside abstract env only.
4. **Not** an NR-V2X Mode-2 result. R3-M’s Mode-2-negative finding stands. See `SEQUENCE_STATUS.md`.

## Reproduce

```bash
cd SafetyOffloading/workflow_v2/drl
python run_drl_experiment.py
```

## Bound

Frozen: MDP config defaults in `workflow_mdp.py`, checkpoint `outputs/ddqn_best.pt`, metrics in `drl_result.json`. No further retuning required for this claim.
