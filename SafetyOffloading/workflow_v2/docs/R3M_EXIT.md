# R3-M exit — CLOSED

**Status:** Gate complete. Predeclared design frozen in `config/r3m_gate.json`.  
**Outputs:** `outputs/r3m_gate_20261123_223308/`  
**R3 remains frozen.**  
**Mode-2 line:** no R4/DRL *on this WiLab Mode-2 gate* (decision below stands).  
**Later abstract-MDP R4:** separate authorized redesign — accepted/frozen; see `SEQUENCE_STATUS.md` and `R4_DRL_FREEZE_ADDENDUM.md`. Does **not** overturn this Mode-2 result.

## Deadline-valid completions (`n_success` / 12)

| Policy | Seed 1 | Seed 2 | Seed 3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| B0 local | 10 | 10 | 10 | **10.0** |
| B2 queue | 4 | 5 | 5 | 4.7 |
| B3 mobility | 5 | 5 | 5 | 5.0 |
| B4 joint | 6 | 6 | 8 | 6.7 |
| B4-immediate | 6 | 6 | 8 | 6.7 |

Remote stages done (mean signal): B0=0; B2/B3≈2; B4/B4i≈15.

## Mechanism checks (predeclared)

| Mechanism | Observed |
| --- | --- |
| Workflow contention | Yes — no policy reaches 12/12; metric not saturated |
| Helper volatility + remote work | Yes — B4/B4i perform remote stages under scripted helper-2 exit/return |
| Coupled release (HOLD/RELEASE) | **No advantage** — B4 = B4-immediate on every seed |

## Decision (frozen table)

| Finding | Action |
| --- | --- |
| Joint commitments improve deadline-valid completions under this contention | → R4 |
| Interactions exist but B2 ≈ B4 | Simplify; no joint-planning claim |
| No meaningful joint-decision effects | Stop this contribution hypothesis |

**Applied outcome: Stop this contribution hypothesis (no R4).**

- B4 beats B2/B3 (~6.7 vs ~4.7–5.0), so remote joint planning is not identical to queue-only offload.
- **B0 dominates all remote policies (10.0).** Joint Mode-2 offloading does not improve deadline-valid completions relative to local execution under the predeclared contention/churn gate.
- Release control is inert (B4 = B4-immediate).

Therefore the research claim that motivates joint commitment / DRL scheduling **under this Mode-2 WiLab regime** is **not supported**. Do not open Mode-2 R4/DRL from this gate. (A later abstract-MDP learning experiment was authorized separately; it is not a Mode-2 claim.)

## Bound

One corrective iteration allowed only for defects in the gate harness — **not** for retuning arrivals, contact schedule, or policies to manufacture a B4 win.
