# R3-M — model-validity gate

**Purpose:** One final, bounded test of whether the intended research system exhibits the resource interactions that motivate joint scheduling — **not** to demonstrate B4 superiority.

**R3 remains frozen.** No post-hoc workload edits. No Mode-2 DRL from this gate (see `R3M_EXIT.md`). Abstract-MDP R4 is separate — `SEQUENCE_STATUS.md`.

## Frozen numerical design (`config/r3m_gate.json`)

| Item | Value |
| --- | --- |
| Topology | diamond-4, 9/12/12/3 Mcycles, 500 ms deadline |
| Arrivals | `[0.10, 0.13, …, 0.43]` — **12** workflows, 30 ms spacing |
| Rationale | Local serial service \(12\times 72\,\mathrm{ms}=864\,\mathrm{ms}\) from first release exceeds several absolute deadlines → completion metric not saturated a priori |
| Contact | **Scripted (labeled):** helper-2 exits at **0.22 s**, returns at **0.40 s**; helper-3 always reachable |
| BG load | ρ=0.80 (same as R3) |
| Sim horizon | **1.5 s** |
| Seeds | **1, 2, 3** (no expansion after results) |
| Algorithms | B0, B2, B3, B4, B4-immediate (release ablation) |
| Radio | `wilab_mode2` |

## Mechanisms

| Mechanism | Evidence target |
| --- | --- |
| Workflow contention | Overlapping arrivals + incomplete local capacity |
| Helper volatility | Scripted helper-2 exit/return during outstanding work |
| Coupled scheduling / release | B4 vs B4-immediate; compare B0/B2/B3/B4 completions |

## Decision rule

| Finding | Action |
| --- | --- |
| Joint commitments improve deadline-valid completions under this contention | → R4 |
| Interactions exist but B2 ≈ B4 | Simplify; no joint-planning claim |
| No meaningful joint-decision effects | Stop this contribution hypothesis |

## Execution

| Item | Status |
| --- | --- |
| Numbers frozen in `config/r3m_gate.json` | **Yes** |
| Runner | `runR3MGate.m` / parallel `runR3MCase.m` |
| Runs | `outputs/r3m_gate_20261123_223308/` |
| Exit | **CLOSED** — `docs/R3M_EXIT.md` (no Mode-2 R4/DRL; abstract R4 separate) |
