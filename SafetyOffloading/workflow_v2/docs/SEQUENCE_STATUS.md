# workflow_v2 research sequence — closed status

Authoritative end-state after R4 freeze addendum. Supersedes contradictory one-liners in older exits where they conflict.

## Component status

| Component | Status | Binding docs |
| --- | --- | --- |
| R1–R2 implementation validation | **Closed** | `R1_INTERFACE_MAP.md`, `R2_EXIT.md` |
| R3 joint-decision relevance (Mode-2 pilot) | **Closed** | `R3_EXIT.md` |
| R3-M Mode-2 model-validity gate | **Closed** | `R3M_EXIT.md`, `R3M_GATE.md` |
| R4 abstract-MDP DRL experiment | **Accepted and frozen** | `R4_DRL_EXIT.md`, `R4_DRL_IMPLEMENTATION.md`, `R4_DRL_FREEZE_ADDENDUM.md` |
| Matched holdout vs ValueGreedy | **Complete** | `drl/outputs/r4_matched_holdout.json` |
| Observation-information audit | **Complete** | `R4_DRL_FREEZE_ADDENDUM.md` §2 |
| Mode-2 DRL integration | **Not authorized** | — |
| Further algorithm / MDP development | **Not authorized** | — |

## How R3-M and R4 relate (must stay separate)

| Experiment | Question answered | Outcome |
| --- | --- | --- |
| **R3-M** | Under WiLab NR-V2X Mode-2 as frozen, does joint offload beat local on deadline completions? | **No** — B0 ≫ remote. Mode-2 stop stands. |
| **R4** | In an **abstract** capacity-constrained prioritized DAG MDP with **known** future contact times, does DDQN beat tested heuristics? | **Yes vs value-blind; yes vs ValueGreedy on completions (+87.5%) with +11.4% utility** on matched holdout. |

R4 was opened under an **authorized problem redesign** after R3-M’s Mode-2-negative gate. It does **not** reopen or overturn R3-M. Older R3-M text “No R4 / No DRL” means **no R4/DRL on the Mode-2 line**; the later abstract-MDP R4 is a separate, frozen claim.

## Demonstrated vs not demonstrated (R4)

| | |
| --- | --- |
| **Demonstrated** | DDQN improves prioritized workflow **admission and placement** over the tested priority-aware heuristic (ValueGreedy) in an abstract, capacity-constrained vehicular computing environment **with known future contact transitions**. Primary matched signal: **+87.5% completions** with **+11.4% utility**. |
| **Not demonstrated** | A novel, **deployable NR-V2X Mode-2** algorithm that outperforms strong priority-aware optimization. |

## Nonnegotiable boundaries

1. **Privileged observation:** `time-to-exit/return` are exact scripted future contact times. ValueGreedy shares that schedule knowledge → matched comparison is fair *inside* the abstract env; **neither result establishes deployability**.
2. **Abstract radio:** R4 performance is not a Mode-2 performance result. R3-M’s negative finding remains valid for its tested scenario.

## Frozen artifacts to preserve together

- Checkpoint: `drl/outputs/ddqn_best.pt`
- Matched eval: `drl/outputs/r4_matched_holdout.json`
- Privileged-information classification: `R4_DRL_FREEZE_ADDENDUM.md`
- MDP defaults: `drl/workflow_mdp.py` (unchanged)
