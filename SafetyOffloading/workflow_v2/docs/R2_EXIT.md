# R2 exit — workflow engine correctness

**Status:** PASS (2026-11-23)  
**Runner:** `workflow_v2/tests/runR2Acceptance.m`  
**Mode:** `scripted_unit` radio (contract-allowed deterministic fixtures); control still via `WorkflowControlProtocol` (no free oracle).

## Results

| Gate | Outcome |
| --- | --- |
| R2-01 … R2-12 | **12/12 PASS** |
| R1 regression | **7/7 PASS** (frozen) |
| Reproducible event log | `W.log` via `WorkflowMetrics` |
| Dependency / CPU / commit / deadline invariants | Exercised by suite |

## Engine surface

- `core/WorkflowEngine.m` — lifecycle, placement, edges, scripted delivery, buffering
- `core/WorkflowEventDispatcher.m` — §B3 ordered `step_to`
- `core/WorkflowCpuQueue.m` — persistent nonpreemptive EDF (R1)
- R1 adapters unchanged except CPU `meta` optional field (R1-06 still green)

## Next

R3 joint-decision relevance pilot — no DRL until R3 exit and R5 baselines.
