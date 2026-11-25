# workflow_v2

New multi-stage NR-V2X Mode-2 workflow environment (implementation contract v2.0).

**Does not** inherit P4–P9 offloading efficacy claims. Legacy SafetyOffloading MDP remains frozen.

| Stage | Meaning | Status |
| --- | --- | --- |
| R1 | Map & verify WiLab control interfaces → `docs/R1_INTERFACE_MAP.md` | **Frozen PASS** |
| R2 | Workflow engine + R2-01…R2-12 → `docs/R2_EXIT.md` | **Frozen PASS** |
| R3 | Joint-decision relevance → `docs/R3_EXIT.md` | **Closed / frozen** (no joint completion advantage) |
| R3-M | Model-validity gate → `docs/R3M_GATE.md` | Spec ready; run only if authorized |
| R4+ | On hold pending R3-M | Pending |

Config: `config/workflow_pilot.yaml`, `config/r3_pilot.json`  
Acceptance: `tests/runWorkflowAcceptance.m`, `tests/runR2Acceptance.m`  
Pilot: `runWorkflowPilot('smoke'|'pilot')`
