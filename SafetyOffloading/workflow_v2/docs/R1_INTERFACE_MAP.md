# R1 — WiLab / SafetyOffloading interface map

**Status:** Repository inspection complete for this fork. Not a green R1 exit — several capabilities are **adapter-extend** or **go/no-go**.  
**Date:** 2026-11-23  
**MATLAB:** R2026a Update 5 (`26.1.0.3346908`), `C:\Program Files\MATLAB\R2026a`  
**Git SHA:** not recorded (repo `safe.directory` ownership block; do not alter git config in-session)  
**Base:** `safety-v2x-pipeline/SafetyOffloading` + WiLab hooks in `MainFiles/mainV2X.m`  
**Legacy MDP:** frozen under existing `WiLabIntegration/spv2x*`; new work only under `workflow_v2/`.

## Capability matrix (contract §L)

| Capability | Verified mapping | Status | Notes |
| --- | --- | --- | --- |
| Application packet submission | `spv2xWiLabTick` / `makeFragments` → `s.pending_tx{v}` → `stationManagement.pckBuffer(v)=1` | **PASS (adapter)** | App owns fragment list; WiLab owns Mode-2 resource selection. |
| Post-transmission callback | `spv2xWiLabAfterTransmission` → returns `[stationManagement,timeManagement]`; `mainV2X.m` ~L280 persists both | **PASS** | Fixes stale 100 ms SPS between fragments (see `IMPLEMENTATION_AUDIT_RESULTS.md`). |
| Fragment RX / reassembly | `spv2xWiLabReception` counts `input_received`/`output_received` vs `*_fragments`; status flip on full count | **PASS (extend)** | Binary task stages `input`/`result` only today. Workflow edges need `edge_id`/`message_id` in meta. |
| Payload→TB conversion | `cfg.radio.max_transport_block_bytes=1180` in `config/training_data.json`; `ceil(bytes/1180)` in Tick/ScheduleEpoch | **MEASURED** | App-side fragmentation assumes 1180 B = full TB. Contract requires MAC-effective size check before claiming exact TB math. |
| Application HOLD/RELEASE | No named HOLD API. Hold = keep edge off `pending_tx` / do not set `pckBuffer`. Release = enqueue fragments then refill buffer | **PASS (pattern)** | Whole-edge only; no per-fragment pacing control (matches contract §F1). |
| Safety traffic coexistence | `spv2xWiLabSafetyGenerated`; `generationInterval`, 300 B profile | **PASS** | Remains independent background; do not optimize offload count. |
| Neighbor / contact observability | Positions via `positionManagement.distanceReal`; CBR via `stationManagement.cbrCV2Xvalues` | **PARTIAL** | Distance/CBR used in legacy obs. Beacon-derived contact state not wrapped as `OBSERVED_REACHABLE` enum. |
| Vehicle mobility | Existing WiLab highway mobility / road traces | **PASS** | Contact churn must be **measured** in R3, not scripted in pilot scenarios. |
| Dynamic CPU service | `workflow_v2/core/WorkflowCpuQueue.m` (nonpreemptive EDF + `set_rate` settle) | **PASS (adapter)** | R1-06 rate-change integral verified. Legacy Tick queues untouched. |
| HELPER_ADV control traffic | `workflow_v2/network/WorkflowControlProtocol.m` → `enqueue_ctrl` onto `pending_tx` | **PASS (adapter)** | App payload `HELPER_ADV` with TTL; no free oracle. R1-05 covered. |
| COMMIT_REQ/ACK/NACK | Same module: `make_req` / `helper_handle_req` / ACK|NACK + timeout | **PASS (adapter)** | Nonce-idempotent ledger; remote RELEASE gated on `commitment_acked`. |
| Radio statistics | CBR logged; PRR via safety packets; resource_id sometimes in radio_rows | **PARTIAL** | Fabricate nothing; null unsupported internals. |
| Scenario RNG pairing | WiLab `seed` + overlay decision counters | **EXTEND** | Contract demands named streams keyed by `(scenario_seed, entity, event, time, draw_index)`. |

## Legacy hooks to reuse (not rewrite)

| Hook | Path | Role for workflow_v2 |
| --- | --- | --- |
| Tick / next event | `MainFiles/mainV2X.m` + `spv2xWiLabTick` | Replace body with `WorkflowEventDispatcher` behind new policy flag; keep callback registration pattern. |
| After TX | `spv2xWiLabAfterTransmission.m` | Reuse; generalize meta kinds beyond `compute`/`safety`. |
| Reception | `spv2xWiLabReception.m` | Extend meta for `workflow_id`/`edge_id`/`commitment`; keep fate-row parsing. |
| Safety gen | `spv2xWiLabSafetyGenerated.m` | Keep enabled for coexistence logs. |
| Deployable strip precedent | `spv2xDeployableObservation.m` | Pattern only — new obs schema in `WorkflowObservation.m`. |

## Explicit non-assumptions confirmed

- No direct PHY slot / RB / MCS actions from application (correct).
- No application HARQ retransmit path in current overlay (correct for D-first).
- Policy cannot write helper true queues remotely today (correct); admission must be ACK-gated.
- Fragmentation is application-precomputed, not recomputed from live MAC TB — **must upgrade or document limitation** before R3 radio claims.

## R1 go / no-go

| Gate | Decision |
| --- | --- |
| Packet enqueue + Mode-2 TX/RX + timestamp persistence | **GO** — verified |
| Whole-edge HOLD/RELEASE via deferred enqueue | **GO** — implementable without MAC surgery |
| Helper ADV + COMMIT over Mode-2 as app payloads | **GO** — adapter + R1-05 acceptance pass (unit); air path reuses `pending_tx` |
| Exact MAC TB / overhead introspection | **LIMITATION** — freeze 1180 B assumption or add measurement test before TB-sensitive claims |
| Proceed to R2 workflow engine | **Allowed** — local + remote commitment protocol available; R3 still blocked until R2 fixtures pass |

## R1 corrective iteration (2026-11-23) — landed

| Module | Path | Tests |
| --- | --- | --- |
| Mode-2 HOLD/RELEASE | `network/WorkflowMode2Adapter.m` | R1-01, R1-03 |
| Control protocol | `network/WorkflowControlProtocol.m` | R1-05 |
| CPU queue | `core/WorkflowCpuQueue.m` | R1-06 |
| Deployable obs | `policy/WorkflowObservation.m` | R1-04, R1-07 |
| Acceptance runner | `tests/runWorkflowAcceptance.m` | R1-01…R1-07 |

## Next actions (bounded)

1. R1 **frozen**. R2 **PASS** — see `docs/R2_EXIT.md`.  
2. Proceed to R3 relevance pilot when authorized — no DRL until R3+R5.
