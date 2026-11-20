# Implementation audit and pre-training gate

## Decision

DRL training remains suspended. The implementation audit found and fixed the remote-transport defect, verified genuine CPU contention, and established positive ideal, physical-radio, and safety-feasible offloading headroom. The next valid step is to regenerate training data with the corrected transport path; the old DQN and risk models were trained against the defective environment and are not evidence of policy performance.

## Deterministic acceptance test

The test releases two simultaneous 36 Mcycle tasks at 150 ms with a 100 ms deadline. The origin is fixed at 0.5 GHz and helpers at 2 GHz. The 1,500-byte input is segmented into two WiLab transport blocks.

Results from `outputs/acceptance_20261120_174944_061`:

| Case | On-time tasks | Offloaded | Inputs delivered | Result |
|---|---:|---:|---:|---|
| Local only | 1 / 2 | 0 | 0 | CPU contention passes |
| Ideal helper communication | 2 / 2 | 1 | 1 | Helper computation passes |
| Actual Mode-2 helper | 2 / 2 | 1 | 1 | Bidirectional Mode-2 path passes |

Actual Mode-2 timing for the offloaded task:

- release: 150.000 ms;
- input fragment 1: 164.964 ms;
- input fragment 2 / reassembly: 174.964 ms;
- helper computation: 174.964–192.964 ms;
- result delivered: 196.464 ms;
- end-to-end latency: 46.464 ms.

## Defect and correction

Before the correction, fragment 2 transmitted at 264.964 ms—one full 100 ms SPS/RRI after fragment 1—and missed the 250 ms deadline.

`spv2xWiLabAfterTransmission` correctly assigned the new fragment's generation timestamp to `timeManagement.timeLastPacket`, but `timeManagement` was passed by value and not returned to `mainV2X`. WiLab therefore never observed the new timestamp and retained the old periodic reservation.

The callback now returns both `stationManagement` and `timeManagement`, and `mainV2X` persists both values. WiLab consequently performs a fresh Mode-2 selection inside T1/T2; the two input fragments are now 10 ms apart rather than 100 ms apart.

## CPU audit

The original corrected-workload development run contains 116 tasks, not 109. Local-only completes **109/116**. CPU state persists, overlapping arrivals queue, and deadline checks include queue time.

| Origin | CPU (GHz) | Tasks | Overlaps | Max queue (ms) | On time |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.011 | 20 | 7 | 53.019 | 20 |
| 13 | 1.612 | 24 | 4 | 15.517 | 24 |
| 26 | 1.582 | 21 | 5 | 29.416 | 21 |
| 38 | 0.934 | 33 | 21 | 98.201 | 26 |
| 50 | 1.163 | 18 | 7 | 67.856 | 18 |

All seven local failures occur at origin 38. The deterministic test separately fixes the origin at the configured 0.5 GHz lower bound, because seed 4's slowest selected origin is 0.934 GHz.

## Corrected 36-Mcycle radio results

Matched seed-4, density-50, 2-second results:

| Method | Completed | Gain vs local | Offloads | Action-window safety harm | 1% constraint |
|---|---:|---:|---:|---:|---|
| Local only | 109 / 116 | 0 | 0 | 0 | Pass |
| Instantaneous-communication oracle | 116 / 116 | +7 | 4 | 0 | Pass |
| Actual Mode-2 radio oracle | 116 / 116 | +7 | 4 | 2.794% | Fail |
| Safety oracle, one privileged admission | 110 / 116 | +1 | 1 | 0.941% (4/425) | Pass |
| Existing risk-aware heuristic | 107 / 116 | -2 | 0 | 0 | Pass, but no benefit |

Therefore:

- `Delta_offload = 116 - 109 = +7` tasks for the actual-radio oracle;
- `Delta_safe >= 110 - 109 = +1` task under the configured 1% action-window harm limit;
- the existing risk-aware heuristic leaves measurable headroom for a learned policy.

The global paired PRR drift is also retained as a diagnostic, but the safety constraint is evaluated over each admitted task's 100 ms causal window, matching the risk-training label. Counting the entire remainder of a paired run as one action's externality mixes later RNG-stream divergence into that action's label.

## Pre-fix failure decomposition

The old unrestricted run completed only 1/116 tasks. Its failures were:

| Failure category | Tasks | Share |
|---|---:|---:|
| Input scheduling never produced a transmission | 65 | 56.03% |
| First input transmission after deadline | 44 | 37.93% |
| Input fragment loss or late fragment | 2 | 1.72% |
| Result loss | 4 | 3.45% |
| Completed | 1 | 0.86% |

After the callback fix, unrestricted offloading completes 93/116 on the same corrected workload. It remains inferior to selective offloading, but the remote path is functional rather than nearly dead.

## Training gate

The four required preconditions now have direct evidence:

1. genuine local contention: 109/116 local completion;
2. functional bidirectional V2V offloading: deterministic actual Mode-2 2/2;
3. measurable radio-offloading benefit: +7 oracle tasks;
4. measurable safety-feasible headroom beyond the existing heuristic: safety oracle +1 versus risk-aware -2.

Training may resume only after regenerating experience and risk labels through the corrected callback. Previously frozen models must not be presented as corrected results.

## Reproduction artifacts

- `runDeterministicAcceptance.m` — deterministic local/ideal/Mode-2 gate.
- `auditDevelopmentRun.m` — CPU reconstruction and failure decomposition.
- `runWiLabOpportunityGate.m` — five-method opportunity experiment.
- `summarizeWiLabOpportunityGate.m` — global and action-window safety metrics.
- `outputs/acceptance_20261120_174944_061/` — passing acceptance run.
- `outputs/opportunity_gate_20261120_175521_371/` — local, ideal, radio, initial safety, and risk-aware results.
- `outputs/paired_safety_oracle_rate12tps_density50_20261120_175741_323/` — one-admission safety-oracle result.
