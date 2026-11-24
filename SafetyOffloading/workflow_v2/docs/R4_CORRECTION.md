# R4 correction — causal observation + Mode-2-faithful transport

**Intent:** Preserve the frozen R4 experiment; fix its two limitations without redesigning the research problem or DDQN architecture.

| Problem | Correction | Unchanged |
| --- | --- | --- |
| Exact future contact times in obs | Causal contact age + advertisement age | DAG, objective, actions, 16-D net |
| Abstract V2V delay | Event-driven `mode2_event` radio | R4 workload, priorities, CPU, deadlines, admission |

**R4 reference checkpoint `drl/outputs/ddqn_best.pt` is never overwritten.**

## Correction A — observation

Removed (privileged):

```text
to_exit = max(0, contact_exit - t) / horizon
to_ret  = max(0, contact_return - t) / horizon
```

Replaced (causal, ego-only):

| Feature | Definition |
| --- | --- |
| `contact_age` | If helper currently in contact: `(t - last_contact_change_t) / horizon`; else `0` |
| `advertisement_age` | `(t - last_adv_rx) / ttl` clipped to `[0,1]`; `1` if never received |

Advertisements are received **only while contact is up**, on a fixed ego RX poll period. The private mobility schedule still drives ground-truth contact flips but is **not** readable by `_obs()` in `obs_mode=causal` or by deployable baselines (`policy_b3` / ValueGreedy).

`obs_mode=privileged` remains available solely to reproduce the R4 reference.

New checkpoint: `drl/outputs/ddqn_causal_obs.pt` (trained with causal obs + abstract radio).

## Correction B — radio

`radio_mode=mode2_event` replaces the single abstract delay with:

```text
COMMIT_REQ → ACK → input fragment TX → helper CPU enqueue → execution → output fragment TX → stage DONE
```

Implemented in `drl/mode2_radio.py` to match R1–R3 adapter semantics (TB 1180 B, fragmentation, contact gating, stranding). Remote CPU **cannot** start before `INPUT_READY`. Utility requires terminal readiness before the deadline.

This is **event-driven Mode-2-faithful transport inside the Python MDP**, not a full WiLab PHY run. Full WiLab NR-V2X Matrix remains a later verification layer if authorized; it is not claimed here as complete PHY fidelity.

New checkpoint: `drl/outputs/ddqn_mode2.pt` (causal obs + `mode2_event`).

## Evaluation matrix

| Evaluation | Observation | Radio | Checkpoint |
| --- | --- | --- | --- |
| R4 reference | privileged | abstract | `ddqn_best.pt` |
| Corrected observation | causal | abstract | `ddqn_causal_obs.pt` |
| Corrected radio | causal | mode2_event | `ddqn_mode2.pt` |
| Priority-aware comparator | same as row | same as row | ValueGreedy (causal B3) |

Runner: `drl/run_r4_correction.py` / `run_r4_correction_resume.py` → `drl/outputs/r4_correction_results.json`.

Matched holdout seeds: **300–339**. Metrics: deadline-valid **utility** and **completion count**, paired vs ValueGreedy.

## Holdout results (seeds 300–339)

| Evaluation | DDQN utility / N | ValueGreedy utility / N | Δ utility | Δ N |
| --- | ---: | ---: | ---: | ---: |
| Corrected observation (causal + abstract) | 30.25 / 11.05 | 30.43 / 7.12 | **−0.6%** | **+55.1%** |
| Corrected radio (causal + mode2_event) | 19.30 / 6.20 | 19.40 / 4.10 | **−0.5%** | **+51.2%** |

Paired completion wins remain strong (obs: 39/1/0; radio: 35/3/2). Utility vs ValueGreedy is **not** a material gain after removing privileged leakage / adding event-driven radio.

`ddqn_best.pt` preserved. New: `ddqn_causal_obs.pt`, `ddqn_mode2.pt`.

## Causal dueling holdout (BC + early-DROP; no privileged obs)

Full write-up: [`CAUSAL_DUELING_DDQN.md`](CAUSAL_DUELING_DDQN.md).

Phased-arrival MDP; PDP (BC teacher) warm-start + Dueling DDQN fine-tune. Seeds 300–339:

| Policy | Utility | Completions |
| --- | ---: | ---: |
| ValueGreedy | 78.40 | 7.45 |
| PDP teacher (Always-Drop-Low; not a literature baseline) | 99.30 | — |
| **DDQN causal dueling** | **101.80** | — |

vs ValueGreedy: **+23.4 utility (+29.8%)**; ambitious 15% target met. Checkpoint `ddqn_causal_dueling.pt`; `causal_dueling_holdout.json`.

## Reproduce

```bash
cd SafetyOffloading/workflow_v2/drl
python run_r4_correction.py
# or resume after causal ckpt exists:
python run_r4_correction_resume.py
# causal utility run (BC + dueling):
python train_causal_dueling_ddqn.py
```
