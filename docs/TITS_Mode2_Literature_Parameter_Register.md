# Literature-grounded parameter register for Safety-Preserving V2V Computing

**Date:** 17 November 2026. **Status:** Executable starting configuration at the parameter level; simulator commit/API compatibility, computing input fragmentation/reassembly, and arrival process require engineering verification. This document supersedes the blank experiment-configuration template **for parameter choices**, not as a claim of validated simulation fidelity.

## Source keys

- **T26:** V. Todisco et al., *5G NR-V2X Scheduling Approaches for CPM Variable Size Traffic*, IEEE VNC 2026 / arXiv:2606.26746v1 (supplied PDF), §§II, III, V. https://arxiv.org/abs/2606.26746
- **A21:** Z. Ali et al., *3GPP NR V2X Mode 2: Overview, Models and System-Level Evaluation*, IEEE Access 2021, doi:10.1109/ACCESS.2021.3090855. https://doi.org/10.1109/ACCESS.2021.3090855
- **D25:** M. Dong et al., *Task Offloading and Resource Allocation in Vehicular Cooperative Perception With Integrated Sensing, Communication, and Computation*, IEEE T-ITS 2025, doi:10.1109/TITS.2025.3542365 (supplied PDF), §VI-A.
- **Y26:** S.-J. Yu and C. Mun, *RSU-Assisted Resource Map Sharing to Enhance Reliability in 5G NR V2X Mode 2*, Electronics Letters 2026, doi:10.1049/ell2.70566, Table 1. https://doi.org/10.1049/ell2.70566
- **L25:** J. Liu et al., *Task Offloading and Resource Allocation for ICVs in Vehicular Edge Computing Networks Based on Hybrid Hierarchical Deep Reinforcement Learning*, Sensors 2025, 25(22), 6914, Table 1. https://doi.org/10.3390/s25226914

## 1. Reference radio profile — reproduce T26 first

| Parameter | Fixed value | Source | Interpretation |
|---|---|---|---|
| Sidelink resource allocation | NR-V2X Mode-2 | T26 | Decentralized, sensing/exclusion-based |
| Channel width | **20 MHz** | T26, §V-A | Do not mix with Y26's **10 MHz** configuration |
| Subcarrier spacing | **30 kHz** | T26 | **0.5-ms** slot |
| Number of subchannels | **5** | T26 | **10 resource blocks/subchannel** |
| Tx power | **23 dBm** | T26 | Antenna gain **3 dBi**, noise figure **9 dB** |
| MCS reference | **12** | T26 | 16-QAM, code rate approximately 0.43; reference maximum TB **1180 bytes** for its stated configuration, not an unconditional MAC limit |
| Channel model | ETSI TR 103 439 rural highway | T26 | Do not silently substitute WINNER+ B1 used by Y26 |
| SPS keep probability | **0.8** | T26 | Applies to paper's semi-persistent strategies |
| Periodic source interval / reservation interval | **100 ms** for non-congested CPM; **500 ms** for congested CPM | T26 | RRI equals each source's configured packet period in its experiments. For new 100-ms safety source start with RRI=100 ms |
| Retransmissions | **none** | T26 | It explicitly omits retransmissions; HARQ-on is a sensitivity extension, not a base replication |
| Topology | 8-km, six-lane highway | T26 | 3 lanes in each direction |
| Vehicle density | **50, 150 vehicles/km** | T26 | 400, 1200 active vehicles along 8 km |
| Speeds | **110 ± 11**, **70 ± 7 km/h** | T26 | Paper describes average and SD for the two regimes; retain its mobility distribution implementation |
| CPM traffic construction | P = 53 + 16N + P_sec bytes | T26, §II | P_sec = 231 or 89 B; full certificate once/second; payloads are data-dependent |
| CPM source rate | **100 ms**, **500 ms** | T26 | 100 ms at 50 veh/km; 500 ms at 150 veh/km. This co-varies density and message rate: **do not claim density-only causality** |

**Required model detail, not safely filled by a literature number:** sensing-window length, exact selection-window values, resource-candidate exclusion thresholds, reselection-counter distribution, UE priority-to-MAC mapping, and simulator repository release. These are implementation/profile-specific and must come from an audited checked-out version of the radio implementation and its 3GPP configuration. A value invented from memory would compromise standards fidelity.

## 2. Independent safety-message profile

| Parameter | Start value | Evidence | Qualification |
|---|---|---|---|
| Periodic safety packet | **300 B** | A21 simulation traffic, and Y26 Table 1 CAM | Clearly an example traffic profile, not a universal CAM size |
| Generation period | **100 ms** | A21 and Y26 | 10 Hz periodic traffic |
| Physical channel | Same 20-MHz, 30-kHz T26 resource pool | **Research design** | Necessary to measure safety externality; neither paper evaluates this exact mixed workflow |
| Packet deadline | **100 ms** | **Study choice**, equal to message period | The papers do not establish a universal safety deadline from this value; evaluate deadline sensitivity |
| PDR threshold | **0.90** | Y26 uses PDR >= 0.9 in its coverage metric | This is a published **performance benchmark**, not a 3GPP safety guarantee |
| Evaluation range bins | 0–300 m in 50-m bins | **Study choice** | Publish the actual receiver range distribution and PDR vs range |

## 3. Computing workload profiles — do not silently combine them

### D25: lightweight sensing-subregion processing

- Each subregion: **12 kbits = 1500 bytes** (D25, §VI-A).
- Computational intensity: **200 cycles/bit**; task requires **2.4 million cycles** if represented as one subregion.
- Vehicle CPU frequency: **500–2000 Mcycles/s** (sampled from the cited interval); 10-Gcycles/s RSU is **not** a V2V helper and must not be copied into a V2V-only architecture.
- D25 does **not** provide an independently justified V2V **result payload**, a matching task-input arrival distribution, or a fixed universal deadline for the 1500-B task. For a pilot, set result to **300 B** and deadline **100 ms**, both explicitly marked as study assumptions.
- Even a 1500-B task can exceed the 1180-B nominal reference TB from T26. Thus, packetization or a higher capable allocation is mandatory; **never silently treat 1500 bytes as one 1180-byte TB**.

### L25: heavier vehicular computing sensitivity profile (not a Mode-2 validation)

L25, Table 1 reports **input 0.5–11 Mbits**, **1–4.5 Gcycles/Mbit** computational intensity, **0.5–3 s deadlines**, and vehicle CPU **0.5–1.55 GHz**. Its source uses RSU MEC, not V2V Mode-2; these are **workload-range transfer assumptions**. Exact joint distributions and task arrival process are *not* established by the parameter range alone. The YAML specifies a study-chosen stratified-uniform sampler, explicitly not a claim about the paper's sampler.

Large inputs in this profile correspond to **62.5 kB–1.375 MB**, requiring many Mode-2 transport blocks; a highly loaded packet pool may make V2V offloading mostly infeasible. This is a valid potential negative finding. The experiment must not quietly substitute an ideal wideband input transfer.

## 4. Safety thresholds: separate published benchmark from research risk budget

- **Absolute failure target 0.10** is the complement of the Y26 paper's **PDR >= 0.90** reporting benchmark. It is a chosen **study constraint**, not a safety regulation.
- **Incremental harmful-flip target 0.01** (and sensitivity **0.005, 0.02**) are explicitly **researcher-selected**, because no inspected paper prescribes a matched-world allowable safety-externality threshold. These cannot truthfully be called 'derived from literature'.
- Study design: stratify by density and distance, and require baseline feasibility. If zero-offload PDR is below 0.90 in a stratum, flag the absolute constraint as **infeasible** rather than reporting protection success.
- For **n** independent harmful-flip Bernoulli observations with zero harmful events, a one-sided 95% exact upper bound below **1%** requires **at least 299** independent trials (since log(0.05)/log(0.99) ≈ 298.07). Receiver-packet observations from a common traffic trace are correlated; use independent scenario clusters / cluster-robust methods and do not equate millions of packets with millions of independent observations.

## 5. Experimental run length, platform, and mixing rules

- **200 s** is reported by Y26 Table 1, **not** T26. Adopting it for our run length is a cross-paper **study design**; it is not replication of T26.
- **20 s warmup**, **50 scenario seeds per regime**, and **100-ms admission decision interval** are **study choices**. The source literature does not prescribe them for this problem; they do not prove 1% reliability.
- **WiLabV2XSim** is used by T26 and Y26. A verified repository **version/commit** must be frozen. Supporting CPU queues, packet fragmentation/reassembly, bidirectional task inputs/results, action hooks, and paired-world random streams requires explicit code additions and tests.
- **OMNeT++** remains an alternative, but **OMNeT++ + INET + Veins + SUMO is not an NR Mode-2 implementation by itself**; a validated Mode-2 extension is required. Never mark its Mode-2 features as verified without inspecting and running the chosen implementation.
- Keep **physical configuration T26** as the primary 20-MHz, 30-kHz profile. If validating against Y26, run a *separate* 10-MHz, 15-kHz, 5 × 12-PRB, MCS-7 (CAM), WINNER+ B1 reference profile—not a hybrid of T26 and Y26 parameter tables.

## 6. What still requires an explicit design decision, not an invented literature citation

1. V2V result-packet size and code/task output compression, which vary by application.
2. Computing job arrival process; papers often report task property ranges without a common arrival distribution.
3. Traffic-class priority configuration and the specific standard-based preemption features supported by the simulator.
4. Formal acceptable **incremental** safety-harm threshold, which is a policy/design objective, not a number mandated by the reviewed sources.
5. Feasibility of large input tasks on the chosen channel under actual fragmentation, expiration, and simultaneous safety traffic.

The YAML makes these either explicit study assumptions or version-dependent configuration requirements. This is the honest distinction between **specifying a reproducible baseline** and **falsely claiming that every parameter was established by the literature**.
