# Safety-Preserving Distributed Vehicular Computing over NR-V2X Mode-2
## System Model, Constrained Optimization, POMDP/CMDP, Algorithm Alternatives, and Implementation Protocol
**Research design specification — 17 November 2026**  
**Status:** Implementable research design; **not** a claim of proven algorithm superiority or a certified over-the-air safety guarantee.

## Document control

- **Research question:** When safety communications and optional V2V computational offloading share an NR-V2X Mode-2 resource pool, which tasks should be admitted, where should they execute, and when should their packets be sent so as to maximize deadline-valid computing benefit **without excessive incremental harm to pre-existing safety messages**?
- **System boundary:** Vehicles make application-layer admission, local-execution and helper-selection decisions. A standards-based Mode-2 MAC/PHY independently performs permissible sidelink resource (re)selection, reservation, priority handling and packet transmissions. DRL **does not directly overwrite arbitrary time-frequency allocations**.
- **Primary novelty hypothesis to test, not assume:** Estimating **offload-induced safety-message harm** and using it in admission/placement yields a superior computing–safety performance frontier to traffic-priority-only, congestion-threshold and existing cross-layer offloading controllers.
- **Primary implementation platform:** validated 5G-LENA/ns-3 NR-V2X Mode-2 extension **or** WiLabV2XSim, selected after a concrete feature-gap audit. Record versions, 3GPP release, exact MAC capabilities and any extensions. These are alternatives, not interchangeable implementations.
- **No fixed RL algorithm:** train and compare discrete Q-learning, actor–critic, off-policy and safe-RL approaches after building the simulator and nonlearning baselines.

## 1. Evidence and nonclaims

### 1.1 Literature anchoring

| Source | Verified system-level observation | Implication and boundary |
|---|---|---|
| Dong et al., *Task Offloading and Resource Allocation in Vehicular Cooperative Perception With Integrated Sensing, Communication, and Computation*, **T-ITS 2025**, DOI 10.1109/TITS.2025.3542365; user PDF pp. 1, 4–10 | Central unit and RSU/edge-assisted sensing assignments, V2I OFDMA bandwidth and compute allocation; value/latency/energy objective and PPO-based downstream decisions | Joint computation and communications optimization **already exists**. This design instead focuses on externalities to **independent safety packets** from optional V2V computation traffic and Mode-2 packet contention. |
| Chan et al., *Energy-Efficient Collaborative Perception: A Block-Skipping DNN Approach With Dynamic Frequency Scaling and Environment Awareness*, **T-ITS 2026**, DOI 10.1109/TITS.2026.3655760; user PDF pp. 1, 5–7 | Computes energy–perception–response-time trade-offs with DNN skipping, frequency, associations and power; an idealized rate/communication delay expression | Traffic-related deadlines alone are not novel; distinguish per-safety-packet harm and packet-level radio competition. |
| Jia et al., *C-MASS: Combinatorial Mobility-Aware Sensor Scheduling for Collaborative Perception With Second-Order Topology Approximation*, **TVT 2026**, DOI 10.1109/TVT.2025.3621636; user PDF pp. 1–3, 7–9 | Complementary perception utility and a theoretical scheduling scheme | This project's main reward is **deadline-valid computation**, not a new claim of value-aware sensor selection. Optional perception extension must be justified separately. |
| Todisco et al., *5G NR-V2X Scheduling Approaches for CPM Variable Size Traffic*, **VNC 2026 preprint**, arXiv:2606.26746v1; user PDF pp. 1–7 | Packet-size-dependent reservation churn, padding, MCS adaptation and observed effects on PRR in a Mode-2 system-level implementation; no relevance filtering in its traffic reconstruction | Preserve realistic MAC behavior; task payloads and bidirectional offload traffic can create nonperiodic demands. No claim that a simple bandwidth surrogate is Mode-2. |
| Ali et al., *3GPP NR V2X Mode 2: Overview, Models and System-Level Evaluation*, **IEEE Access 2021**, DOI 10.1109/ACCESS.2021.3090855 | Describes sensing/resource selection and open 5G-LENA/ns-3-based simulator | Implementation fidelity reference; must confirm exact software version before reproducing. |
| 3GPP TR 38.885 / TS 38.321 (selected releases) | Mode-2 UE-autonomous resource selection; sidelink MAC grant and re-evaluation rules | Use the normative release targeted by the actual simulator; not a free centralized resource scheduler. |
| Indulekha and Venkatesh, *Integrated Sensing, Communication, and Computing for NR-V2X: A Cross-Layer Resource Allocation Framework Using Multi-Agent Reinforcement Learning*, arXiv:2605.24972 (2026) | Multiagent cross-layer SPS and computation-offloading control already proposed | **MAPPO itself is not novelty**. Include a matched cross-layer learning baseline if implementation can be reconciled. |
| *Modeling and Analysis of Priority-Aware Resource Preemption in 5G NR-V2X Mode 2 Sidelink Communications*, IEEE INFINITYS 2026, DOI 10.1109/INFINITYS69204.2026.11666334 | Safety-message priority and preemption behavior are established | Protecting high-priority messaging is not a new idea. Test incremental protection from *computing admission* beyond existing priority controls. |
| Zhang et al., *CVaR-Constrained Policy Optimization for Safe Reinforcement Learning*, **TNNLS 2025**, DOI 10.1109/TNNLS.2023.3331304 | Risk-sensitive constrained RL methods exist | CVaR, Lagrangians, and safety shields are candidate solvers, not contributions by themselves. |

**Literature limitation:** The four uploaded PDFs were directly inspected; other sources above were verified at accessible publisher/standards/abstract level. This is a specification informed by available literature, not an exhaustive novelty proof. Avoid unqualified 'first' claims.

### 1.2 Distinct research claim and falsification

**Hypothesis H1:** At matched safety-message conditions, enabling ordinary deadline-aware V2V computation offloading can produce a measurable fraction of **newly failing** safety packet–receiver opportunities relative to no offloading.

**H2:** A controller using conditional incremental-risk estimates improves the Pareto frontier between completed task value and incremental safety harm relative to safe-priority-only and CBR-threshold strategies.

**H3:** Helper selection matters beyond aggregate admitted bytes because two placements with similar payloads can create different spatial/temporal interference patterns and task-result traffic.

**Reject or narrow the project if:** H1 is negligible under representative, credible traffic regimes; H2 is not significant versus tuned nonlearning safeguards; or H3 disappears when task payload and transmission opportunities are matched. A new RL agent by itself is not a contribution.

## 2. System model

### 2.1 Entities, topology, and time

- Road region \(\mathcal R\) with vehicles \(\mathcal V(t)\) having positions \(p_v(t)\), speeds \(u_v(t)\), neighbor reachability \(\mathcal N_v(t)\), channel and processor states. No central gNB scheduler is assumed.
- Two traffic classes **on the same specified Mode-2 resource pool**: (S) independently generated safety/application messages, and (C) optional computing-control messages, task inputs, and result packets. Safety traffic generation must not depend on the admission policy.
- Underlying packet/PHY clock: radio slots \(\Delta_s\) configured by numerology. Application decision epochs \(\Delta_a\) (e.g., a multiple of slots) are not assumed equal to the safety-message generation interval. Packet delivery, reservations, mobility, and CPU completions proceed event-by-event between decisions.
- Representative safety flows: periodic awareness and event-driven warnings; add CPM only when source and payload construction are justified. **Message names do not automatically imply a standardized bearer, QoS or priority in the simulator.** Configure and report actual flows/priority mapping.
- Task \(k\) is released at \(r_k\), has initiator \(o(k)\), input bytes \(b_k^{in}\), output bytes \(b_k^{out}\), required cycles \(c_k\), deadline \(d_k\), optional application value \(v_k\), and finite candidate helpers \(\mathcal H_k\subseteq\mathcal N_{o(k)}\). Include setup/control overhead only if actually simulated.
- Vehicle \(v\): CPU rate \(f_v(t)\) cycles/s, work queue \(Q_v(t)\), current energy budget if studied, and processing discipline defined in §2.4.

### 2.2 Baseline and treatment worlds

For each exogenous realization \(\omega\) (vehicle trajectories, safety packet *generation*, propagation-driving randomness, CPU demand), create:

1. **Reference world 0:** same vehicles and safety traffic, with *all optional computing-related PC5 packets absent*. Local computation may still occur; no offload transmissions.
2. **Policy world \(\pi\):** identical exogenous inputs, safety traffic remains the same, plus permitted offloading-related packets and their causal effects on queueing, sensing, reservation and interference.

Do **not** force the resulting MAC internal states, collisions or resource reservations to match across worlds; those are endogenous effects. Maintain common random numbers for exogenous mobility, fading generation and safety releases where simulator supports them; document desynchronization of random-number streams if it does not. Every safety transmission opportunity has a stable \((\text{flow ID},\text{sequence},\text{receiver ID},\text{deadline})\) key in both worlds.

### 2.3 Radio model and assumptions

Implement 3GPP-based NR sidelink Mode-2 resource sensing and selection, SCI-informed reservations, subchannel/slot allocation, SPS reselection counters/keep probability where supported, PSSCH/PSCCH, BLER versus SINR, half-duplex collision limits and optional retransmission/HARQ **only if supported and enabled**. Define the chosen simulator and exact features in a feature-audit table prior to experiments. Do not import an aggregate logarithmic rate equation as a replacement for these mechanisms.

For every PC5 packet \(m\): source, intended receivers, class and configured priority, generation and expiry times, payload and protocol overhead bytes, segmentation if enabled, radio resource grants, attempts, decode events and receiver-specific success. For computing tasks, model both input and result packet(s), with their different transmitters and sizes. Safety messages and computation messages must contend for the *same* configured resource pool in the principal treatment; use separate-pool conditions only as controls.

**Radio-to-application interface:** the offloading controller chooses **admit, placement and, optionally, application submission time within the task slack**. It does not directly set arbitrary sidelink reservation maps. The MAC enforces protocol rules, message priorities and resource availability.

### 2.4 Computation model and packet/task event flow

1. Task released and enters origin's decision queue.
2. Controller may reject, retain for local execution or assign helper \(j\in\mathcal H_k\). Reject may be an intentional action; accepted-but-expired is not treated as successful.
3. Local: enqueue cycles in \(Q_{o(k)}\); finishing by \(d_k\) yields local task completion, with no computing-related PC5 traffic.
4. Offloaded: packetize input, transmit via PC5 to helper, decode; on successful complete input reception enqueue on \(Q_j\). After processing, packetize result and transmit back to origin; origin must fully decode the result by \(d_k\). A late result produces no deadline-valid reward.
5. Failures: no successful input, helper disconnects, queue expires, no successful result return, task deadline passes. Specify abandonment/timeout and whether retries, duplication, cancellation and preemption are enabled. **Default v1: no task replication, no re-offloading and no application-level retries; MAC retransmission behavior follows chosen simulator.**

For task \(k\), \(C_k^\pi=1\) iff local execution finishes before deadline OR (input fully delivered + helper processing completed + result fully delivered by deadline). Admission alone never counts as completion.

**CPU service:** use preemptive or nonpreemptive EDF consistently; recommended v1 nonpreemptive EDF per processor with deterministic service \(c_k/f_j\) under fixed \(f_j\), then add CPU variability as sensitivity. Record both waiting and processing times. When tasks are pending in multiple queues, enforce a single-execution owner.

## 3. Optimization problem

### 3.1 Decision variables

At application epoch \(t\), for each *unassigned* available task \(k\):

\[x_{k,j,t}\in\{0,1\},\quad j\in\{\mathrm{local}\}\cup\mathcal H_k;\qquad z_{k,t}\in\{0,1\}\text{ (reject)}.\]

Each task can receive at most one definitive initial assignment: \(\sum_{t,j}x_{k,j,t}+\sum_t z_{k,t}\le1\), with a terminal resolution rule to avoid indefinite pending tasks. If deferral is allowed, task stays in queue and consumes no optional radio until assigned. An optional \(\tau_{k,t}\) controls permitted submission delay, but is OFF in v1. Choose one action for one task at a time or an explicit bounded assignment batch; batching is not free of CPU and radio feasibility constraints.

### 3.2 Objective

\[\max_\pi J(\pi)=\mathbb E_\omega\Big[\sum_{k\in\mathcal K(\omega)} v_k C_k^\pi\Big].\tag{P0}\]

When task-value heterogeneity is not independently justified, set \(v_k=1\), report count of deadline-valid completions and CPU cycles completed as distinct metrics, and do not infer their ordering from one another. Rejected or late tasks contribute zero to this objective. Secondary energy or radio usage goes in constraints or Pareto plots, **not arbitrary unreported reward coefficients**.

### 3.3 Physical feasibility constraints

- **Single owner:** unique action, one destination, no duplicate execution unless an explicit replication variant is introduced.
- **Availability:** offloading only to currently reachable candidates as estimated at decision time; no clairvoyant future link availability.
- **Computing:** CPU service capacity, queue order, and nonnegative available cycles. Admit local tasks only when memory/queue caps permit; compute a conservative deadline-feasibility mask if desired.
- **Radio:** all computing packets pass the actual Mode-2 MAC with allowed priorities, resource selection and buffers; packets beyond buffer/deadline limits are dropped.
- **Causality:** computation at helper cannot start before complete input arrival; result transmission cannot begin before computation completion.
- **Optional budgets:** energy and application admission rate, specified separately from baseline safety protection.

### 3.4 Counterfactual safety externality

Let \(e=(m,r)\) denote a safety packet–receiver opportunity within a predeclared evaluation range and required packet deadline. Define

\[Y_e^0=\mathbf1\{\text{baseline safety packet }m\text{ is decoded by receiver }r\text{ on time}\},\]
\[Y_e^\pi=\mathbf1\{\text{same packet/receiver succeeds on time under policy }\pi\}.\]

Define *newly harmful failure* \(H_e^\pi=Y_e^0(1-Y_e^\pi)\) and *avoided failure* \(G_e^\pi=(1-Y_e^0)Y_e^\pi\). Crucially, both may occur across different exposures; net PRR degradation alone hides harm.

For evaluation stratum \(g\) (density, safety-flow type, receiver distance bin, mobility/channel class), define

\[h_g(\pi)=\Pr(H_e^\pi=1\mid e\in g),\qquad f_g(\pi)=\Pr(Y_e^\pi=0\mid e\in g).\]

The core constrained problem is

\[\begin{aligned}
\max_\pi\quad&J(\pi)\\
\mathrm{s.t.}\quad& h_g(\pi)\le\delta_g,&&\forall g\in\mathcal G_\mathrm{eligible},\\
&f_g(\pi)\le\epsilon_g,&&\forall g\in\mathcal G_\mathrm{eligible},\\
&\pi\in\Pi_{\mathrm{causal,feasible}}.
\end{aligned}\tag{P1}\]

The thresholds \(\delta_g,\epsilon_g\) are **design choices/requirements, not sourced 3GPP mandates**. Set them only after inspecting no-offload feasibility. If \(\Pr(Y_e^0=0\mid g)>\epsilon_g\), the absolute requirement may already be infeasible for every policy; flag this rather than falsely calling a controller unsafe. Consider distinct values for emergency vs awareness traffic.

**Exact identity and sufficient bound:**

\[f_g(\pi)=f_g(0)+h_g(\pi)-\Pr(G_e^\pi=1\mid g)\le f_g(0)+h_g(\pi).\tag{1}\]

This requires matched potential outcomes for analysis, **not real-time observation of both worlds**. It is not a new theorem. It shows that limiting harmful flips controls one sufficient component of additional safety failure; it does not automatically prove guaranteed absolute PRR.

### 3.5 Practical constraint accounting and finite-sample validity

The counterfactual \(Y_e^0\) is unavailable online. Collect paired simulator data to train a predictive model \(\hat h_\psi(o,a,g)\) with out-of-sample calibration; optionally predict task feasibility and resulting completed value. The predictor returns an **upper confidence/risk estimate**, not an assumed oracle probability. Use a conservative action filter or admission rule only with a documented bound and support set; otherwise report it as empirical risk control, not certified safety.

Strata matter: global mean harm can hide severe near-field or emergency-message degradation. Evaluate at a coarser hierarchy if per-stratum sample size insufficient; never silently collapse strata after looking at outcomes. Do not treat individual packets sharing one traffic realization as independent statistical repetitions. Use independent scenario seeds as the principal replication unit; within-run packet events are clustered.

For sufficiently independent Bernoulli scenario-level indicators with **zero** observed harmful failures, the one-sided 95% upper bound is \(1-0.05^{1/n}\); at least 149 independent indicators are needed to put this bound below 2%. For actual clustered and nonzero counts, use confidence intervals based on independent scenarios and/or hierarchical/bootstrap methods. Keep final evaluation seeds completely separate from training and calibration.

## 4. Decision process and corresponding MDP / POMDP

### 4.1 Underlying event-driven Markov state

The full simulator state is

\[s_t=(M_t,R_t,L_t,C_t,K_t,S_t,B_t),\]

where \(M_t\)=mobility/topology; \(R_t\)=UE reservation, sensing, counters, and MAC buffers; \(L_t\)=link propagation/channel state; \(C_t\)=CPU queues/remaining service; \(K_t\)=task and result lifecycles; \(S_t\)=safety packet queues and receiver-specific status; \(B_t\)=budgets and elapsed time. Include simulator random-state sufficient statistics to make the implementation Markov if stochastic processes have memory. The application controller does not observe full \(s_t\).

At each application epoch, the simulator evolves using an event-driven transition kernel

\[s_{t+1}\sim P(s_{t+1}\mid s_t,a_t,\zeta_t),\]

where \(\zeta_t\) includes exogenous task/safety arrivals and mobility/channel randomness. **Any counterfactual simulator state is training/evaluation infrastructure, not part of deployable observation.**

### 4.2 Observation available to a vehicle

Recommended normalized observation \(o_{v,t}\):

| Group | Features | Information provenance |
|---|---|---|
| Task | input/output bytes, cycles, value if defined, remaining deadline, age, origin ID embedding | Origin-local request metadata |
| CPU | own queue cycles, estimated local finish time; neighbor advertised capacity/queue age | Local CPU and explicit periodic neighbor status; do not assume hidden true queues |
| Radio | local CBR over defined window, measured S-RSRP/SINR or link quality estimates, recent own delivery/ACK outcomes, configured priority, own SPS counter/occupancy if accessible | Local UE/MAC measurements; no other UE's true future grants |
| Mobility | relative position, range, relative speed, neighbor age and expiry flags | GNSS / received beacons |
| Safety | recent local safety delivery indicators (with known reporting delays), own safety queue occupancy, estimated safety load/risk-bound state | Local data; receiver-specific PRR needs reported feedback, not omniscient observation |
| Constraint history | risk budget counters/rolling estimates derived solely from past observations | Controller-side estimator |

Unknown fields must be masked or represented with missingness/age. Avoid a 'global PRR' feature that the deployed vehicle could never know. Separate **centralized training privileged critic** (optional) from deployable policy observation.

### 4.3 Action space and masks

Preferred v1 **discrete variable-size candidates**:

\[\mathcal A_t=\{\mathrm{reject},\mathrm{defer},\mathrm{local}\}\cup\{\mathrm{offload}(j):j\in\mathcal H_t\},\]

plus deterministic masks: expired tasks, unreachable helpers, full destination queue (if known), disallowed class, or prohibited task forwarding. If exact queue knowledge is missing, use advertised conservative capacity estimates rather than unauthorized access. Candidate order must be permutation-consistent (sort by stable task/helper IDs or use permutation-invariant encoders). Include a hard cap on pending tasks and neighbors for fixed input implementations; record overflows and decisions, and test scaling.

One decision **per currently presented task** is simplest. Task-selection as a separate agent action is an extension; avoid selecting different policies through inconsistent task presentation. v1 presentation rule: ready tasks sorted by earliest absolute deadline, deterministic ties.

### 4.4 Reward and cost signals

Use **event-based terminal task reward** \(r_t=\sum_{k\mathrm{~completed~at~}t} v_k\), no reward for admission. Additional task latency/bytes are reported metrics or explicit Pareto constraints. Optional latency shaping must be disabled in the canonical scientific comparison to avoid changing the objective.

**Training costs** cannot require observing \(Y^0\) at deployment. Two schemes:

- **Simulator paired-label training:** per action/context, estimate harmful-flip probability through synchronized baseline/policy rollouts; use \(\hat c_{g,t}=\hat h_\psi(o_t,a_t,g)\) as policy cost, and evaluate true paired harm offline. Predictor must not be trained on held-out evaluation seeds.
- **Observable proxy training:** safety-message miss, deadline-overrun or elevated CBR, with separate validation against *actual paired harm*. Do not claim a proxy-constrained CMDP enforces \(h_g\le\delta_g\) unless empirically calibrated/derived.

For one-step conditional risk constraints, prefer action masking/filtering based on conservative upper bounds; for long-run service budgets, define

\[J_{c,g}(\pi)=\limsup_{T\to\infty}{1\over T}\mathbb E_\pi\sum_{t=0}^{T-1} c_g(o_t,a_t)\le d_g.\]

This expected average-risk CMDP is **a surrogate** to the original true counterfactual safety condition; document the gap. If \(d_g\) represents a probability per safety opportunity, normalize appropriately by exposure count rather than application epochs.

### 4.5 POMDP, multi-agent and credit assignment

Local observations create a **constrained POMDP** even if the full simulator is Markov. Several independently acting vehicles make the natural network model a **constrained partially observable Markov game**. For implementability, start with **one trainable controller per origin, with other nodes using fixed admission policies**; then introduce shared-parameter distributed agents with CTDE. A single global centralized scheduler is **not** a Mode-2 requirement and should not be assumed for execution.

Delayed credit: task rewards appear on completion, potentially several epochs after admission. Log initiating decision IDs and terminal outcomes. Risk costs are delayed/estimated; train a separate cost critic or calibrated estimator and distinguish lookahead offline oracle from online policy.

### 4.6 Episode, reset and termination

Episode = a complete traffic trace segment of length \(T_{ep}\) plus a drain period **only for events whose original deadline falls in the observed horizon**. Reset mobility, MAC queues/reservations, CPU queues, safety flow generators, task arrivals and RNG streams independently at each seed. Warm up radio state before scoring, but include warmup in matched no-offload world; no learning on final evaluation seeds. Terminate on horizon after closing task outcomes; count expired work exactly once.

## 5. Safe-DRL solution alternatives — no chosen architecture

| Family | Action support | Strengths | Limitations / verification |
|---|---|---|---|
| Discrete DQN / Double-DQN / distributional DQN | Limited candidates | Strong low-complexity baseline, off-policy data efficiency | Unconstrained by default; distributional return is not counterfactual risk certification |
| Discrete SAC with Lagrangian costs | Discrete candidates | Entropy-regularized exploration and replay | Multiplier instability, expected costs ≠ strict guarantees |
| PPO / A2C with primal–dual penalties | Discrete or mixed | Straightforward on-policy masked actions | Sample inefficiency; penalty tuning does not certify safety |
| CPO / trust-region constrained actor–critic | Primarily policy-gradient | Explicit expected-cost updates | Approximation error and sample limits; rarely guarantees per-message protection |
| CVaR-aware distributional policy learning | Discrete/mixed | Controls upper-tail cumulative risk where justified | Tail-risk calibration and sample complexity; CVaR is not automatically the same as harm probability |
| Offline conservative RL | Discrete candidates | Avoids unsafe online exploration using logged simulator experience | Coverage/support mismatch, extrapolation risk |
| Hierarchical/structured learning (rank admissible actions + risk filter) | Variable candidate sets | Keeps safety admission independent of policy family; efficient | Risk filter is only as valid as its calibration and causal model |
| Multi-agent parameter-sharing with CTDE | Distributed local choices | Scalability and local execution | Nonstationarity, communication costs, coordination and credit assignment |
| Model-predictive control / rolling-horizon approximate dynamic programming | Discrete candidates | Transparent competing **non-DRL** solution | Compute-time and model-mismatch costs |

**Selection protocol:** implement 3–4 representative families rather than all. Benchmark against nonlearning methods first. Choose an architecture on held-out development seeds based on completion–harm frontier, safety reliability, data efficiency, wall-clock inference latency, and robustness. A 'safe' RL paper must not assert hard guarantees merely because a constraint appears in its reward or Lagrangian.

### 5.1 Proposed modular algorithm (independent of policy family)

```
On task presentation at origin vehicle v at epoch t:
    obs = build_local_observation(v, task, historical_indicators)
    candidates = feasible_local_reject_defer_helper_actions(obs)
    for a in candidates:
        ub[a] = risk_model.upper_bound(obs, a, stratum, confidence_rule)
        optionally compute conservative remaining deadline feasibility
    allowed = {a in candidates: ub[a] <= current_budget[stratum] and feasible(a)}
    if allowed is empty:
        choose local or defer/reject according to deterministic safe fallback
    else:
        choose a = policy_family.select(obs, allowed)
    execute only application-level admit/placement choice
    pass resulting input/result packets to unmodified Mode-2 MAC
    log decision, observation provenance, mask, risk bound and eventual outcome
```

**Critical:** The instantaneous \(\hat h(o,a)\) is not a proof that the cumulative risk budget holds; aggregation and calibration must be specified. The fallback itself must be validated, particularly because local execution can compete for origin CPU with safety applications if co-hosted.

## 6. Risk estimator and counterfactual labeling pipeline

### 6.1 Data generation without leakage

Collect \(\mathcal D_{risk}=\{(o_t,a_t,g,\hat H_{t:t+W},\text{metadata})\}\) with exploration policies restricted to simulation, paired no-offload baseline rollouts, and multiple independent seeds. Define a fixed causal exposure window \(W\) in milliseconds during which a decision's induced packets can change safety outcomes. **Per-decision attribution is not unique when multiple tasks overlap**; start with a *policy-level total harm* estimator, and use isolated interventions or randomized incremental task additions for marginal labeling. Do not pretend every failed packet has one uniquely responsible task.

Train predictive harm by policy/context windows, with grouping by scenario seed and traffic regime. Split **by complete traffic scenarios and random seeds**, not packet rows. Consider logistic/calibrated gradient-boosted trees or small neural classifiers as alternatives. Use held-out calibration (isotonic, temperature, conformal where exchangeability is justified) and evaluate conditional coverage by stratum. A calibrated global probability alone does not guarantee upper confidence bounds under distribution shift.

### 6.2 Baseline-free online observability

At runtime the controller sees its own local radio and compute metrics, not the baseline outcome. Use a predictor and risk-bound uncertainty from training; no action depends on simulator-only counterfactual labels or global hidden future reservations. Add an explicit OOD/unsupported-state detector: if risk estimate lacks sufficient support, block optional offloading (conservative fallback) and log how frequently this occurs.

### 6.3 A critical estimand distinction

- **Absolute failure** \(f_g(\pi)\): fraction of all eligible safety packet–receiver opportunities that fail in policy world.
- **Harmful flip** \(h_g(\pi)\): fraction that **succeed in baseline and fail in policy**.
- **Net degradation** \(f_g(\pi)-f_g(0)=h_g-g_g\): allows improvements in some opportunities to offset harmful losses in others.

Report all three. Compare with **safety-priority-only** and **zero-offload** baselines, and verify paired keys unambiguously.

## 7. Simulation protocol and data schema

### 7.1 Staged engineering plan, with hard gates

**Phase G0: verify MAC fidelity.** Confirm Mode-2 feature support: sensing horizon, selection-window procedure, SPS reservations, candidate exclusion, PSCCH/PSSCH, HARQ, priority/preemption, configured pools, PHY interference/BLER, half-duplex, logging. Mark each as native, extension, unsupported. Document deviations from specified 3GPP release. **Do not claim standards fidelity for an absent mechanism.**

**Phase G1: zero-computing control.** Reproduce safety PRR versus distance, CBR, packet inter-reception times and loss mechanisms in at least two density regimes; test sensitivity to payload, RRI, MCS and resource selection. Record published reference curves where comparable (not necessarily exact due to different configurations). Confirm deterministic reproducibility of exogenous trace inputs.

**Phase G2: computation and accounting.** Validate input–compute–result causal chain; local-only and offload-only unit cases; deadlines, CPU queue, failures, byte accounting, application-level radio overhead. Test with a single helper and no competing safety traffic, then add baseline safety traffic.

**Phase G3: externality existence.** Compare no-offload, unrestricted offload, safety-priority-only, and CBR threshold with paired seeds and actual packet-level outcomes. If harm cannot be measured or H1 fails robustly, stop before DRL.

**Phase G4: oracle and heuristic controls.** Solve small finite instances with exact enumeration/oracle where possible; compute a full-information clairvoyant bound **only as an upper bound**. Implement nonlearning risk-aware greedy and model-predictive admission, including runtime.

**Phase G5: risk estimation.** Train/calibrate \(\hat h\) and compare predicted risks against actual paired outcomes out of sample. Report ECE/Brier and upper-bound violations by regime; no deployment safety claim if poorly calibrated.

**Phase G6: compare DRL alternatives.** Fixed training budget, independent training seeds, equal observation/action availability, identical risk shields where used, matched paired evaluation seeds. Tune algorithms only on development runs.

**Phase G7: stress and final assessment.** Test unseen density/channel/CPU heterogeneity, message mixes, priority settings, different task result-to-input ratios, safety PDB, feedback delays, hidden nodes and OOD. Freeze code and final test seeds before final comparison.

### 7.2 Default scenario grid — a pre-registration proposal, not paper-derived constants

Define three density regimes after an initial capacity characterization (light, transition, congested); do not choose densities just to manufacture a crossover. Sweep task arrival load, input and result bytes, CPU heterogeneity, deadline tightness, safety packet interval, priority treatment and resource-pool size. Evaluate both a moving highway and an urban LOS/NLOS scenario if selected simulator supports them. Fix consistent PHY and traffic parameters within matched comparisons. All numeric values belong in versioned YAML, with provenance (standard, paper or engineering assumption).

### 7.3 Baseline matrix

- **B0** no optional V2V offloading, local computing allowed.
- **B1** unrestricted feasible offloading (CPU/endpoint scheduling only, not malicious MAC bypass).
- **B2** EDF with local/nearest/lightest-load helper selection.
- **B3** admission based on CBR threshold plus helper-selection heuristic; tune threshold on development set.
- **B4** Mode-2 safety priority/preemption with B1/B2 application scheduler (where implemented).
- **B5** model-based risk-aware heuristic with same observations and same risk estimator as DRL.
- **B6** standard unconstrained DRL with identical task observation/action interface.
- **B7** constrained DRL variants and calibrated shield alternatives.
- **B8 (if feasible)** adapted MAPPO-SPS / published cross-layer learning, matched to *permissible* actions and radio features. Document incompatible control assumptions instead of silently changing competitor.
- **Oracle** small-state exact policy and optional clairvoyant upper bound, clearly labelled as non-deployable.

### 7.4 Primary measurements and analyses

| Family | Mandatory metrics |
|---|---|
| Safety | paired harmful-flip probability \(h_g\); absolute failures \(f_g\); baseline failures \(f_g(0)\); net difference; PRR vs range; deadline-valid safety PDR; packet inter-reception time; violation per stratum |
| Computing | task count and task value completed by deadline; CPU cycles completed by deadline; admission/rejection; turnaround percentiles; helper distribution; input/result delivery success |
| Radio | CBR, occupied subchannels, SPS reselection frequency, collision/PHY loss counts, task/safety byte loads, MCS, BLER, half-duplex events |
| Learning/online | training reward and cost separately; constraint multipliers and risk calibration; inference decision time and memory; shield block fraction; OOD block fraction |
| Statistical | paired per-seed differences; confidence intervals by regime and method; median/worst-seed outcomes; training-seed dispersion; runtime–performance frontier |

Require safety feasibility **at each prespecified regime/stratum**, not merely the grand mean. Plot completion versus harmful risk Pareto curves with confidence intervals, including baseline points. Report infeasible traffic regimes rather than excluding them.

### 7.5 Structured logging contract

1. `scenario.csv`: `scenario_id,seed,regime,trace_id,sim_version,release,config_hash,pool_id,policy_id`.
2. `safety_packets.csv`: `scenario_id,world,flow_id,seq,tx_id,rx_id,created_ms,deadline_ms,decoded_ms,success_on_time,bytes,priority,range_m,loss_reason`.
3. `tasks.csv`: `scenario_id,task_id,origin_id,release_ms,deadline_ms,in_bytes,out_bytes,cycles,value,admit_time_ms,destination,decision_id,input_delivered_ms,cpu_start_ms,cpu_finish_ms,result_delivered_ms,completed_on_time,failure_reason`.
4. `decisions.csv`: `scenario_id,decision_id,time_ms,vehicle_id,task_id,observation_schema_version,action_mask,chosen_action,risk_estimate,risk_upper,policy_version,decision_runtime_ms`.
5. `radio_events.csv`: `scenario_id,time_ms,ue_id,packet_id,traffic_class,priority,subchannel_ids,reservation_counter,reselect_reason,mcs,tx_attempt,sinr_db,decoded,receiver_id`.
6. `paired_outcomes.csv` **derived after simulation**: join `safety_packets.csv` in worlds 0/π on `(scenario_id,flow_id,seq,rx_id)`; fields `baseline_success,policy_success,harmful_flip,avoided_failure,stratum`.

Use stable deterministic IDs and explicit units. Missing packet record in treatment must be resolved to failure/expiration with justified generation alignment; do not drop unmatched records from harm denominator.

### 7.6 Software module interfaces

```
ScenarioFactory(seed, regime, config) -> exogenous_trace
Mode2Adapter.reset(config, trace); .inject(packet); .run_until(t); .query_local(ue)
ComputeEngine.submit_local(task,ue); .submit_remote_on_input_delivery(task,helper)
TaskManager.on_release(task); .on_packet_delivery(packet); .on_cpu_finish(job)
SafetyTrafficSource.emit_independently(trace)
ObservationBuilder.build(ue,task,history) -> local_obs, legal_mask
AdmissionPolicy.act(local_obs, legal_mask) -> action
RiskModel.upper_bound(local_obs,action,stratum) -> risk_ub, validity
EventLogger.record(...)
PairedEvaluator.align_and_score(baseline_world, policy_world)
Trainer.fit(environment, policy_family, constraints, split)
Evaluator.run_frozen(policy, heldout_scenarios) -> per_seed_report
```

These interfaces specify *responsibilities*, not a claim that any selected simulator has all these hooks. Add adapters only after G0.

### 7.7 Acceptance tests (must be automated)

1. **No compute packets:** reference and treatment safety logs are bitwise identical under deterministic replay (or numerically identical per designated event keys when RNG streams differ); harmful flip \(=0\).
2. **Single task, perfect radio:** local execution and remote input → CPU → result paths obey deadlines exactly.
3. **Intentional radio failure:** input failure prevents helper execution; result failure prevents task completion.
4. **Expiry:** late result yields zero deadline-valid completion; no double counting after timeout.
5. **CPU capacity:** no task executes on two processors simultaneously; service never exceeds rate/cycle budget.
6. **Paired-key matching:** all eligible safety packet–receiver exposures appear in both worlds.
7. **Safety harm accounting:** `harmful_flip = baseline_success AND NOT policy_success`, and identity (1) holds numerically per stratum.
8. **Observation integrity:** no future, other-world or hidden simulator state enters deployable observations.
9. **Action integrity:** application policy never directly writes a forbidden Mode-2 reservation state.
10. **Reproducibility:** identical seed/config/version yields identical outputs within simulator determinism limits.
11. **Risk support:** uncalibrated/OOD states trigger documented fallback, not invented risk confidence.
12. **Statistical unit:** final inference uses independent scenario seeds and prespecified safety strata, not treating correlated packet rows as iid trials.

## 8. Expected paper contribution — contingent, not promised

1. **System-level finding:** quantify incremental safety-message harm from admitted V2V computation traffic under realistic distributed Mode-2 resource selection, beyond mean PRR and CBR.
2. **Optimization formulation:** protect baseline safety service using an explicit matched-world harmful-flip constraint and an absolute reliability constraint while maximizing deadline-valid computation.
3. **Deployable control:** learn/estimate incremental harm from *local* observations; separate estimator calibration and admission constraint from the choice of DRL architecture.
4. **Reproducible evidence:** show whether the resulting method expands the best safety–computation Pareto frontier relative to tuned nonlearning and existing constrained-learning alternatives, including unseen regimes.

**The paper is NOT substantiated if:** the harmful-flip effect is negligible, the estimator cannot calibrate or generalize, safety controls already match performance, or the algorithm merely exchanges risk for throughput without satisfying the prespecified thresholds. Publishable novelty would require empirical and/or theoretical results **in addition** to this design.

## 9. Ordered implementation checklist

1. Freeze radio release, simulator version and supported MAC features; record audited behavior.
2. Implement independent safety traffic and no-offload world; validate packet-level reference results.
3. Add CPU event queues and complete input/result offloading flow; unit-test deadlines and accounting.
4. Implement matched-world runs and harmful-flip join; prove baseline zero-harm sanity tests.
5. Run unrestricted and priority-only policies to establish existence and magnitude of the externality.
6. Build and tune EDF, CBR-threshold and risk-aware nonlearning controllers.
7. Train/calibrate risk predictor on training/dev paired simulations; test subgroup validity/OOD.
8. Compare at least three different policy families under exactly the same action, observations and safety estimator.
9. Freeze final model, run held-out paired seeds and compute confidence intervals and Pareto frontiers.
10. Proceed to manuscript only if both reliability constraints and meaningful gains survive matched evaluation.

## 10. Sources and provenance

**User-supplied full PDFs** (exact file names available in conversation):
- Dong et al. DOI 10.1109/TITS.2025.3542365 (IEEE T-ITS 2025).
- Chan et al. DOI 10.1109/TITS.2026.3655760 (IEEE T-ITS 2026).
- Jia et al. DOI 10.1109/TVT.2025.3621636 (IEEE TVT 2026).
- Todisco et al. arXiv:2606.26746v1 (2026), *5G NR-V2X Scheduling Approaches for CPM Variable Size Traffic*.

**Additional verified external references:**
- Ali et al., IEEE Access 2021, DOI 10.1109/ACCESS.2021.3090855, https://pmc.ncbi.nlm.nih.gov/articles/PMC10350958/.
- 3GPP TR 38.885 https://www.atis.org/wp-content/uploads/3gpp-documents/Rel16/ATIS.3GPP.38.885.V1600.pdf.
- ETSI TS 138 321, Release 17 MAC, https://www.etsi.org/deliver/etsi_ts/138300_138399/138321/17.13.00_60/ts_138321v171300p.pdf.
- Indulekha and Venkatesh (2026), arXiv:2605.24972, https://arxiv.org/abs/2605.24972.
- *Modeling and Analysis of Priority-Aware Resource Preemption in 5G NR-V2X Mode 2 Sidelink Communications* (IEEE INFINITYS 2026), DOI 10.1109/INFINITYS69204.2026.11666334.
- Zhang et al., CVaR-Constrained Policy Optimization (IEEE TNNLS 2025), DOI 10.1109/TNNLS.2023.3331304.
- *A Survey of Constraint Formulations in Safe Reinforcement Learning*, IJCAI 2024, https://www.ijcai.org/proceedings/2024/0913.pdf.
- Shen et al., *Communication-Aware Distributed Control for Cooperative Ramp Merging in Vehicular Networks*, IEEE TMC Early Access 2026, DOI 10.1109/TMC.2026.3739758, distinguishes network-aware safety control from task-admission protection.

**Provenance rule:** all threshold choices, observation sets, workloads, alternative RL architectures and evaluation grids proposed here are *new project design decisions* and must be checked in the target simulator. They are not attributed to any of the papers above.

---
## Appendix A (17 November 2026): literature-grounded numerical configuration

**Supersedes** unresolved values in the original `TITS_Mode2_Experiment_Config_Template.yaml`. Use `TITS_Mode2_Experiment_Config_Literature_Grounded.yaml` and `TITS_Mode2_Literature_Parameter_Register.md` for an explicitly sourced starting profile and provenance of every non-source design decision.

**Primary physical configuration: Todisco et al. (2026)** — 20 MHz / 30 kHz SCS / 0.5-ms slots / 5 subchannels of 10 RB each / 23 dBm / 3 dBi gain / 9 dB noise figure / MCS 12 / keep probability 0.8 / no retransmissions / 8-km six-lane highway / densities 50, 150 veh/km / mean speeds 110, 70 km/h with SD 11, 7 km/h. CPM generation is 100 ms and 500 ms at the two densities, respectively. Keep the varying-period confound explicit. Details and source: uploaded Todisco VNC 2026 preprint, §§II and V.

**Safety traffic: Ali et al. (2021)** — 300 bytes/100 ms; same profile also reported by Yu and Mun (2026). The 100-ms safety packet deadline and proposed 1% incremental harmful-flip limit are *study constraints*, not values established by those authors. Yu and Mun use PDR >= 90% as a reporting benchmark, which may be used as the absolute baseline requirement, but does not prove a universal regulatory reliability threshold.

**Computing:** Dong et al., T-ITS 2025, §VI-A: 12-kbit sensing subregion, 200 cycles/bit, and vehicle resources 500–2000 Mcycles/s. For heavy-workload sensitivity, Liu et al., Sensors 2025, Table 1: 0.5–11 Mbit inputs, 1–4.5 Gcycles/Mbit and 0.5–3 s deadlines. These profiles must remain separate and cannot be treated as one empirically measured joint workload.

**Critical implementation validations:** The chosen WiLabV2XSim release/commit, sensing windows, selection windows, HARQ features, priority mapping, and maximum supported transport-block configurations must be verified from the code and the standard implementation. Large task inputs must actually be segmented and delivered; the baseline model's 1180-byte TB must not be used as a magical unlimited payload channel. See the parameter register for unit consistency, provenance, caveats, and the OMNeT++ alternative.
