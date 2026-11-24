# Final baseline and ablation specification (LOCKED)

**Status:** Frozen by user directive. Do not deviate unless the user explicitly says so, or grants permission after being asked.

Source: user message dated 2026-11-24. This file is the project copy of that contract.

---

## 1. Literature selection

| Published method | Decision for this MDP |
| --- | --- |
| DPTO, Adhikari et al., IEEE IoTJ 2020 | Basis of one priority-aware deadline/placement baseline |
| EDF–EFT, Mobility- and Deadline-Aware Task Offloading, IEEE VTC 2026 | Scheduling principles folded into the same baseline (not a near-duplicate) |
| DQNP, Uddin et al., IEEE TVT 2025 | **Exclude** — controls/system differ |
| MARINA, IEEE T-ITS 2023 | Related work only — **not** a baseline |

One external literature-grounded competitor: **Priority-EDF–EFT (adapted)**. Foundations are published; the online DAG admission mapping is ours. **Not** an exact reproduction of DPTO or the VTC algorithm.

---

## 2. Main comparison

| ID | Method | Status | Scientific role |
| --- | --- | --- | --- |
| B0 | Local EDF | Existing / straightforward | No-cooperation control |
| B1 | ValueGreedy | Existing, frozen | Historical priority-aware comparator |
| B2 | Always-Drop-Low | Existing, frozen | Strong admission-rule competitor **and** BC teacher |
| B3 | Priority-EDF–EFT (adapted) | **One new implementation** | Literature-grounded priority/deadline/placement competitor |
| P | Causal Dueling DDQN + BC | Frozen | Proposed method |

**Forbidden in the main performance table:** HEFT, DQNP, MARINA, VNS, random-action policies, additional ad hoc heuristics.

B3 is the **only** new baseline implementation.

---

## 3–4. B3 implementation contract (summary)

- Same actions: LOCAL, H2, H3, HOLD, DROP.
- Same **16-dimensional causal observation**, legal mask, diamond-4 structure, fixed public config as DDQN.
- Must **not** use privileged / hidden simulator state (true helper queues, future contact).
- Soft-EDF stage presentation stays with the environment.
- Admission: high-value priority; no invented `n_active >= 4` threshold.
- Placement: minimize estimated finish time \(\widehat F_{v,j}\) from observable queues/CPU/contact/ads.
- Remaining-DAG deadline estimate required (parallel branches sync).
- DROP when no legal deadline-feasible continuation; else environment non-drop fallback.
- HOLD only with concrete reason + existing defer limits/mask.
- Tie-break (fixed): lowest finish → lower comm cost → LOCAL before H2 before H3.

**If** full capacity-displacement needs fields absent from the 16-D obs, use the **reduced** priority-EDF–EFT rule (§4 of user contract) only — do not pull hidden state. Stronger displacement test is **not authorized** unless inputs are proven common.

---

## 5. Ablations (separate experiment)

| Variant | Change | Purpose |
| --- | --- | --- |
| A1: BC-only | Stop after BC; no TD fine-tune | RL contribution |
| A2: Without BC | Random init; keep Dueling DDQN + other training choices | Dependence on demos |
| A3: Without DROP shaping | Remove DROP reward bonuses; keep BC + DDQN | Dependence on admission shaping |
| Full | Existing Causal Dueling DDQN | Reference |

A3: retain action mask and DROP-biased exploration (isolates reward shaping only).  
No-dueling ablation: not required unless the paper claims dueling-specific gain.

---

## 6. Reproducibility

- One shared evaluator; **separate** baseline vs ablation reports.
- Workload: original frozen causal experiment.
- Holdout seeds **300–339**.
- Primary: deadline-valid total utility. Secondary: completions.
- Diagnostics: priority-specific completions/drops/expirations/placement/runtime.
- Stats: paired mean diffs, CIs, wins/ties/losses.
- Separate ablation checkpoints; full checkpoint frozen.
- **Gate before B3 work:** reproduce holdout means ValueGreedy **78.40**, Always-Drop-Low **99.30**, Full **101.80**.
- Log exact observation + legal mask at every decision (verify no privileged leak to planners).

---

## 7. Paper tables

- **Main:** B0, B1, B2, B3, P. Primary conclusion vs the **strongest** competitor, not solely ValueGreedy.
- **Ablation:** A1, A2, A3, Full.
- Existing numbers: ~+29.8% vs ValueGreedy, ~+2.5% vs Always-Drop-Low — teacher comparison indispensable.

---

## 8. Scope lock

One new literature-derived heuristic (B3). Three ablations. One common evaluator. No invented claims of reproducing published algorithms unchanged.
