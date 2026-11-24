# R3 exit — joint-decision relevance pilot (**frozen**)

**Status:** **CLOSED / FROZEN** — 2026-11-23  
**Pilot root:** `workflow_v2/outputs/r3_pilot_20261123_220750` (+ abstract twin `outputs/r3_abstract_fix2`)  
**Freeze (unchanged after start):** diamond-4, 36 Mcycles, 500 ms, ρ=0.80 BG, arrival `{0.10}` s, sim 1.0 s, 2 nearest helpers, seeds `{1,2,3}`  
**Corrective iteration:** config/schema/meta/MBest/release/abstract time-clamp — **no workload retuning**.

**Scientific result:** negative for the proposed **joint-scheduling completion advantage**; positive for **implementation feasibility**, **queue-aware policy superiority over the tested greedy**, and **packet-level Mode-2 latency sensitivity**.

**Do not proceed to R4 or DRL from this pilot.** Next authorized step (if any): one-time **R3-M** model-validity gate — see `docs/R3M_GATE.md`.

## Results (WiLab Mode-2, deadline-valid completions / 3 seeds)

| Method | Success | Expired | Radio data frags (typ.) | Remote stages done |
| --- | ---: | ---: | ---: | ---: |
| B0 Local EDF | **3/3** | 0 | 0 | 0 |
| B1 Independent greedy | **0/3** | 3 | 0 | 0 |
| B2 Queue-aware | **3/3** | 0 | 0 | 0 |
| B3 Mobility-aware | **3/3** | 0 | 0 | 0 |
| B4 Joint (Mode-2) | **3/3** | 0 | **8** | **2** |
| B5 Privileged (seed 1) | 1/1 | 0 | 0 | 0 |
| B4 Joint (abstract twin) | **3/3** | 0 | (scripted) | 2 |

B4 Mode-2 completion ≈ **0.27–0.29 s**; abstract twin ≈ **0.17 s**; B0 local ≈ **0.17 s**.

## Hypothesis ledger

| Hypothesis | Evidence | Decision |
| --- | --- | --- |
| Distributed workflows can execute through Mode-2 | B4 3/3 with remote stages + Mode-2 data fragments | **Established** |
| Queue-aware policy beats tested independent greedy | B1 0/3 vs B2 3/3 | **Retain queue awareness**; see causal caveat below |
| Packet-level Mode-2 affects execution latency | ~280 ms vs ~170 ms abstract | **Established for this pilot** |
| Cooperative execution improves over local EDF | B0 = B4 = 3/3; B0 faster | **Not demonstrated** |
| Joint scheduling improves workflow completion | B4 matches B0/B2/B3 | **Not demonstrated** |
| Mobility-aware decisions add value | B3 matches B2 | **Not demonstrated** |
| Controlled packet release adds value | No isolated release comparison | **Undetermined** |

### Causal caveats (normative)

1. **B1 vs B2** shows superiority of the *tested* queue-aware rule over the *tested* greedy rule. It does **not** isolate “queue observation alone” as the causal factor; other decision-rule differences remain confounded until controlled.
2. The ~100 ms Mode-2 vs abstract gap shows **timing sensitivity**. It does **not** by itself prove Mode-2 changes the *optimal* scheduling decision under this saturated completion metric.

## Why discriminatory power was limited

One workflow per seed. With a 500 ms deadline and local completion ~170 ms, B0 already clears the entire arrival set:

\[
N_{\mathrm{completed}}^{B0} = N_{\mathrm{completed}}^{B4} = N_{\mathrm{arrived}}.
\]

Completion count is saturated; no policy can improve it. This does **not** prove joint scheduling is intrinsically useless — it means this operating point cannot discriminate on the primary metric.

The pilot also did not exercise the intended dynamics: overlapping workflows, backlog accumulation, shared helper scarcity, helper churn during outstanding commitments, or competition among intermediate-data transmissions. Background ρ=0.80 alone does not guarantee workflow contention.

**Issue = workload identifiability**, not a license to retune until B4 wins.

## Stopping rule applied

- **No DRL** from this R3 outcome.
- **No open-ended R3 redesign** to manufacture joint advantage.
- **R4 on hold.** Optional next step: **R3-M** only — one bounded, predeclared model-validity gate (`docs/R3M_GATE.md`).
