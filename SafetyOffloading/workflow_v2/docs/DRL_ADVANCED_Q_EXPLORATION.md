# Exploration: advanced Q-learning components for workflow offloading

**Status:** analysis only (while causal BC+DDQN run continues). No architecture swap authorized by this note alone.  
**Context:** discrete 5-action MDP (`LOCAL/H2/H3/HOLD/DROP`), causal 16-D obs, sparse prioritized utility, contact stranding, optional `mode2_event` radio.

Rainbow (Hessel et al., 2018) bundles six DQN improvements. Farimani et al. (ESWA 2024) apply full Rainbow to deadline-aware vehicular offloading and report better delay/energy and stabler learning vs DQN/DDQN/DRQN. QR-style distributional methods also appear in MEC offloading (e.g. QRMP-DQN, TVT 2024).

## Fit to *this* problem

| Component | What it changes | Applicability here | Expected merit | Cost / risk |
| --- | --- | --- | --- | --- |
| **Dueling** (already in use) | Separate state-value vs action-advantage | High — many states share value; actions differ mainly when `DROP` vs place | Stabilizes ranking among similar obs; good with masked actions | Low; already adopted |
| **Double DQN** (already in use) | Decouple select/eval for bootstrap target | High — overestimation on rare high-utility successes | Less optimistic Q on `DROP`/`H2` | Low |
| **Multi-step returns** (n-step) | Bootstrap over n transitions | **High** — early `DROP` pays off many steps later when late highs complete | Directly targets credit assignment for *earlier admission* | Need reward scale care; n≈3–5 typical |
| **Prioritized replay (PER)** | Sample high-TD-error transitions more | **High** — high-value success/expire and early-DROP transitions are rare | Keeps learning pressure on the oracle-gap events | Extra buffer complexity; β annealing |
| **Noisy nets** | Learned stochastic exploration | **High** — `DROP` is easy to under-explore with ε-greedy | Can replace hand DROP-bias / reduce ε schedule tuning | Noisy layers break naive BC CE slightly; fine-tune after BC |
| **Distributional (C51 / QR-DQN)** | Learn return *distribution* not mean | **High** — returns are multi-modal (strand fail vs high-value hit); utility spikes of 12 vs 1 | Better risk-sensitive choice under contact uncertainty; stabler under Mode-2 delay variance | More parameters; atom/quantile tuning |
| **Full Rainbow** | All of the above | Justified for Mode-2 / no-BC ablation | Literature precedent in VEC offloading | Heavier; evaluate components ablated |

## What matters most for *our* remaining gaps

1. **Without BC (ablation / scientific claim that DRL discovers admission):** Noisy nets + PER + n-step are the highest-leverage trio — they attack under-exploration of `DROP` and delayed utility credit. That is closer to “learning earlier admission” than mean DQN alone.  
2. **With BC (current stack):** Distributional Q and n-step help *retain* and *refine* the oracle policy under stochastic contact/radio, rather than regressing during RL fine-tune (current run already showed post-BC eval dip risk).  
3. **Mode-2-faithful radio:** Return variance rises (fragmentation, opportunities, stranding). Distributional RL is the component most specifically motivated by that transition model — aligned with Farimani’s rationale for Rainbow in dynamic vehicular settings.  
4. **Dueling + Double:** Keep; low cost, already consistent with Rainbow subset.

## Recommended adoption order (if/when implemented)

| Priority | Component | Why first |
| --- | ---: | --- |
| 1 | n-step (n=3) | Cheap; matches delayed DROP→utility causality |
| 2 | Prioritized replay | Amplifies rare high-value / DROP transitions |
| 3 | Noisy nets (replace ε for RL phase) | Principled exploration of `DROP` without manual 70% bias |
| 4 | QR-DQN or C51 | For Mode-2 variance and risk over mean utility |
| 5 | Full Rainbow only if ablations show complementary gains | Avoid bundling before evidence |

## What would *not* be a good reason to switch

- Chasing Rainbow because it is SOTA on Atari.  
- Expecting architecture alone to create a utility gap when the MDP leaves none (prior flat-vs-VG case).  
- Replacing the BC warm-start story with “Rainbow will find DROP” without measuring a no-BC ablation.

## Relation to the live experiment

Current run: **oracle BC → Dueling Double DQN + DROP-biased ε + early-DROP rewards**.  
Advanced Q components are **orthogonal upgrades** for (a) closing the no-BC discovery gap, (b) stabilizing fine-tune toward oracle utility on holdout, (c) Mode-2 transfer — not a substitute for the admission-control problem design.

## Bottom line

For this discrete admission/placement MDP, **n-step + PER + noisy nets** are the most justified next Q-learning upgrades; **distributional RL** becomes first-class once Mode-2 event variance dominates. Full Rainbow is applicable and literature-supported in VEC offloading, but should enter as an ablated stack, not a blind swap.
