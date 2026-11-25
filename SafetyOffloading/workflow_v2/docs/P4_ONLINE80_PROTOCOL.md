# Efficient WiLab Dueling-DDQN convergence protocol

## Fixed training budget

Each independent run has 16 main blocks. Each main block performs five physical
WiLab training rollouts and then 80 Python replay-learning updates. The main
budget is therefore 80 rollouts and 1,280 optimizer updates. Eight additional
physical rollouts and 128 optimizer updates form the predeclared continuation,
for a total of 88 training simulations.

`rollout` means one physical WiLab simulation. `optimizer update` means one
Python learning pass. Neither an update nor replay-derived reward is a physical
rollout or validation result.

## Validation schedule

Only the block-end checkpoint is saved for each of the 16 main blocks. It is
evaluated greedily on common fixed seed 1301. At blocks 4, 8, 12, and 16, the
same checkpoint is also evaluated on seeds 1302–1305; seed 1301 is reused and
is never repeated. Thus every run has 16 common-seed validations and 16
additional milestone validations: 32 total validation simulations.

The continuation produces no automatic validation. A supplementary evaluation
is allowed only to resolve a documented ambiguous final plateau. Held-out test
seeds are separate from all training and validation seeds.

## Evidence retained without extra WiLab work

Every optimizer update records mean absolute TD error, TD-error quantiles,
fixed-causal-observation Q/value/advantage quantiles, Q drift, and greedy
action agreement with the preceding update. Every physical rollout saves its
unshaped utility and trailing ten-rollout mean. These diagnostics use saved
replay and cached causal observations; they launch no physical simulation.

## Convergence interpretation

The common-seed trajectory provides the dense block-level learning trajectory.
The four five-seed milestones corroborate, but do not replace, its pattern.

1. **Progressive learning:** compare early and later checkpoints in blocks
   1–8; record whether the later common-seed mean improves by 10% and whether
   three checkpoint-to-checkpoint improvements occur. Corroborate with blocks
   4 and 8 five-seed means. A single seed alone is not a statistical claim.
2. **Policy refinement:** inspect blocks 8–12 for maintained or improved
   utility and shrinking trajectory changes, with TD-error variability and
   Q-drift stable or declining.
3. **Stable convergence:** inspect blocks 12–16 for no material late decline;
   require the block-16 five-seed mean to retain at least 95% of the best
   preceding milestone mean, and late greedy-action agreement of at least 90%.

Five independent runs use the same schedule and validation seeds, while their
network initialization, rollout seeds, and exploration seeds differ. Baseline
comparisons are not convergence evidence and occur only after this gate.

## Physical budget

| Activity | Simulations per run |
| --- | ---: |
| Training rollouts (including continuation) | 88 |
| Common-seed block-end validation | 16 |
| Additional milestone validation | 16 |
| **Total** | **120** |
