# Mandatory WiLab Dueling-DDQN convergence experiment

## Preserved training budget

Each of five independent runs retains 88 full packet-level WiLab training
rollouts and 1,280 optimizer updates. Sixteen main blocks contain five
rollouts and 80 updates each. The predeclared continuation contains eight
rollouts and 128 updates. Episode duration, Mode-2 radio model, replay, and
exploration configuration are unchanged.

Every main-block end policy is checkpointed. Each optimizer update records TD
error, Q/value/advantage statistics, Q drift, greedy-action agreement on cached
causal observations, and replay-derived training diagnostics. Each rollout
records unshaped physical utility and its trailing-ten-rollout mean. These
records launch no extra WiLab simulation.

## Matched validation milestones

Full packet-level fixed validation is performed only at main blocks 4, 8, 12,
and 16, corresponding to optimizer updates 320, 640, 960, and 1,280. At every
milestone, the identical five matched validation seeds 1301–1305 are evaluated
greedily. They are disjoint from training and held-out testing seeds.

Thus every run uses 88 training plus 20 validation simulations: **108 total**.
The final held-out test is separate. Supplementary validation is permitted only
to resolve a documented ambiguous equivalence result; training is never rerun
solely to increase validation replication.

## Convergence evidence

- **Stage I, M1→M2:** target at least 20% matched-scenario mean utility
  improvement; report all five paired changes and the intervening training
  return trajectory. The improvement must not be confined to one scenario.
- **Stage II, M2→M3:** retain at least 95% of M2 mean utility while policy
  changes, TD-error variability, and Q drift stabilize; inspect decisions to
  distinguish better scheduling from increased rejection.
- **Stage III, M3→M4:** test paired utility equivalence against a prespecified
  ±5% M3-mean margin. The 95% paired t confidence interval must lie within the
  margin, final greedy-action agreement must be at least 90%, and TD/Q metrics
  must show no late upward drift or deterioration.

The five independent runs—not optimizer updates—establish reproducibility.
The acceptance target is four of five showing all three stages. Baseline
comparisons are explicitly separate from convergence evidence.

## Runtime and execution

The efficient schedule removes redundant checkpoint simulations without
reducing training experience. The runner logs timing for every rollout,
optimizer block, and validation. Independent runs write isolated roots and are
launched with a bounded two-worker parallel scheduler on this host until
measured throughput supports a higher concurrency level.
