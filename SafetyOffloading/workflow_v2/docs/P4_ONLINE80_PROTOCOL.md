# P4 online training protocol — frozen before execution

**Status:** finalized; do not start until explicitly authorized.

## Architecture identity gate

This protocol requires a **Dueling Double DQN** convergence claim. Before any
run is started, the saved model and trainer must be audited for separate
state-value and action-advantage streams. The current 15 -> 96 -> 1 candidate
scorer is non-dueling; it must not be called Dueling DDQN or used as evidence
for that claim. An architecture change requires separate explicit approval.

## Terms

| Term | Meaning | Produces a physical utility value? |
| --- | --- | --- |
| Training rollout | One complete WiLab simulation, from initialization to 5.5 simulated seconds. | Yes, exactly one. |
| Training epoch | One replay-learning pass in Python after rollouts have been collected. | No. |
| Optimizer step | One neural-network parameter update on one sampled replay transition. | No. |
| Policy-update block | Five training rollouts followed by 80 training epochs. | Five rollout utilities. |
| Probe | A frozen policy evaluated on the fixed five-seed WiLab suite. | Yes, but it is evaluation, not training reward. |

Use **rollout** for a physical training simulation. Do not call an epoch an
episode or a rollout.

## Fixed schedule

The run contains 16 policy-update blocks.

For block `b = 1..16`:

1. Run five training rollouts, numbered `5(b-1)+1` through `5b`, with seed
   `3000 + rollout_number`.
2. Save one positive physical utility value from each rollout:
   `summary.total_value`.
3. Rebuild replay from every completed training-rollout trace available at
   that point.
4. Starting from the prior block's policy (BC policy for block 1), train for
   80 training epochs.
5. Save every epoch's model checkpoint and its full training row.
6. At epochs 5, 10, 15, ..., 80, evaluate that epoch checkpoint on the same
   five fixed probe seeds. Save every probe output.
7. Promote only the epoch-80 policy to become the input policy for the next
   block. Checkpoint selection is never performed from a probe result.

Totals when complete:

- 80 training rollouts;
- 1,280 training epochs;
- 16 policy-update blocks;
- 1,280 physical probe simulations (16 blocks × 16 probe epochs × 5 seeds).

## Reward and convergence records

The training-reward series has exactly 80 entries: one `summary.total_value`
per training rollout. The only rolling-reward plot is its trailing
10-rollout mean.

The probe series is separate. It contains mean physical utility across the
five fixed evaluation seeds after every fifth training epoch. It is used for
convergence analysis only; it must not be called reward or merged into the
80-rollout reward series.

The following must never appear in the 80-rollout reward plot: TD loss,
replay-sample reward, expiry penalty, optimizer-step loss, or probe mean.

## Required preserved artifacts

- Each training rollout: complete WiLab output folder, config, events,
  decision trace, raw fresh trace, and summary.
- Rollout ledger: one ordered raw utility row per training rollout.
- Each policy-update block: exact input policy, replay archive, replay
  manifest, all 80 epoch checkpoints, all 80 epoch training rows, and final
  output policy.
- Each probe: complete WiLab output folder and summary, indexed by update
  block, epoch checkpoint, and fixed probe seed.

No artifact is deleted, overwritten, or relabeled during the run.

## Convergence evidence required before any final comparison

Baseline comparisons are not convergence evidence. They are deferred until
after the following evidence has been produced from the learning process.

### Per-rollout and per-epoch records

For every physical training rollout, retain its utility, completion count,
expiry count, action histogram, and full raw WiLab artifacts. For every
training epoch, retain TD-error statistics, fixed-state Q-value quantiles,
value/advantage statistics for a Dueling model, policy-action agreement on a
fixed state set, model checkpoint, and replay manifest.

### Fixed checkpoint probes

Every fifth training epoch, evaluate the frozen checkpoint on the same five
probe seeds. Save individual-seed returns, their mean, standard deviation,
safety metrics, action histogram, and Q-value quantiles. These are the
checkpoint learning curve; do not mix them with rollout rewards.

### Three required learning stages

1. **Progressive learning:** the trailing ten-rollout mean and fixed-probe
   mean show sustained improvement over the first two policy-update blocks.
   The improvement must persist in all independent runs, not appear in one
   isolated seed.
2. **Policy refinement:** subsequent checkpoint gains remain positive but
   shrink. At least three consecutive fifth-epoch checkpoints must show
   gains no larger than 5% of the current best probe mean while retaining the
   learned action distribution and satisfying the safety gate.
3. **Stable convergence:** at least four consecutive fifth-epoch checkpoints
   remain within +/-5% of their mean probe utility; fixed-state greedy-action
   agreement is at least 98%; TD-error and Q-value quantiles have no material
   trend or explosion; and no safety metric deteriorates.

### Independent-run and continued-learning tests

Run three independent training trajectories with distinct initialization and
rollout seeds under this identical protocol. After each reaches its candidate
plateau, continue its training for 40 additional epochs and collect one
additional five-rollout exploration block. A trajectory is rejected as
premature stabilization if either extension improves the fixed-probe mean by
more than 5%, changes more than 2% of fixed-state greedy actions, or produces
a materially higher-quality policy in another independent run.

Only after all three trajectories pass these tests may a final policy be
frozen. DropLow, BC-only, and external-baseline comparisons occur after this
gate and establish relative performance only.
