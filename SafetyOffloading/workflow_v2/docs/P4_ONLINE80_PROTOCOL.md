# P4 online-80 training protocol

This is a training protocol, not a frozen policy. The policy is updated after
each block of five physical WiLab rollouts.

## Immutable reported metric

One value is recorded per rollout: `summary.total_value`, the positive physical
workflow utility from that rollout. The only rolling reward plot is the
trailing ten-rollout mean of these 80 saved values.

## Preservation requirements

- Every rollout retains its WiLab folder, trace, summary, events, and config.
- `episode_rewards.json` is rebuilt after every rollout from raw summaries.
- Every five-rollout update retains its replay archive and manifest.
- Every optimizer epoch retains its model checkpoint and row in `epochs.json`.
- Every five-rollout update retains its input policy and output `policy.json`.
- Every fifth training epoch is physically evaluated on the fixed five-seed
  probe suite and retained under that update block.

No replay penalty, TD loss, or checkpoint probe mean is allowed in the reward
plot.
