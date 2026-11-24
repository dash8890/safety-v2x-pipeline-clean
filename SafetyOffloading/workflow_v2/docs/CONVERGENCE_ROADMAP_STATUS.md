# Fresh WiLab convergence and evaluation plan

Authority: `FINAL CODING-AGENT DIRECTIVE — WiLab NR-V2X MODE-2 DRL`
(2026-11-24), revised 2026-11-24 after review of the fixed-replay DDQN
attempt. Historical abstract-radio and prior five-action Outcome-A artifacts
are not training or publication evidence under this plan.

## Terms and decision rule

**Training convergence** and **policy superiority** are different claims.

- *Convergence* is demonstrated inside development training: the learned
  exploit policy improves, then becomes stable under repeated probes, with
  stable critic/behavior diagnostics and without collapse to a lower-return
  action mode.
- *Superiority* is demonstrated only after the model is frozen: matched,
  held-out physical WiLab seeds compare that fixed policy with the accepted
  baselines.

No held-out seed is used to select an exploration schedule, a checkpoint, or
a stopping point. A training curve alone also does not prove global
optimality; the protection against a local optimum is diverse, coverage-led
development exploration plus independent restart and fixed-probe agreement.

## Completed foundations

### P0 — packet-level/action-interface correctness: PASS

- R1 adapter acceptance: 7/7 passed.
- R2 workflow/CPU/ACK/fragment/deadline acceptance: 12/12 passed.
- Fresh scheduling-action acceptance: 3/3 passed.
- The shared interface exposes only `COMMIT_LOCAL`,
  `REQUEST_REMOTE_COMMIT`, `RELEASE_EDGE`, `REJECT_WORKFLOW`, and
  `STOP_PLAN`. Remote release is ACK-gated and no planner action advances
  physical time.

Evidence: `tests/runBaselineAcceptance.m`.

### P1 — physically real scheduling headroom: PASS

Selected development regime: high CPU contention (`cpu_offered_load=0.85`),
development seeds 1101–1105.

- `LOCAL_EDF`: 8.0 completed workflows on average.
- causal `QUEUE_AWARE`: 15.4 completed workflows on average (+92.5%).
- `QUEUE_AWARE` completed 3.6 remote stages and sent 13.8 Mode-2 data
  fragments per seed on average.

Evidence: `outputs/p1_opportunity_run1/per_seed.csv`.

### P2/P3 — causal comparator and BC warm start: PASS

The accepted causal joint-placement comparator averaged 21.8 utility. The
candidate-index behavioral-cloning policy reached 99.8% development label
accuracy and 24.0 mean physical utility over its five matched development
seeds. BC is the warm start and a comparator, not a claim of RL convergence.

## Retired fixed-replay DDQN attempt

The current fixed-replay DDQN update is a **control only**. It repeatedly
optimizes 6,770 pre-existing transitions and has only five random-epsilon
exploration traces. It cannot discover further behavior while training and
does not log the probes required to make a convergence claim. Its result is
not eligible for checkpoint selection or held-out evaluation under this plan.

## P4 — exploration-driven development training

The candidate scorer remains the existing 15 -> 96 -> 1 non-dueling network.
This phase changes data collection and replay, not the policy architecture.
All exploratory actions pass the existing feasibility and safety masks.

### P4.1 — deterministic probe suite

Create and freeze a development-only probe suite before the first update:

- five fixed, untrained probe scenarios, distinct from rollout scenarios and
  final held-out scenarios;
- fixed candidate-state snapshot set, with stage, helper, slack, CBR, and
  queue-pressure strata;
- metrics at every probe: exploit utility/completions, deadline misses,
  safety PRR/loss/CBR, remote completions, action histogram, TD loss, and
  Q-value quantiles.

The probe uses greedy exploitation only; it never injects exploration noise.

### P4.2 — coverage-directed exploration

Run iterative, physical WiLab collection batches on development rollout
scenarios. Each batch contains independent scenarios across the selected
contact and queue-pressure strata. The behavior policy mixes:

1. BC-guided legal actions, to preserve known viable behavior;
2. epsilon-greedy legal actions with a scheduled epsilon decay; and
3. coverage-directed legal actions, sampled inversely to visitation counts
   for `(stage, candidate kind, helper/contact, slack, queue-pressure)` bins.

Every batch reserves a fixed minimum share for each reachable action class
and rare, high-pressure bins. Dominated or unsafe candidates remain masked;
exploration is broad over feasible decisions, not random unsafe injection.
All traces append to an episode replay store with scenario, behavior-mode,
and bin-coverage metadata.

### P4.3 — replay and learning cycle

For each collection batch:

1. build three-step returns from physical traces so delayed completion value
   reaches the responsible early placement/admission decision;
2. sample replay with priorities from TD error and explicit minimum quotas
   for rare-success, deadline-failure, remote-stranding, and early-admission
   transitions;
3. train from the BC initialization with the existing Double-DQN target and a
   decaying BC-retention term;
4. run the frozen greedy probe suite and save the checkpoint, replay manifest,
   and diagnostics; and
5. run a second independent restart from the same BC initializer on the same
   accumulated replay at each checkpoint-selection boundary.

The best checkpoint is chosen by the probe suite's predeclared primary metric
(mean valid utility), subject to the safety gate; never by a final held-out
result.

### P4.4 — convergence gate

Declare development convergence only when all conditions hold:

- three consecutive probe checkpoints have mean exploit utility within 5% of
  their best value and no downward trend;
- their fixed-snapshot greedy action agreement is at least 98%;
- median TD loss and Q-value quantiles are stable across those checkpoints,
  with no value explosion or action-mode collapse;
- the independent restart selects the same action mode and has probe utility
  within 5% of the selected checkpoint; and
- every probe satisfies the predeclared safety PRR/loss/CBR gate.

The retained checkpoint is then frozen with its replay manifest, seed list,
probe report, and source commit.

## P5 — final, separate physical evaluation

Only the frozen P4 checkpoint advances here. Run matched multi-seed WiLab
evaluation against `LOCAL_EDF`, causal queue-aware, causal joint, BC, EDF,
CBR-threshold, priority-only, unrestricted, and risk-aware policies. Report
completion/utility, latency/deadline behavior, safety PRR/loss/CBR,
fragmentation/load, paired deltas, and confidence intervals. No retraining,
exploration, or checkpoint selection occurs in P5.

## Operational sequencing

1. Retire the fixed-replay update and its queued physical evaluation as a
   diagnostic control.
2. Implement P4.1 probe logging and P4.2 coverage-directed collector.
3. Implement P4.3 n-step prioritized episode replay while preserving the
   15 -> 96 -> 1 non-dueling scorer.
4. Execute exploration/training/probe cycles until every P4.4 condition is
   satisfied.
5. Freeze exactly one checkpoint and execute P5 once.
