# Fresh WiLab convergence roadmap status

Authority: `FINAL CODING-AGENT DIRECTIVE — WiLab NR-V2X MODE-2 DRL` (2026-11-24).
Historical abstract-radio and prior five-action Outcome-A artifacts are not
training or publication evidence under this roadmap.

## P0 — packet-level/action-interface correctness

Status: **PASS** on 2026-11-24.

- R1 adapter acceptance: 7/7 passed.
- R2 workflow/CPU/ACK/fragment/deadline acceptance: 12/12 passed.
- Fresh scheduling-action acceptance: 3/3 passed.
- Shared interface now exposes only `COMMIT_LOCAL`,
  `REQUEST_REMOTE_COMMIT`, `RELEASE_EDGE`, `REJECT_WORKFLOW`, and
  `STOP_PLAN`.  Remote release remains ACK-gated and no planner action
  advances physical time.

Evidence: `tests/runBaselineAcceptance.m`.

## P1 — physically real scheduling headroom

Status: **next active stage**.

Before BC/DDQN convergence work starts, run the predeclared six-cell
development matrix (three CPU-contention levels x favorable/intermittent
observed helper contact, five matched development seeds per cell) against the
fresh action interface.  Required evidence is queue pressure, packet latency,
remote completion, radio cost, and attainable utility headroom versus
`LOCAL_EDF`.  The P1 output selects one physically plausible regime; it is
not a final-test result.

## Convergence policy

P3/P4 training is prohibited until P1 opportunity evidence and P2 comparator
acceptance are complete.  The legacy five-action DDQN and its old checkpoint
curves are retained only as historical engineering artifacts and are not used
for model selection, convergence, or paper comparison.
