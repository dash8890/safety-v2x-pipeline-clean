# Implementation contract v2.0 — index

Full prose contract was provided 2026-11-23 (workflow redesign). Normative executable settings live in:

- `workflow_v2/config/workflow_pilot.yaml`
- `workflow_v2/docs/R1_INTERFACE_MAP.md` (filled from this repo)

## Boundaries (non-negotiable)

- **New** workflow MDP under `workflow_v2/` — not an extension of P4–P9 binary offloading claims.
- Preserve corrected Mode-2 transport / fragmentation / CPU audit infrastructure.
- Safety packets = coexistence traffic, not optimization objective.
- Historical contract text deferred DRL until Mode-2 model-validity; **R3-M Mode-2 was negative**. A later **abstract-MDP** R4 was authorized separately and is now **accepted/frozen** — see `SEQUENCE_STATUS.md`. Mode-2 DRL remains **not authorized**.
- R1 maps interfaces; R2 builds engine; R3/R3-M Mode-2 line **closed**; abstract R4 **frozen**.

## Diamond-4 reference (engineering pilot)

```
             B
           /   \
Input → A         D → Result
           \   /
             C
```

Cycles A/B/C/D = 9/12/12/3 Mcycles; deadline 500 ms; helpers = 2 in deterministic tests.

## Module targets

See contract §I1. Stub paths created under `workflow_v2/{core,network,policy,metrics,tests}`.

## Acceptance

- R1: interface map + adapter tests — **PASS / frozen**.
- R2: workflow engine + R2-01…R2-12 — **PASS / frozen**.
- R3: joint-decision relevance — **CLOSED / frozen**; see `docs/R3_EXIT.md`.
- R3-M: Mode-2 gate — **CLOSED**; see `docs/R3M_EXIT.md` (B0 ≫ remote; no Mode-2 DRL).
- R4: abstract-MDP DDQN — **ACCEPTED / FROZEN**; see `SEQUENCE_STATUS.md`, `R4_DRL_FREEZE_ADDENDUM.md`.
- Mode-2 DRL / further algorithm development — **not authorized**.
