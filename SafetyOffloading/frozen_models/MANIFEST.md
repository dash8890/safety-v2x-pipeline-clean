# Frozen policy manifest

Frozen for held-out evaluation on 18 November 2026 (Asia/Seoul).

| Artifact | SHA-256 |
|---|---|
| `dqn_policy_v1.json` | `076BBDA22DD2ABBC2314A86F044C2BDF17B9D7CCC8D114950476DF9538B3F258` |
| `risk_estimator_v1.json` | `F377526DF95263DF4B0CF3298B13F8F5E9915525F187F27BEA82BA2BC24627DA` |

Training used native WiLab paired engineering scenarios with seeds 1–3.
Held-out configuration `config/heldout.json` uses disjoint seeds 101–105.
Do not modify these artifacts after inspecting held-out results; create a new
version and repeat the complete split if retraining is required.
