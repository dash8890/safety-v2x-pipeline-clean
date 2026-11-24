#!/usr/bin/env python3
"""Resume after causal train: finish holdout rows + mode2 train/eval. Keeps ddqn_best.pt."""
from __future__ import annotations

import json
import os
import sys

# Avoid Windows cp1252 crashes on any stray unicode
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch

from run_drl_experiment import make_agent_policy
from run_r4_correction import (
    CAUSAL_CKPT,
    HOLDOUT,
    MODE2_CKPT,
    OUT,
    REF_CKPT,
    eval_cfg,
    load_agent,
    paired,
    train_checkpoint,
)
from workflow_mdp import policy_b4, policy_value_greedy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    assert CAUSAL_CKPT.is_file(), f"missing {CAUSAL_CKPT}"
    assert REF_CKPT.is_file(), f"missing {REF_CKPT}"

    causal_agent = load_agent(CAUSAL_CKPT, device)
    ref_agent = load_agent(REF_CKPT, device)

    print("EVAL holdout corrected observation", flush=True)
    vg_c = eval_cfg("ValueGreedy", policy_value_greedy, HOLDOUT, "causal", "abstract")
    b4_c = eval_cfg("B4_joint", policy_b4, HOLDOUT, "causal", "abstract")
    dd_c = eval_cfg("DDQN_causal", make_agent_policy(causal_agent), HOLDOUT, "causal", "abstract")
    dd_ref = eval_cfg(
        "DDQN_R4_ref", make_agent_policy(ref_agent), HOLDOUT, "privileged", "abstract"
    )
    vg_ref = eval_cfg("ValueGreedy_R4obs", policy_value_greedy, HOLDOUT, "privileged", "abstract")
    dd_xfer = eval_cfg(
        "DDQN_R4_on_causal", make_agent_policy(ref_agent), HOLDOUT, "causal", "abstract"
    )
    print(
        f"causal DDQN V={dd_c['mean_value']:.2f} N={dd_c['mean_success']:.2f} "
        f"VG V={vg_c['mean_value']:.2f} N={vg_c['mean_success']:.2f}",
        flush=True,
    )

    print("TRAIN causal/mode2_event ->", MODE2_CKPT, flush=True)
    mode2_agent = train_checkpoint(
        MODE2_CKPT, "causal", "mode2_event", n_episodes=500, eval_every=50, device=device
    )

    print("EVAL holdout corrected radio", flush=True)
    vg_m = eval_cfg("ValueGreedy", policy_value_greedy, HOLDOUT, "causal", "mode2_event")
    b4_m = eval_cfg("B4_joint", policy_b4, HOLDOUT, "causal", "mode2_event")
    dd_m = eval_cfg("DDQN_mode2", make_agent_policy(mode2_agent), HOLDOUT, "causal", "mode2_event")

    result = {
        "preserved_r4_checkpoint": str(REF_CKPT.resolve()),
        "causal_checkpoint": str(CAUSAL_CKPT.resolve()),
        "mode2_checkpoint": str(MODE2_CKPT.resolve()),
        "holdout_seeds": HOLDOUT,
        "R4_reference_privileged_abstract": {
            "ddqn": dd_ref,
            "valuegreedy": vg_ref,
            "vs_vg": paired(dd_ref, vg_ref),
        },
        "corrected_observation_causal_abstract": {
            "ddqn": dd_c,
            "valuegreedy": vg_c,
            "b4": b4_c,
            "vs_vg": paired(dd_c, vg_c),
            "r4_ckpt_transfer_on_causal": dd_xfer,
        },
        "corrected_radio_causal_mode2_event": {
            "ddqn": dd_m,
            "valuegreedy": vg_m,
            "b4": b4_m,
            "vs_vg": paired(dd_m, vg_m),
            "note": "mode2_event = COMMIT->ACK->frag input->CPU->frag output; not full WiLab PHY",
        },
    }
    out = OUT / "r4_correction_results.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"WROTE {out}", flush=True)
    print("CORRECTED OBS vs VG:", result["corrected_observation_causal_abstract"]["vs_vg"], flush=True)
    print("CORRECTED RADIO vs VG:", result["corrected_radio_causal_mode2_event"]["vs_vg"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
