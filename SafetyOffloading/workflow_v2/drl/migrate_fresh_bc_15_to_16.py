#!/usr/bin/env python3
"""Migrate a 15-feature fresh BC checkpoint to the appended 16-feature schema.

The current schema appends ``frag_rate`` as feature 16.  A zero column in the
first layer preserves every legacy score exactly while making the checkpoint
loadable by the 16-feature MATLAB policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.input.read_text(encoding="utf-8"))
    if document.get("input_dim") != 15:
        raise ValueError(f"Expected a 15-feature checkpoint, got {document.get('input_dim')}")
    weights = document["state_dict"]["net.0.weight"]
    if not weights or any(len(row) != 15 for row in weights):
        raise ValueError("Checkpoint net.0.weight is not shaped [hidden, 15]")

    document["state_dict"]["net.0.weight"] = [row + [0.0] for row in weights]
    document["input_dim"] = 16
    document["source"] = "migrated_15_feature_bc_with_zero_frag_rate_weight"
    document["migration"] = {
        "from_input_dim": 15,
        "to_input_dim": 16,
        "added_feature": "frag_rate",
        "added_weight": 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document), encoding="utf-8")


if __name__ == "__main__":
    main()
