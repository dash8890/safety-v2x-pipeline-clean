#!/usr/bin/env python3
"""Build n-step, coverage-labelled replay from fresh physical WiLab traces."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "fresh_p4_replay.npz"
GAMMA = 0.99
N_STEP = 3


def decision_reward(decisions: list[dict], terminal: float, i: int) -> float:
    after = float(decisions[i + 1]["utility_before"]) if i + 1 < len(decisions) else terminal
    return after - float(decisions[i]["utility_before"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    replay = []
    manifests = []
    for source in args.source:
        for path in sorted(source.glob("**/fresh_trace.json")):
            raw = json.loads(path.read_text())
            decisions = [d for d in raw["decisions"] if d.get("kind") != "STOP_PLAN"]
            if not decisions:
                continue
            # Match the physical utility scale: successful workflows earn 1,
            # while each expired workflow incurs a -1 terminal penalty.
            terminal = float(raw["terminal_utility"]) - float(raw.get("expired_workflows", 0))
            manifests.append(str(path))
            for i, current in enumerate(decisions):
                reward = 0.0
                last = i
                for k in range(N_STEP):
                    j = i + k
                    if j >= len(decisions):
                        break
                    reward += (GAMMA**k) * decision_reward(decisions, terminal, j)
                    last = j
                    if j + 1 == len(decisions):
                        break
                done = last + 1 == len(decisions)
                next_features = (np.zeros((0, len(current["candidate_features"][0])), np.float32)
                                 if done else np.asarray(decisions[last + 1]["candidate_features"], np.float32))
                features = np.asarray(current["candidate_features"], np.float32)
                action = int(current["action_index"]) - 1
                kind = str(current.get("kind", ""))
                # Preserve scarce outcomes even before the first TD update.
                rare = bool(abs(reward) > 0 or kind in {"REQUEST_REMOTE_COMMIT", "RELEASE_EDGE", "REJECT_WORKFLOW"})
                replay.append({
                    "cur": features, "action": action, "next": next_features,
                    "reward": float(reward), "done": bool(done), "kind": kind,
                    "coverage_bin": int(current.get("coverage_bin", 0)), "rare": rare,
                    "source": str(path),
                })
    if not replay:
        raise RuntimeError("no executable P4 exploration decisions found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, transitions=np.asarray(replay, dtype=object))
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"n_step": N_STEP, "gamma": GAMMA,
                                    "traces": manifests, "transitions": len(replay)}, indent=2))
    print(json.dumps({"transitions": len(replay), "output": str(args.output), "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
