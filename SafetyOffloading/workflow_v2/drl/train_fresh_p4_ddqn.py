#!/usr/bin/env python3
"""Prioritized, n-step Double-DQN update for the fixed fresh candidate scorer."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPLAY = ROOT / "outputs" / "fresh_p4_replay.npz"
BC = ROOT / "outputs" / "fresh_candidate_bc.json"
OUT = ROOT / "outputs" / "fresh_ddqn.json"
REPORT = ROOT / "outputs" / "fresh_p4_train_report.json"
DEMO = ROOT.parent / "outputs" / "p3_bc_dataset_v3"
GAMMA3 = 0.99 ** 3


class Q(nn.Module):
    # Deliberately identical to the approved non-dueling 15 -> 96 -> 1 scorer.
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(15, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_state(model: Q, path: Path) -> None:
    state = json.loads(path.read_text())["state_dict"]
    model.load_state_dict({k: torch.tensor(v) for k, v in state.items()})


def load_demos():
    demos = []
    for path in DEMO.glob("**/fresh_demos.json"):
        for row in json.loads(path.read_text()):
            x = np.asarray(row["candidate_features"], np.float32)
            y = int(row["action_index"]) - 1
            if x.ndim == 1:
                x = x[None, :]
            if 0 <= y < len(x):
                demos.append((x, y))
    return demos


def draw(priorities, rare, count, rng):
    """PER sample with a guaranteed rare-transition share."""
    rare_i = np.flatnonzero(rare)
    normal_i = np.arange(len(priorities))
    n_rare = min(len(rare_i), max(1, count // 4))
    choose_rare = rng.choice(rare_i, n_rare, replace=len(rare_i) < n_rare,
                              p=priorities[rare_i] / priorities[rare_i].sum()) if len(rare_i) else np.empty(0, int)
    choose_all = rng.choice(normal_i, count - len(choose_rare), replace=True,
                             p=priorities / priorities.sum())
    return np.concatenate((choose_rare, choose_all))


def write_model(path: Path, model: Q, source: str) -> None:
    state = {k: v.detach().tolist() for k, v in model.state_dict().items()}
    path.write_text(json.dumps({"input_dim": 15, "hidden": 96, "state_dict": state,
                                "source": source}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "outputs" / "p4_checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(23)
    torch.manual_seed(23)
    data = np.load(REPLAY, allow_pickle=True)["transitions"]
    rare = np.asarray([bool(x["rare"]) for x in data])
    q = Q(); load_state(q, BC)
    target = Q(); target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-4)
    priorities = np.ones(len(data), np.float64)
    demos = load_demos()
    history = []
    for epoch in range(80):
        losses = []
        for idx in draw(priorities, rare, max(512, len(data) // 2), rng):
            row = data[idx]
            cur = torch.tensor(row["cur"]); nxt = torch.tensor(row["next"])
            qa = q(cur)[int(row["action"])]
            with torch.no_grad():
                if row["done"] or len(nxt) == 0:
                    y = torch.tensor(float(row["reward"]))
                else:
                    online = q(nxt); pick = int(online.argmax())
                    y = torch.tensor(float(row["reward"])) + GAMMA3 * target(nxt)[pick]
            td = qa - y
            loss = F.smooth_l1_loss(qa, y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(q.parameters(), 5); opt.step()
            priorities[idx] = abs(float(td.detach())) + 1e-3
            losses.append(float(loss.detach()))
        # BC retention decays: it initializes viable behavior but does not
        # prevent exploration-supported improvement.
        retention = 0.50 * (1 - epoch / 80)
        for x, y in [demos[k] for k in rng.choice(len(demos), min(128, len(demos)), replace=False)]:
            loss = retention * F.cross_entropy(q(torch.tensor(x))[None, :], torch.tensor([y]))
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            target.load_state_dict(q.state_dict())
        history.append({"epoch": epoch + 1, "mean_td_loss": float(np.mean(losses)),
                        "priority_p50": float(np.median(priorities)),
                        "priority_p95": float(np.quantile(priorities, .95)),
                        "bc_retention": retention})
        print(json.dumps(history[-1]), flush=True)
        if (epoch + 1) % args.checkpoint_every == 0:
            write_model(args.checkpoint_dir / f"fresh_ddqn_epoch{epoch + 1:03d}.json", q,
                        "fresh_p4_nstep_prioritized_ddqn_checkpoint")
    write_model(OUT, q, "fresh_p4_nstep_prioritized_ddqn")
    REPORT.write_text(json.dumps({"history": history, "transitions": len(data),
                                  "rare_transitions": int(rare.sum())}, indent=2))


if __name__ == "__main__":
    main()
