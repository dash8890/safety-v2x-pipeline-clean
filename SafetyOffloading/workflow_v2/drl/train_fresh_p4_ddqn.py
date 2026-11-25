#!/usr/bin/env python3
"""Prioritized n-step Dueling Double-DQN training on physical WiLab traces."""
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


class DuelingQ(nn.Module):
    """Q(s,a)=V(s)+A(s,a)-mean_a A(s,a), for a variable candidate set."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(15, 96), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1))

    def components(self, x):
        if x.ndim == 1:
            x = x[None, :]
        h = self.encoder(x)
        # The state stream is permutation-invariant over currently executable
        # candidate actions, rather than an independent score per candidate.
        state = h.mean(dim=0, keepdim=True)
        value = self.value(state).squeeze(-1).expand(h.shape[0])
        advantage = self.advantage(torch.cat((h, state.expand_as(h)), dim=1)).squeeze(-1)
        return value, advantage

    def forward(self, x):
        value, advantage = self.components(x)
        return value + advantage - advantage.mean()


def load_state(model: DuelingQ, path: Path) -> None:
    state = json.loads(path.read_text())["state_dict"]
    tensors = {k: torch.tensor(v) for k, v in state.items()}
    if "architecture" in json.loads(path.read_text()) and json.loads(path.read_text())["architecture"] == "dueling_ddqn":
        model.load_state_dict(tensors)
        return
    # A BC scorer is permitted only as an encoder warm start; it is not a
    # Dueling checkpoint and cannot supply value or advantage heads.
    if "net.0.weight" in tensors:
        model.encoder[0].weight.data.copy_(tensors["net.0.weight"])
        model.encoder[0].bias.data.copy_(tensors["net.0.bias"])


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


def write_model(path: Path, model: DuelingQ, source: str) -> None:
    state = {k: v.detach().tolist() for k, v in model.state_dict().items()}
    path.write_text(json.dumps({"architecture": "dueling_ddqn", "input_dim": 15,
                                "encoder_hidden": 96, "head_hidden": 64, "state_dict": state,
                                "source": source}))


def fixed_state_metrics(model: DuelingQ, fixed_states, previous_actions, previous_q):
    q_values, values, advantages, actions = [], [], [], []
    with torch.no_grad():
        for x in fixed_states:
            v, a = model.components(x)
            q = v + a - a.mean()
            q_values.extend(q.tolist()); values.extend(v.tolist()); advantages.extend(a.tolist())
            actions.append(int(q.argmax()))
    agreement = 1.0 if previous_actions is None else float(np.mean(np.asarray(actions) == previous_actions))
    q_values = np.asarray(q_values)
    drift = 0.0 if previous_q is None else float(np.mean(np.abs(q_values - previous_q)))
    return ({"q_p05": float(np.quantile(q_values, .05)), "q_p50": float(np.median(q_values)),
             "q_p95": float(np.quantile(q_values, .95)), "value_p05": float(np.quantile(values, .05)),
             "value_p50": float(np.median(values)), "value_p95": float(np.quantile(values, .95)),
             "advantage_p05": float(np.quantile(advantages, .05)), "advantage_p50": float(np.median(advantages)),
             "advantage_p95": float(np.quantile(advantages, .95)), "fixed_state_action_agreement": agreement,
             "q_mean_absolute_drift": drift}, np.asarray(actions), q_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "outputs" / "p4_checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.replay, allow_pickle=True)["transitions"]
    rare = np.asarray([bool(x["rare"]) for x in data])
    q = DuelingQ(); load_state(q, args.init if args.init and args.init.is_file() else BC)
    target = DuelingQ(); target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-4)
    priorities = np.ones(len(data), np.float64)
    demos = load_demos()
    history = []
    fixed_states = [torch.tensor(row["cur"]) for row in data[:min(128, len(data))]]
    previous_actions = None
    previous_q = None
    for epoch in range(args.epochs):
        losses = []
        abs_td_errors = []
        sampled_rewards = []
        for idx in draw(priorities, rare, max(512, len(data) // 2), rng):
            row = data[idx]
            sampled_rewards.append(float(row["reward"]))
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
            abs_td_errors.append(abs(float(td.detach())))
        # BC retention decays: it initializes viable behavior but does not
        # prevent exploration-supported improvement.
        retention = 0.50 * max(0, 1 - epoch / args.epochs)
        for x, y in [demos[k] for k in rng.choice(len(demos), min(128, len(demos)), replace=False)]:
            loss = retention * F.cross_entropy(q(torch.tensor(x))[None, :], torch.tensor([y]))
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            target.load_state_dict(q.state_dict())
        metrics, previous_actions, previous_q = fixed_state_metrics(q, fixed_states, previous_actions, previous_q)
        history.append({"optimizer_update": epoch + 1, "mean_td_loss": float(np.mean(losses)),
                        "mean_absolute_td_error": float(np.mean(abs_td_errors)),
                        "td_error_p50": float(np.median(abs_td_errors)),
                        "td_error_p95": float(np.quantile(abs_td_errors, .95)),
                        "mean_sampled_reward": float(np.mean(sampled_rewards)),
                        "sum_sampled_reward": float(np.sum(sampled_rewards)),
                        "priority_p50": float(np.median(priorities)),
                        "priority_p95": float(np.quantile(priorities, .95)),
                        "bc_retention": retention, **metrics})
        print(json.dumps(history[-1]), flush=True)
        if (epoch + 1) % args.checkpoint_every == 0:
            write_model(args.checkpoint_dir / f"dueling_ddqn_update{epoch + 1:03d}.json", q,
                        "dueling_double_dqn_checkpoint")
    write_model(args.output, q, "dueling_double_dqn")
    args.report.write_text(json.dumps({"architecture": "dueling_ddqn", "seed": args.seed, "optimizer_updates": args.epochs, "history": history, "transitions": len(data),
                                  "rare_transitions": int(rare.sum())}, indent=2))


if __name__ == "__main__":
    main()
