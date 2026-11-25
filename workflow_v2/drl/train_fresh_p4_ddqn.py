#!/usr/bin/env python3
"""Prioritized n-step Dueling Double-DQN training on physical WiLab traces.

Corrected BC-retention schedule: decays over the full 1,280-update training
budget (global_update / total_updates) instead of resetting each block.
"""
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


class NoisyLinear(nn.Module):
    """Factorised NoisyNet linear layer (Fortunato et al. 2017).

    Noise is active in train mode; eval mode uses deterministic mu weights.
    Factorised parameterisation: outer product of p+q independent noise scalars
    instead of p*q, keeping the noise parameter count O(p+q).
    """
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.weight_sigma = nn.Parameter(
            torch.full((out_features, in_features), sigma0 / in_features ** 0.5))
        self.bias_sigma = nn.Parameter(
            torch.full((out_features,), sigma0 / in_features ** 0.5))
        self.register_buffer('weight_eps', torch.zeros(out_features, in_features))
        self.register_buffer('bias_eps', torch.zeros(out_features))
        self._reset_parameters()
        self.sample_noise()

    def _reset_parameters(self):
        bound = 1.0 / self.in_features ** 0.5
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class DuelingQ(nn.Module):
    """Q(s,a)=V(s)+A(s,a)-mean_a A(s,a), for a variable candidate set.

    Value head is deterministic; advantage head uses NoisyLinear layers.
    Noise forces the advantage head to learn meaningful per-action differentials:
    the signal must be large enough to overcome injected noise, preventing the
    advantage stream from collapsing to near-zero (pure value-function mode).
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(15, 96), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(
            NoisyLinear(192, 64), nn.ReLU(), NoisyLinear(64, 1))

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()

    def components(self, x):
        if x.ndim == 1:
            x = x[None, :]
        h = self.encoder(x)
        state = h.mean(dim=0, keepdim=True)
        value = self.value(state).squeeze(-1).expand(h.shape[0])
        advantage = self.advantage(torch.cat((h, state.expand_as(h)), dim=1)).squeeze(-1)
        return value, advantage

    def forward(self, x):
        value, advantage = self.components(x)
        return value + advantage - advantage.mean()


def load_state(model: DuelingQ, path: Path) -> None:
    doc = json.loads(path.read_text())
    state = doc["state_dict"]
    tensors = {k: torch.tensor(v) for k, v in state.items()}
    if doc.get("architecture") == "dueling_ddqn":
        # Weights saved by write_model use standard names (weight/bias, no _mu suffix).
        # Load them into the model, mapping standard names → NoisyLinear.weight_mu where needed.
        own = dict(model.named_parameters())
        own_buf = dict(model.named_buffers())
        for k, t in tensors.items():
            if k in own:
                own[k].data.copy_(t)
            elif k.replace('.weight', '.weight_mu') in own:
                own[k.replace('.weight', '.weight_mu')].data.copy_(t)
            elif k.replace('.bias', '.bias_mu') in own:
                own[k.replace('.bias', '.bias_mu')].data.copy_(t)
        return
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
    # For NoisyLinear layers, export only mu weights under the standard key names
    # (drop _sigma and _eps, rename weight_mu/bias_mu → weight/bias) so that the
    # MATLAB inference code (WorkflowPolicyFreshDDQN/bcScore) sees the same key
    # schema as plain Linear layers and requires no modification.
    state = {}
    for k, v in model.state_dict().items():
        if k.endswith('_sigma') or k.endswith('_eps'):
            continue
        k = k.replace('_mu', '')
        state[k] = v.detach().tolist()
    path.write_text(json.dumps({"architecture": "dueling_ddqn", "input_dim": 15,
                                "encoder_hidden": 96, "head_hidden": 64, "state_dict": state,
                                "source": source}))


def fixed_state_metrics(model: DuelingQ, fixed_states, previous_actions, previous_q):
    q_values, values, advantages, actions = [], [], [], []
    model.eval()
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
    # Corrected BC retention: cumulative position in the full training budget.
    # Pass (block-1)*80 from the orchestrator so retention is global, not per-block.
    parser.add_argument("--global-update-start", type=int, default=0,
                        help="Cumulative optimizer updates completed before this block call")
    parser.add_argument("--total-updates", type=int, default=1280,
                        help="Total optimizer updates in the full training run (1280 for 16 main blocks)")
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.replay, allow_pickle=True)["transitions"]
    rare = np.asarray([bool(x["rare"]) for x in data])
    q = DuelingQ(); load_state(q, args.init if args.init and args.init.is_file() else BC)
    target = DuelingQ(); target.load_state_dict(q.state_dict()); target.eval()
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
            q.sample_noise()
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
        # BC retention decays globally over the full 1,280-update training budget.
        # global_update is the cumulative update index across all blocks; it never
        # resets between block calls, so the BC anchor persists through mid-training
        # rather than expiring within the first block.
        global_update = args.global_update_start + epoch
        retention = 0.50 * max(0.0, 1.0 - global_update / args.total_updates)
        for x, y in [demos[k] for k in rng.choice(len(demos), min(128, len(demos)), replace=False)]:
            loss = retention * F.cross_entropy(q(torch.tensor(x))[None, :], torch.tensor([y]))
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            target.load_state_dict(q.state_dict()); target.eval()
        metrics, previous_actions, previous_q = fixed_state_metrics(q, fixed_states, previous_actions, previous_q)
        q.train()
        history.append({"optimizer_update": epoch + 1, "mean_td_loss": float(np.mean(losses)),
                        "mean_absolute_td_error": float(np.mean(abs_td_errors)),
                        "td_error_p50": float(np.median(abs_td_errors)),
                        "td_error_p95": float(np.quantile(abs_td_errors, .95)),
                        "mean_sampled_reward": float(np.mean(sampled_rewards)),
                        "sum_sampled_reward": float(np.sum(sampled_rewards)),
                        "priority_p50": float(np.median(priorities)),
                        "priority_p95": float(np.quantile(priorities, .95)),
                        "bc_retention": retention,
                        "global_update": global_update, **metrics})
        print(json.dumps(history[-1]), flush=True)
        if (epoch + 1) % args.checkpoint_every == 0:
            write_model(args.checkpoint_dir / f"dueling_ddqn_update{epoch + 1:03d}.json", q,
                        "dueling_double_dqn_checkpoint")
    write_model(args.output, q, "dueling_double_dqn")
    args.report.write_text(json.dumps({"architecture": "dueling_ddqn", "seed": args.seed,
                                  "optimizer_updates": args.epochs,
                                  "global_update_start": args.global_update_start,
                                  "total_updates": args.total_updates,
                                  "history": history, "transitions": len(data),
                                  "rare_transitions": int(rare.sum())}, indent=2))


if __name__ == "__main__":
    main()
