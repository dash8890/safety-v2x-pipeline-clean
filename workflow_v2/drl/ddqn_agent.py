"""Double DQN agent for discrete workflow placement (Farimani-style deadline DRL)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Deque, Tuple


class QNet(nn.Module):
    """Standard MLP Q (R4 frozen checkpoints)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNet(nn.Module):
    """Dueling DQN (Wang et al.) for utility-focused training."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)
        self.a = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        a = self.a(x)
        return v + a - a.mean(dim=-1, keepdim=True)


class Replay:
    def __init__(self, cap: int = 50000):
        self.buf: Deque = deque(maxlen=cap)

    def push(self, s, a, r, ns, d, mask, nmask):
        self.buf.append((s, a, r, ns, d, mask, nmask))

    def sample(self, n: int):
        idx = np.random.choice(len(self.buf), size=n, replace=False)
        batch = [self.buf[i] for i in idx]
        s, a, r, ns, d, m, nm = zip(*batch)
        return (
            np.stack(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.stack(ns),
            np.array(d, dtype=np.float32),
            np.stack(m),
            np.stack(nm),
        )

    def __len__(self):
        return len(self.buf)


class DDQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.98,
        device: str = "cpu",
        dueling: bool = False,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)
        Net = DuelingQNet if dueling else QNet
        self.q = Net(obs_dim, n_actions).to(self.device)
        self.tgt = Net(obs_dim, n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.replay = Replay()
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995

    def act(self, obs: np.ndarray, mask: np.ndarray, explore: bool = True) -> int:
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return 0
        if explore and np.random.rand() < self.eps:
            return int(np.random.choice(legal))
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            q = q.squeeze(0).cpu().numpy()
        q[~mask] = -1e9
        return int(np.argmax(q))

    def push(self, s, a, r, ns, d, mask, nmask):
        self.replay.push(s, a, r, ns, d, mask.astype(np.float32), nmask.astype(np.float32))

    def train_step(self, batch_size: int = 64) -> float:
        if len(self.replay) < batch_size:
            return 0.0
        s, a, r, ns, d, m, nm = self.replay.sample(batch_size)
        s_t = torch.as_tensor(s, device=self.device)
        ns_t = torch.as_tensor(ns, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        r_t = torch.as_tensor(r, device=self.device)
        d_t = torch.as_tensor(d, device=self.device)
        nm_t = torch.as_tensor(nm, device=self.device)

        q = self.q(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q(ns_t)
            q_next = q_next.masked_fill(nm_t < 0.5, -1e9)
            a_star = q_next.argmax(dim=1)
            q_tgt = self.tgt(ns_t).gather(1, a_star.view(-1, 1)).squeeze(1)
            y = r_t + self.gamma * (1.0 - d_t) * q_tgt
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.tgt.load_state_dict(self.q.state_dict())

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path: str):
        torch.save({"q": self.q.state_dict(), "eps": self.eps}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.tgt.load_state_dict(ckpt["q"])
        self.eps = ckpt.get("eps", self.eps_min)
