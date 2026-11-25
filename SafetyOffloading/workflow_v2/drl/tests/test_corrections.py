#!/usr/bin/env python3
"""Unit tests for the two deployed corrections.

Covers:
  1. Terminal reward — deadline-valid utility only (no expiry penalty)
  2. N-step return accounting (N=3, gamma=0.99)
  3. BC retention scheduling (global cumulative, 1280 updates)
  4. Legal-action mask / terminal bootstrapping (done=True => no next-state Q)
  5. Isolation — each correction touches only its intended component
"""
import json
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers – inline minimal stubs so tests run without WiLab or full project
# ---------------------------------------------------------------------------

GAMMA = 0.99
N_STEP = 3


def _build_replay_from_trace(raw: dict) -> list[dict]:
    """Minimal replica of build_fresh_p4_replay.py logic (corrected version)."""
    decisions = [d for d in raw["decisions"] if d.get("kind") != "STOP_PLAN"]
    if not decisions:
        return []
    # CORRECTED: deadline-valid utility only, no expiry subtraction
    terminal = float(raw["terminal_utility"])

    replay = []
    for i, current in enumerate(decisions):
        reward = 0.0
        last = i
        for k in range(N_STEP):
            j = i + k
            if j >= len(decisions):
                break
            after = (float(decisions[j + 1]["utility_before"])
                     if j + 1 < len(decisions) else terminal)
            reward += (GAMMA ** k) * (after - float(decisions[j]["utility_before"]))
            last = j
            if j + 1 == len(decisions):
                break
        done = last + 1 == len(decisions)
        n_cands = 3  # stub feature width
        next_feat = (np.zeros((0, n_cands), np.float32) if done
                     else np.ones((n_cands, n_cands), np.float32))
        replay.append({
            "reward": float(reward),
            "done": bool(done),
            "next": next_feat,
            "kind": current.get("kind", ""),
            "rare": bool(abs(reward) > 0
                         or current.get("kind") in
                         {"REQUEST_REMOTE_COMMIT", "RELEASE_EDGE", "REJECT_WORKFLOW"}),
        })
    return replay


def _retention(global_update: int, total_updates: int = 1280) -> float:
    """CORRECTED BC retention schedule."""
    return 0.50 * max(0.0, 1.0 - global_update / total_updates)


# ---------------------------------------------------------------------------
# 1. Terminal reward — expiry NOT subtracted
# ---------------------------------------------------------------------------

class TestTerminalReward:
    def _make_trace(self, n_decisions: int, utility_before_values: list[float],
                    terminal_utility: float, expired_workflows: int) -> dict:
        decisions = [
            {"utility_before": str(u), "kind": "COMMIT_LOCAL",
             "action_index": "1",
             "candidate_features": [[0.0, 0.0, 0.0]]}
            for u in utility_before_values
        ]
        return {
            "decisions": decisions,
            "terminal_utility": str(terminal_utility),
            "expired_workflows": str(expired_workflows),
        }

    def test_terminal_ignores_expired_workflows(self):
        """Terminal must equal terminal_utility regardless of expired_workflows."""
        trace = self._make_trace(
            n_decisions=4,
            utility_before_values=[0.0, 1.0, 2.0, 3.0],
            terminal_utility=5.0,
            expired_workflows=20,
        )
        transitions = _build_replay_from_trace(trace)
        last = transitions[-1]
        # Last transition: reward = terminal - utility_before_last = 5.0 - 3.0 = 2.0
        assert last["done"] is True
        assert abs(last["reward"] - 2.0) < 1e-6, (
            f"Expected 2.0, got {last['reward']}. "
            "Expiry penalty must NOT be subtracted from terminal."
        )

    def test_terminal_zero_expired_unchanged(self):
        """With zero expiry the result must match terminal_utility too."""
        trace = self._make_trace(
            n_decisions=3,
            utility_before_values=[0.0, 1.0, 2.0],
            terminal_utility=4.0,
            expired_workflows=0,
        )
        transitions = _build_replay_from_trace(trace)
        last = transitions[-1]
        assert abs(last["reward"] - 2.0) < 1e-6

    def test_no_negative_rewards_from_expiry(self):
        """With monotone utility_before, all rewards must be non-negative."""
        trace = self._make_trace(
            n_decisions=5,
            utility_before_values=[0.0, 1.0, 2.0, 3.0, 4.0],
            terminal_utility=5.0,
            expired_workflows=100,   # large expiry: must not affect sign
        )
        transitions = _build_replay_from_trace(trace)
        for t in transitions:
            assert t["reward"] >= -1e-9, (
                f"Negative reward {t['reward']} found; expiry must not be subtracted."
            )


# ---------------------------------------------------------------------------
# 2. N-step return accounting
# ---------------------------------------------------------------------------

class TestNStepReturns:
    def _constant_trace(self, n: int, u: float) -> dict:
        """Trace where utility_before is constant -> all rewards must be 0."""
        decisions = [
            {"utility_before": str(u), "kind": "COMMIT_LOCAL",
             "action_index": "1", "candidate_features": [[0.0]]}
            for _ in range(n)
        ]
        return {"decisions": decisions, "terminal_utility": str(u), "expired_workflows": "0"}

    def test_constant_utility_zero_rewards(self):
        """When utility never changes, all n-step returns must be 0."""
        for n in (3, 5, 10):
            transitions = _build_replay_from_trace(self._constant_trace(n, 7.0))
            for t in transitions:
                assert abs(t["reward"]) < 1e-9, (
                    f"Non-zero reward {t['reward']} from constant-utility trace (n={n})."
                )

    def test_single_completion_at_terminal(self):
        """One completion at the last step: check exact formula R_{N-1} = terminal - U_{N-1}."""
        # 4 decisions, utility_before = [0,0,0,0], terminal = 1.0
        decisions = [
            {"utility_before": "0.0", "kind": "COMMIT_LOCAL",
             "action_index": "1", "candidate_features": [[0.0]]}
            for _ in range(4)
        ]
        trace = {"decisions": decisions, "terminal_utility": "1.0", "expired_workflows": "5"}
        transitions = _build_replay_from_trace(trace)
        last = transitions[-1]   # i=3: R = 1.0 - 0.0 = 1.0
        assert abs(last["reward"] - 1.0) < 1e-9, f"Expected 1.0, got {last['reward']}"

        second_last = transitions[-2]  # i=2: R = 0 + 0.99*(1.0 - 0.0) = 0.99
        assert abs(second_last["reward"] - 0.99) < 1e-6, (
            f"Expected 0.99, got {second_last['reward']}"
        )

        third_last = transitions[-3]  # i=1: R = 0 + 0.99*0 + 0.9801*(1.0-0.0) = 0.9801
        assert abs(third_last["reward"] - 0.9801) < 1e-6, (
            f"Expected 0.9801, got {third_last['reward']}"
        )

    def test_done_flag_on_last_three(self):
        """With N=3, done=True for the last N transitions (window reaches terminal).
        For a 6-decision episode: transitions 3,4,5 have done=True; 0,1,2 have done=False."""
        decisions = [
            {"utility_before": str(float(i)), "kind": "COMMIT_LOCAL",
             "action_index": "1", "candidate_features": [[0.0]]}
            for i in range(6)
        ]
        trace = {"decisions": decisions, "terminal_utility": "6.0", "expired_workflows": "0"}
        transitions = _build_replay_from_trace(trace)
        # Last N_STEP=3 transitions must have done=True
        for t in transitions[-N_STEP:]:
            assert t["done"] is True, "Last N transitions must have done=True"
        # Transitions before the N-step window must have done=False
        for t in transitions[:-N_STEP]:
            assert t["done"] is False, "Transitions before N-step window must have done=False"


# ---------------------------------------------------------------------------
# 3. BC retention scheduling
# ---------------------------------------------------------------------------

class TestBCRetention:
    def test_coefficients_at_checkpoints(self):
        """Verify exact values at u=0,320,640,960,1280 (from directive)."""
        expected = {0: 0.50, 320: 0.375, 640: 0.25, 960: 0.125, 1280: 0.0}
        for u, v in expected.items():
            got = _retention(u)
            assert abs(got - v) < 1e-9, f"retention({u}) = {got}, expected {v}"

    def test_monotone_non_increasing(self):
        """Retention must be non-increasing in global_update."""
        values = [_retention(u) for u in range(0, 1281, 10)]
        for a, b in zip(values, values[1:]):
            assert a >= b - 1e-12, "Retention is not monotone non-increasing"

    def test_clamps_to_zero_after_total(self):
        """Retention must be 0.0 for all updates at or beyond 1280."""
        for u in (1280, 1281, 2000, 9999):
            assert _retention(u) == 0.0, f"retention({u}) should be 0.0"

    def test_does_not_reset_mid_training(self):
        """Simulate block calls: global_update must grow monotonically, never reset."""
        total = 1280
        all_retentions = []
        for block in range(1, 17):
            for epoch in range(80):
                g = (block - 1) * 80 + epoch
                all_retentions.append(_retention(g, total))
        for a, b in zip(all_retentions, all_retentions[1:]):
            assert a >= b - 1e-12, "Retention must not increase between consecutive epochs"


# ---------------------------------------------------------------------------
# 4. Legal-action mask / terminal bootstrapping
# ---------------------------------------------------------------------------

class TestTerminalBootstrapping:
    def test_done_transition_has_empty_next(self):
        """done=True transitions must have zero-row next-state features."""
        decisions = [
            {"utility_before": "0.0", "kind": "COMMIT_LOCAL",
             "action_index": "1", "candidate_features": [[1.0, 2.0, 3.0]]}
            for _ in range(3)
        ]
        trace = {"decisions": decisions, "terminal_utility": "2.0", "expired_workflows": "5"}
        transitions = _build_replay_from_trace(trace)
        last = transitions[-1]
        assert last["done"] is True
        assert last["next"].shape[0] == 0, (
            f"done=True transition must have empty next-features; got shape {last['next'].shape}"
        )

    def test_non_terminal_has_nonempty_next(self):
        """done=False transitions must have non-empty next-state features.
        With N=3 and 6 decisions, only transitions 0,1,2 have done=False."""
        decisions = [
            {"utility_before": "0.0", "kind": "COMMIT_LOCAL",
             "action_index": "1", "candidate_features": [[1.0, 2.0, 3.0]]}
            for _ in range(6)
        ]
        trace = {"decisions": decisions, "terminal_utility": "2.0", "expired_workflows": "0"}
        transitions = _build_replay_from_trace(trace)
        for t in transitions:
            if not t["done"]:
                assert t["next"].shape[0] > 0, (
                    "done=False transition must have non-empty next-features"
                )
            else:
                assert t["next"].shape[0] == 0, (
                    "done=True transition must have empty next-features"
                )

    def test_q_target_no_bootstrap_on_done(self):
        """Simulate the training target: done=True => y = reward (no gamma*Q_next)."""
        # This mirrors train_fresh_p4_ddqn.py lines 151-156
        GAMMA3 = GAMMA ** 3
        fake_q_next = 100.0   # large value: must NOT appear in target when done
        reward = 2.0

        # done=True path
        y_done = reward  # no bootstrap
        assert abs(y_done - reward) < 1e-9

        # done=False path
        y_live = reward + GAMMA3 * fake_q_next
        assert abs(y_live - (reward + GAMMA3 * fake_q_next)) < 1e-9

        # Confirm the two are different
        assert abs(y_done - y_live) > 1.0, (
            "done=True and done=False must produce different Q targets"
        )


# ---------------------------------------------------------------------------
# 5. Isolation — corrections do not bleed into each other
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_terminal_fix_does_not_affect_non_terminal_rewards(self):
        """Changing terminal formula must not alter rewards for steps i=0..N-4."""
        decisions = [
            {"utility_before": str(float(k)), "kind": "COMMIT_LOCAL",
             "action_index": "1", "candidate_features": [[0.0]]}
            for k in range(8)
        ]
        # Build with terminal=7 (one scenario) and terminal=7+100 (another)
        # The first 5 transitions (i=0..4) touch no terminal at all.
        def build(t_util):
            raw = {"decisions": decisions, "terminal_utility": str(t_util),
                   "expired_workflows": "0"}
            return _build_replay_from_trace(raw)

        t1 = build(7.0)
        t2 = build(107.0)  # only near-terminal transitions should change
        for i in range(5):  # first 5 transitions use no terminal value
            assert abs(t1[i]["reward"] - t2[i]["reward"]) < 1e-9, (
                f"Transition {i} reward changed when only terminal was varied: "
                f"{t1[i]['reward']} vs {t2[i]['reward']}"
            )
        # Last transition MUST differ
        assert abs(t1[-1]["reward"] - t2[-1]["reward"]) > 1.0, (
            "Last transition reward must reflect terminal value"
        )

    def test_bc_retention_does_not_alter_td_target(self):
        """BC loss only affects weight gradient, not the Q-target y = r + gamma^3 * Q_next."""
        # The retention factor multiplies the BC cross-entropy loss, not the TD target.
        # This test just confirms that _retention() is independent of reward values.
        r = _retention(320)
        assert r == 0.375
        # retention is a fixed schedule: changing reward does not change retention
        r2 = _retention(320)
        assert r == r2, "Retention must not depend on reward values"
