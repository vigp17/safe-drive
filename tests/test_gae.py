"""
Tests for compute_gae (ppo/ppo.py).

This is the function the principal-engineer review found broken: the old
version flattened (T, N) -> (T*N,) and bootstrapped each step from the *next
environment* at the same timestep instead of the next timestep of the *same*
environment. These tests lock in the fix:

  1. env-0's advantages match an independent scalar reference implementation
  2. scrambling another env's rewards leaves env-0's advantages untouched
     (proof the columns are independent trajectories)
  3. a `done` mid-rollout stops bootstrapping across the episode boundary
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppo.ppo import PPOAgent

GAMMA, LAM = 0.99, 0.95


def make_agent():
    return PPOAgent(obs_dim=259, act_dim=2, device=torch.device("cpu"),
                    gamma=GAMMA, gae_lambda=LAM)


def ref_gae(rewards, values, next_value, dones):
    """Obviously-correct single-trajectory GAE, plain Python scalars."""
    T = len(rewards)
    adv = [0.0] * T
    last = 0.0
    for t in reversed(range(T)):
        nv = next_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * nv * mask - values[t]
        last = delta + GAMMA * LAM * mask * last
        adv[t] = last
    return adv


def test_matches_reference_single_env():
    agent = make_agent()
    rewards = torch.tensor([[1.0], [1.0], [1.0], [2.0]])     # (T=4, N=1)
    values = torch.tensor([[0.5], [0.4], [0.6], [0.5]])
    dones = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
    next_value = torch.tensor([0.0])

    adv, ret = agent.compute_gae(rewards, values, dones, next_value)
    ref = ref_gae([1, 1, 1, 2], [0.5, 0.4, 0.6, 0.5], 0.0, [0, 0, 0, 0])

    for t in range(4):
        assert abs(adv[t, 0].item() - ref[t]) < 1e-5, f"t={t}: {adv[t,0].item()} vs {ref[t]}"
    # returns = advantages + values
    assert torch.allclose(ret, adv + values, atol=1e-6)
    print("  single-env GAE matches scalar reference  PASS")


def test_envs_are_independent():
    """The bug, made into a test: env-0's advantages must not change when
    another env's rewards are scrambled."""
    agent = make_agent()
    T, N = 5, 3
    torch.manual_seed(0)
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = torch.zeros(T, N)
    next_value = torch.randn(N)

    adv_a, _ = agent.compute_gae(rewards, values, dones, next_value)

    # Scramble ONLY env 1 and 2's rewards; env 0 untouched.
    rewards2 = rewards.clone()
    rewards2[:, 1] = torch.randn(T)
    rewards2[:, 2] = torch.randn(T) * 10.0
    adv_b, _ = agent.compute_gae(rewards2, values, dones, next_value)

    assert torch.allclose(adv_a[:, 0], adv_b[:, 0], atol=1e-6), \
        "env-0 advantages changed when other envs' rewards changed — envs not isolated!"
    # and env 1 SHOULD have changed (sanity: the scramble did something)
    assert not torch.allclose(adv_a[:, 1], adv_b[:, 1], atol=1e-3)
    print("  env-0 isolated from other envs' rewards  PASS")


def test_done_stops_bootstrap():
    """A done at time t means step t must not bootstrap from t+1 (mask=0)."""
    agent = make_agent()
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    next_value = torch.tensor([0.0])

    done_mid = torch.tensor([[0.0], [1.0], [0.0]])   # episode ends at t=1
    adv, _ = agent.compute_gae(rewards, values, done_mid, next_value)

    # At t=1 with done=1: delta = r - v = 1 - 0.5 = 0.5, and no carry into t=0
    # beyond the masked term. Verify against the masked reference.
    ref = ref_gae([1, 1, 1], [0.5, 0.5, 0.5], 0.0, [0, 1, 0])
    for t in range(3):
        assert abs(adv[t, 0].item() - ref[t]) < 1e-5
    print("  done mid-rollout stops bootstrap  PASS")


if __name__ == "__main__":
    tests = [test_matches_reference_single_env,
             test_envs_are_independent,
             test_done_stops_bootstrap]
    print("GAE tests")
    print("-" * 50)
    for t in tests:
        print(t.__name__)
        t()
    print("-" * 50)
    print(f"All {len(tests)} GAE tests passed.")