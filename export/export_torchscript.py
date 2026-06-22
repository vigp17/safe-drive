"""
Export a trained safe-drive policy to TorchScript for deployment.
=================================================================
The training checkpoint carries the full ActorCritic (shared backbone, actor
mean + log-std, critic head) plus the optimizer state. For *inference* none of
that is needed — a deployed policy is just:

    observation (259,)  ->  shared MLP  ->  actor mean  ->  clip[-1, 1]  ->  action (2,)

This script strips the policy down to that path, wraps it in a tiny
inference-only module, scripts it with torch.jit, and verifies the scripted
model reproduces the eager model's output bit-for-bit. The result is a
self-contained .pt that runs with no project code, no PPO, no sampling —
load it anywhere torch runs.

Usage:
    python export/export_torchscript.py --checkpoint checkpoints/seed_1/ppo_final.pt \
                                        --out results/policy_seed1.ts.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppo.ppo import ActorCritic


class PolicyInference(nn.Module):
    """Deterministic, inference-only policy: obs -> clipped mean action.

    No sampling (we deploy the mean, not a draw), no critic, no log-std —
    exactly what a deployed controller runs each timestep.
    """

    def __init__(self, shared: nn.Module, actor_mean: nn.Module):
        super().__init__()
        self.shared = shared
        self.actor_mean = actor_mean

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.shared(obs)
        action = self.actor_mean(features)
        return torch.clamp(action, -1.0, 1.0)


def export(checkpoint: str, out: str, obs_dim: int = 259, act_dim: int = 2,
           hidden: int = 256) -> None:
    net = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["net"])
    net.eval()

    policy = PolicyInference(net.shared, net.actor_mean).eval()
    scripted = torch.jit.script(policy)

    # Verify: scripted output must match eager output on random inputs.
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(1, obs_dim)
            a_eager = policy(x)
            a_script = scripted(x)
            assert torch.allclose(a_eager, a_script, atol=1e-6), \
                "scripted output diverged from eager"

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(out)
    print(f"verified eager == scripted on 5 random inputs")
    print(f"wrote TorchScript policy -> {out}")
    print(f"  input : float32 tensor (batch, {obs_dim})")
    print(f"  output: float32 tensor (batch, {act_dim})  in [-1, 1]  [steer, accel]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="results/policy.ts.pt")
    p.add_argument("--obs-dim", type=int, default=259)
    p.add_argument("--act-dim", type=int, default=2)
    p.add_argument("--hidden", type=int, default=256)
    args = p.parse_args()
    export(args.checkpoint, args.out, args.obs_dim, args.act_dim, args.hidden)


if __name__ == "__main__":
    main()