"""
Single-file PPO — CleanRL style.

Actor-Critic with shared MLP backbone.
GAE advantage estimation + PPO-clip objective + entropy bonus.
You own every line; no hidden magic.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# ------------------------------------------------------------------
# Network
# ------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal init — standard for PPO."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )

        # Actor head: mean of Gaussian policy
        self.actor_mean = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        # Learnable log-std (one per action dimension)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic head: scalar state-value
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.shared(x))

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor = None,
    ):
        features = self.shared(x)
        mean = self.actor_mean(features)
        # Clamp log_std: in the first run this was unbounded and entropy
        # climbed from 2.8 -> 6.3 over training (policy got MORE random,
        # not less). Range [-5, 0.5] keeps std in ~[0.007, 1.65].
        log_std = self.actor_log_std.clamp(-5.0, 0.5)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)   # sum over action dims
        entropy = dist.entropy().sum(-1)
        value = self.critic(features)

        return action, log_prob, entropy, value


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        num_minibatches: int = 4,
        hidden: int = 256,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches

        self.net = ActorCritic(obs_dim, act_dim, hidden=hidden).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

    # ------------------------------------------------------------------
    def compute_gae(
        self,
        rewards: torch.Tensor,     # (T, N)
        values: torch.Tensor,      # (T, N)
        dones: torch.Tensor,       # (T, N)
        next_value: torch.Tensor,  # (N,)  — per-env bootstrap value
    ):
        """
        Generalized Advantage Estimation, computed PER ENVIRONMENT along the
        time axis.

        Previous version flattened (T, N) -> (T*N,) and ran a single backward
        pass, which bootstrapped each step from the NEXT env at the same
        timestep (row-major neighbour) instead of the next timestep of the
        SAME env. That silently corrupted advantages. This version keeps the
        (T, N) shape so each column (env) is its own trajectory.
        """
        T, N = rewards.shape
        advantages = torch.zeros(T, N, device=self.device)
        last_gae = torch.zeros(N, device=self.device)

        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else values[t + 1]   # (N,)
            mask = 1.0 - dones[t]                              # (N,)
            delta = rewards[t] + self.gamma * nv * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    def update(
        self,
        obs: torch.Tensor,          # (B, obs_dim)
        actions: torch.Tensor,      # (B, act_dim)
        log_probs_old: torch.Tensor,  # (B,)
        returns: torch.Tensor,      # (B,)
        advantages: torch.Tensor,   # (B,)
    ) -> dict:
        """PPO update: multiple epochs over randomised minibatches."""
        B = obs.shape[0]
        mb_size = B // self.num_minibatches

        pg_losses, v_losses, ent_losses = [], [], []

        for _ in range(self.update_epochs):
            idx = torch.randperm(B, device=self.device)

            for start in range(0, B, mb_size):
                mb = idx[start: start + mb_size]

                _, new_logp, entropy, new_val = self.net.get_action_and_value(
                    obs[mb], actions[mb]
                )

                ratio = (new_logp - log_probs_old[mb]).exp()
                adv = advantages[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # PPO clip loss
                pg1 = -adv * ratio
                pg2 = -adv * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # Value loss (unclipped for simplicity)
                v_loss = 0.5 * (new_val.squeeze() - returns[mb]).pow(2).mean()

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy.mean().item())

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "v_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(ent_losses)),
        }

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({"net": self.net.state_dict(), "opt": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["opt"])