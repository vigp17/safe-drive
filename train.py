"""
AV-RL Training Entrypoint — PPO on MetaDrive

Usage:
    # Local smoke test (4 envs, 10k steps, no W&B)
    python train.py --test

    # Full AWS run (reads configs/ppo.yaml)
    python train.py --full

    # Custom seed (for multi-seed sweep on AWS)
    python train.py --full --seed 3
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from envs.driving_env import DrivingEnv
from ppo.ppo import PPOAgent


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_env(env_config: dict, seed: int):
    """Factory for vectorised env creation."""
    def _init():
        cfg = dict(env_config)
        cfg["start_seed"] = seed
        return DrivingEnv(config=cfg)
    return _init


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--test", action="store_true",
                        help="Smoke test: 4 envs, 10k steps, no W&B")
    parser.add_argument("--full", action="store_true",
                        help="Full training run with W&B logging")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_wandb = args.full and not args.no_wandb

    # ── Smoke test overrides ──────────────────────────────────────
    if args.test:
        cfg["ppo"]["num_envs"] = 1       # MetaDrive = 1 env per process (Panda3D)
        cfg["ppo"]["total_steps"] = 10_000
        cfg["ppo"]["rollout_steps"] = 128
        use_wandb = False
        print("=" * 50)
        print("[TEST MODE]  1 env | 10k steps | no W&B")
        print("=" * 50)

    # ── Device ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── W&B ──────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg["logging"]["project"],
            name=f"{cfg['logging']['run_name']}-seed{args.seed}",
            config=cfg,
        )

    # ── Vectorised envs ───────────────────────────────────────────
    num_envs = cfg["ppo"]["num_envs"]
    # MetaDrive runs on Panda3D, a per-process singleton: you cannot have
    # more than one MetaDrive env in the same process. Therefore:
    #   num_envs == 1  -> SyncVectorEnv  (single instance, single process)
    #   num_envs  > 1  -> AsyncVectorEnv (each env in its own subprocess)
    VecEnv = SyncVectorEnv if num_envs == 1 else AsyncVectorEnv
    envs = VecEnv([
        make_env(cfg["env"], seed=args.seed * 1000 + i)
        for i in range(num_envs)
    ])

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    print(f"Obs dim: {obs_dim} | Act dim: {act_dim} | Envs: {num_envs}")

    # ── Agent ─────────────────────────────────────────────────────
    ppo_cfg = cfg["ppo"]
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        lr=ppo_cfg["lr"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_coef=ppo_cfg["clip_coef"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        update_epochs=ppo_cfg["update_epochs"],
        num_minibatches=ppo_cfg["num_minibatches"],
        hidden=ppo_cfg.get("hidden", 256),
    )

    rollout_steps = ppo_cfg["rollout_steps"]
    total_steps = ppo_cfg["total_steps"]
    batch_size = num_envs * rollout_steps

    # ── Rollout storage ───────────────────────────────────────────
    obs_buf  = torch.zeros(rollout_steps, num_envs, obs_dim, device=device)
    act_buf  = torch.zeros(rollout_steps, num_envs, act_dim, device=device)
    rew_buf  = torch.zeros(rollout_steps, num_envs, device=device)
    done_buf = torch.zeros(rollout_steps, num_envs, device=device)
    logp_buf = torch.zeros(rollout_steps, num_envs, device=device)
    val_buf  = torch.zeros(rollout_steps, num_envs, device=device)

    # ── Init envs ─────────────────────────────────────────────────
    obs_np, _ = envs.reset(seed=args.seed)
    obs = torch.FloatTensor(obs_np).to(device)

    global_step = 0
    update = 0
    ep_returns, ep_lengths = [], []
    ep_return = np.zeros(num_envs)
    ep_length = np.zeros(num_envs, dtype=int)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    start_time = time.time()

    print(f"Starting training | total_steps={total_steps:,} | batch_size={batch_size:,}")

    # ── Training loop ─────────────────────────────────────────────
    while global_step < total_steps:

        # Collect rollout
        for step in range(rollout_steps):
            obs_buf[step] = obs
            with torch.no_grad():
                action, logp, _, value = agent.net.get_action_and_value(obs)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value.squeeze(-1)

            act_np = action.cpu().numpy().clip(-1.0, 1.0)
            next_obs_np, reward, terminated, truncated, info = envs.step(act_np)
            done = terminated | truncated

            rew_buf[step]  = torch.FloatTensor(reward).to(device)
            done_buf[step] = torch.FloatTensor(done.astype(float)).to(device)

            ep_return += reward
            ep_length += 1
            for i, d in enumerate(done):
                if d:
                    ep_returns.append(float(ep_return[i]))
                    ep_lengths.append(int(ep_length[i]))
                    ep_return[i] = 0
                    ep_length[i] = 0

            obs = torch.FloatTensor(next_obs_np).to(device)
            global_step += num_envs

        # Bootstrap value
        with torch.no_grad():
            next_val = agent.net.get_value(obs).squeeze(-1).mean()

        # GAE
        advantages, returns = agent.compute_gae(
            rew_buf.view(-1),
            val_buf.view(-1),
            done_buf.view(-1),
            next_val,
        )

        # PPO update
        metrics = agent.update(
            obs_buf.view(batch_size, obs_dim),
            act_buf.view(batch_size, act_dim),
            logp_buf.view(batch_size),
            returns,
            advantages,
        )

        update += 1
        sps = global_step / (time.time() - start_time)

        # Logging
        if update % cfg["logging"]["log_interval"] == 0:
            mean_ret = float(np.mean(ep_returns[-20:])) if ep_returns else 0.0
            mean_len = float(np.mean(ep_lengths[-20:])) if ep_lengths else 0.0
            print(
                f"step={global_step:>8,} | "
                f"return={mean_ret:>7.2f} | "
                f"ep_len={mean_len:>6.1f} | "
                f"pg={metrics['pg_loss']:.4f} | "
                f"v={metrics['v_loss']:.4f} | "
                f"ent={metrics['entropy']:.4f} | "
                f"sps={sps:.0f}"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "charts/mean_return": mean_ret,
                    "charts/mean_ep_length": mean_len,
                    "losses/pg_loss": metrics["pg_loss"],
                    "losses/v_loss": metrics["v_loss"],
                    "losses/entropy": metrics["entropy"],
                    "charts/sps": sps,
                    "global_step": global_step,
                })

        # Checkpoint
        save_interval = cfg["logging"].get("save_interval", 50)
        if update % save_interval == 0:
            ckpt_path = ckpt_dir / f"ppo_step{global_step}.pt"
            agent.save(str(ckpt_path))
            print(f"  Saved checkpoint → {ckpt_path}")

    # Done
    envs.close()
    agent.save(str(ckpt_dir / "ppo_final.pt"))
    if use_wandb:
        import wandb
        wandb.finish()
    print(f"\nTraining complete. Steps: {global_step:,}")


if __name__ == "__main__":
    main()