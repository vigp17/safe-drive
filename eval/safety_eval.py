"""
Safety Evaluation Harness — AV-RL
===================================
Runs a trained PPO policy on held-out scenarios and reports
safety KPIs that read like a real ADAS V&V report:

  - Collision rate (%)
  - Route completion rate (%)
  - Time-to-collision (TTC) distribution
  - Generalization gap (train vs held-out)

Usage:
    python eval/safety_eval.py --checkpoint checkpoints/ppo_final.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.driving_env import DrivingEnv
from ppo.ppo import PPOAgent


# ------------------------------------------------------------------
# TTC estimation (simplified, lidar-based)
# ------------------------------------------------------------------

def estimate_ttc(obs: np.ndarray, speed: float, lidar_range: float = 50.0) -> float:
    """
    Rough TTC from the first 240 lidar channels.
    Real V&V would use radar/camera fusion; this gives a proxy.
    """
    lidar = obs[:240]
    # Forward sector: channels roughly covering ±30° ahead
    forward = lidar[100:140]
    min_dist = float(np.min(forward)) * lidar_range  # normalised → metres
    if speed < 0.5:
        return float("inf")
    return min_dist / speed


# ------------------------------------------------------------------
# Single episode rollout
# ------------------------------------------------------------------

def run_episode(env: DrivingEnv, agent: PPOAgent, device: torch.device, max_steps: int = 1000):
    obs, info = env.reset()
    ep_reward = 0.0
    ttc_values = []
    crashed = False
    arrived = False

    for _ in range(max_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.net.get_action_and_value(obs_t)
        act_np = action.cpu().numpy()[0].clip(-1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(act_np)
        ep_reward += reward

        speed = float(info.get("speed", 0.0))
        ttc_values.append(estimate_ttc(obs, speed))

        if info.get("crash", False) or info.get("crash_vehicle", False):
            crashed = True
        if info.get("arrive_dest", False):
            arrived = True

        if terminated or truncated:
            break

    route_completion = float(info.get("route_completion", 0.0))
    return {
        "reward": ep_reward,
        "crashed": crashed,
        "arrived": arrived,
        "route_completion": route_completion,
        "ttc_values": ttc_values,
    }


# ------------------------------------------------------------------
# Full evaluation
# ------------------------------------------------------------------

def evaluate(
    agent: PPOAgent,
    device: torch.device,
    env_config: dict,
    seed_range: range,
    label: str = "eval",
) -> dict:
    results = []
    print(f"\n[{label}] Running {len(seed_range)} episodes...")

    for seed in seed_range:
        cfg = dict(env_config)
        cfg["start_seed"] = seed
        env = DrivingEnv(config=cfg)
        ep = run_episode(env, agent, device)
        results.append(ep)
        env.close()

        status = "CRASH" if ep["crashed"] else ("DEST" if ep["arrived"] else "---")
        print(
            f"  seed={seed:4d} | "
            f"ret={ep['reward']:7.2f} | "
            f"route={ep['route_completion']:.2f} | "
            f"{status}"
        )

    collision_rate = np.mean([r["crashed"] for r in results]) * 100
    arrival_rate = np.mean([r["arrived"] for r in results]) * 100
    mean_route = np.mean([r["route_completion"] for r in results]) * 100
    mean_reward = np.mean([r["reward"] for r in results])

    all_ttc = [t for r in results for t in r["ttc_values"] if t < 100.0]

    print(f"\n{'─'*50}")
    print(f"[{label}] SAFETY KPI REPORT")
    print(f"{'─'*50}")
    print(f"  Episodes         : {len(results)}")
    print(f"  Collision rate   : {collision_rate:.1f}%")
    print(f"  Arrival rate     : {arrival_rate:.1f}%")
    print(f"  Route completion : {mean_route:.1f}%")
    print(f"  Mean reward      : {mean_reward:.2f}")
    if all_ttc:
        print(f"  TTC p10/p50/p90  : {np.percentile(all_ttc,10):.1f}s / "
              f"{np.percentile(all_ttc,50):.1f}s / "
              f"{np.percentile(all_ttc,90):.1f}s")
    print(f"{'─'*50}")

    return {
        "collision_rate": collision_rate,
        "arrival_rate": arrival_rate,
        "mean_route_completion": mean_route,
        "mean_reward": mean_reward,
        "ttc": all_ttc,
    }


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint (e.g. checkpoints/ppo_final.pt)")
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--n-train", type=int, default=20,
                        help="Episodes on training seed range (seeds 0–N)")
    parser.add_argument("--n-heldout", type=int, default=20,
                        help="Episodes on held-out seed range (seeds 500–500+N)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    env_config = cfg["env"]
    env_config["use_render"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load a single env just to get obs/act dims
    tmp_env = DrivingEnv(config=dict(env_config))
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    tmp_env.close()

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)
    agent.load(args.checkpoint)
    agent.net.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Train-distribution evaluation
    train_results = evaluate(
        agent, device, env_config,
        seed_range=range(0, args.n_train),
        label="TRAIN-DIST",
    )

    # Held-out evaluation (seeds the agent never trained on)
    heldout_results = evaluate(
        agent, device, env_config,
        seed_range=range(500, 500 + args.n_heldout),
        label="HELD-OUT",
    )

    # Generalization gap
    gen_gap = train_results["mean_route_completion"] - heldout_results["mean_route_completion"]
    print(f"\nGeneralization gap (route completion): {gen_gap:+.1f}%")
    print("(closer to 0 = better generalization)\n")


if __name__ == "__main__":
    main()