"""
Render a short top-down clip of a trained safe-drive policy.
============================================================
Runs the deterministic policy (mean action, no sampling) on a few easy
scenarios and saves a top-down GIF. Defaults to seed 1 — the flagship model
that drives best in the easy regime.

Usage:
    python demo/render_demo.py --checkpoint checkpoints/seed_1/ppo_final.pt
    python demo/render_demo.py --checkpoint checkpoints/seed_1/ppo_final.pt \
        --episodes 4 --traffic-density 0.0 --map 2 --out results/demo_seed1_easy.gif

Notes:
- Needs imageio:  pip install imageio
- MetaDrive's top-down render API shifts a little between versions. If the
  render call throws, paste the traceback — we'll adjust the kwargs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.driving_env import DrivingEnv
from ppo.ppo import PPOAgent


def deterministic_action(agent, obs, device):
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        features = agent.net.shared(obs_t)
        action = agent.net.actor_mean(features)
    return action.cpu().numpy()[0].clip(-1.0, 1.0)


def grab_topdown(env):
    """Return an RGB frame of the top-down view, or None if unavailable.

    Tries the common MetaDrive top-down render signatures so this survives
    minor version differences.
    """
    underlying = env._env
    for kwargs in (
        dict(mode="topdown", window=False, screen_size=(600, 600)),
        dict(mode="topdown", window=False),
        dict(mode="top_down", window=False),
    ):
        try:
            frame = underlying.render(**kwargs)
            if frame is not None:
                return np.asarray(frame)
        except TypeError:
            continue
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--traffic-density", type=float, default=0.0)
    p.add_argument("--map", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--out", default="results/demo.gif")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    import imageio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_config = {
        "use_render": False,
        "traffic_density": args.traffic_density,
        "map": args.map,
        "num_scenarios": 100,
        "accident_prob": 0.0,
        "decision_repeat": 5,
    }
    env = DrivingEnv(config=env_config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)
    agent.load(args.checkpoint)
    agent.net.eval()
    print(f"Loaded {args.checkpoint}")

    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        for _ in range(args.max_steps):
            action = deterministic_action(agent, obs, device)
            obs, _, terminated, truncated, info = env.step(action)
            frame = grab_topdown(env)
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                break
        outcome = ("arrived" if info.get("arrive_dest")
                   else "crashed" if info.get("crash") or info.get("crash_vehicle")
                   else "off-road/timeout")
        print(f"  episode {ep}: {outcome}  (route={info.get('route_completion', 0):.2f})")
    env.close()

    if not frames:
        print("\nNo frames captured — the top-down render API didn't return images. "
              "Paste this message and we'll adjust grab_topdown().")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"\nwrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()