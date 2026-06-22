"""
Baseline policies for safe-drive
================================
Context for the trained PPO agent's numbers. Two references, scored on the
exact same safety KPIs (collision / arrival / route completion) at the same
difficulties, so the curve has a floor and a classical reference:

  random : samples the action space uniformly. The floor — anything a learned
           policy does should sit well above this.
  idm    : MetaDrive's built-in rule-based driver (Intelligent Driver Model
           lane-follow + car-follow). A hand-engineered reference, i.e. "how
           well does a classical non-learned controller do here?"

Usage:
  python eval/baselines.py --policy random --traffic-density 0.0 --map 2
  python eval/baselines.py --policy idm    --traffic-density 0.2 --map 4

The KPI report format matches eval/safety_eval.py so results drop straight
into the same comparison table / plot.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.driving_env import DrivingEnv


def run_episode_random(env, max_steps=1000):
    env.reset()
    crashed = arrived = False
    for _ in range(max_steps):
        action = env.action_space.sample()          # uniform random in [-1, 1]^2
        _, _, terminated, truncated, info = env.step(action)
        if info.get("crash", False) or info.get("crash_vehicle", False):
            crashed = True
        if info.get("arrive_dest", False):
            arrived = True
        if terminated or truncated:
            break
    return crashed, arrived, float(info.get("route_completion", 0.0))


def run_episode_idm(env, max_steps=1000):
    # With agent_policy=IDMPolicy set in the env config, MetaDrive drives the
    # ego with its rule-based controller and ignores the action we pass. We
    # still must call step() with a placeholder action of the right shape.
    env.reset()
    crashed = arrived = False
    dummy = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(max_steps):
        _, _, terminated, truncated, info = env.step(dummy)
        if info.get("crash", False) or info.get("crash_vehicle", False):
            crashed = True
        if info.get("arrive_dest", False):
            arrived = True
        if terminated or truncated:
            break
    return crashed, arrived, float(info.get("route_completion", 0.0))


def evaluate(policy, env_config, seed_range, label, idm=False):
    results = []
    print(f"\n[{label}] Running {len(seed_range)} episodes...")
    for seed in seed_range:
        cfg = dict(env_config)
        cfg["start_seed"] = seed
        env = DrivingEnv(config=cfg)
        if idm:
            crashed, arrived, route = run_episode_idm(env)
        else:
            crashed, arrived, route = run_episode_random(env)
        env.close()
        results.append((crashed, arrived, route))
        status = "CRASH" if crashed else ("DEST" if arrived else "---")
        print(f"  seed={seed:4d} | route={route:.2f} | {status}")

    coll = np.mean([r[0] for r in results]) * 100
    arr = np.mean([r[1] for r in results]) * 100
    rte = np.mean([r[2] for r in results]) * 100
    print(f"\n{'-'*50}")
    print(f"[{label}] BASELINE KPI REPORT ({policy})")
    print(f"{'-'*50}")
    print(f"  Episodes         : {len(results)}")
    print(f"  Collision rate   : {coll:.1f}%")
    print(f"  Arrival rate     : {arr:.1f}%")
    print(f"  Route completion : {rte:.1f}%")
    print(f"{'-'*50}")
    return {"collision": coll, "arrival": arr, "route": rte}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "idm"], required=True)
    parser.add_argument("--traffic-density", type=float, default=0.2)
    parser.add_argument("--map", type=int, default=4)
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-heldout", type=int, default=20)
    args = parser.parse_args()

    env_config = {
        "use_render": False,
        "traffic_density": args.traffic_density,
        "map": args.map,
        "num_scenarios": args.num_scenarios,
        "accident_prob": 0.0,
        "decision_repeat": 5,
    }

    idm = args.policy == "idm"
    if idm:
        # Rule-based ego controller. If this import or the agent_policy wiring
        # errors on your MetaDrive build, paste the traceback — the API for
        # ego-IDM shifts slightly between MetaDrive versions.
        from metadrive.policy.idm_policy import IDMPolicy
        env_config["agent_policy"] = IDMPolicy

    print(f"[baseline] policy={args.policy} | traffic={args.traffic_density} "
          f"map={args.map}")

    train = evaluate(args.policy, env_config, range(0, args.n_train),
                     "TRAIN-DIST", idm=idm)
    held = evaluate(args.policy, env_config, range(500, 500 + args.n_heldout),
                    "HELD-OUT", idm=idm)

    print(f"\nGeneralization gap (route): "
          f"{train['route'] - held['route']:+.1f}%")


if __name__ == "__main__":
    main()