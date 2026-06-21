#!/usr/bin/env python
"""
Curriculum orchestrator for AV-RL.
==================================
Trains ONE seed through difficulty phases, each resuming from the previous
phase's checkpoint. The agent transfers everything it learned in the easy
setting (steer, reach goals, stay on road) and then learns traffic avoidance
on top — far more sample-efficient and stable than throwing full difficulty
at a fresh policy.

This isn't a guess: a fresh policy at full difficulty crashed ~80% and never
arrived, while the SAME reward in an easy setting (no traffic, short maps)
reached goals within 2M steps (15% train / 35% held-out arrival). So we let
the agent master the easy problem first, then ramp difficulty.

Mechanics — each phase is a separate `train.py` process:
  * difficulty set via --traffic-density / --map
  * CUMULATIVE step target via --steps, because train.py's loop is
    `while global_step < total_steps`. Phase 2 trains FROM phase 1's end
    TO its own (larger) target.
  * --resume on every phase after the first. train.py restores network
    weights + Adam optimizer state, and recovers global_step by parsing it
    from the latest checkpoint filename — so phase 2 continues, it does not
    overshoot or restart.

Each seed pins to one GPU via CUDA_VISIBLE_DEVICES, so sweep.py (or a shell
loop) can run 4 seeds in parallel on the 4-GPU box, one per GPU.

Usage:
  python curriculum.py --seed 0 --gpu 0
  python curriculum.py --seed 0 --gpu 0 --smoke     # cheap end-to-end test
  python curriculum.py --seed 0 --dry-run           # print commands, run nothing
"""
import argparse
import os
import subprocess
import sys
import time


# Full curriculum — cumulative step targets. ~15M steps/seed total.
PHASES = [
    {"name": "easy",   "traffic_density": 0.0, "map": 2, "until_step": 5_000_000},
    {"name": "medium", "traffic_density": 0.1, "map": 3, "until_step": 10_000_000},
    {"name": "hard",   "traffic_density": 0.2, "map": 4, "until_step": 15_000_000},
]

# Smoke curriculum — tiny but each phase still crosses a checkpoint boundary
# (checkpoints land every 409,600 steps), so --resume is genuinely exercised
# phase-to-phase. ~1.5M steps total, a few dollars at most.
SMOKE_PHASES = [
    {"name": "easy",   "traffic_density": 0.0, "map": 2, "until_step": 500_000},
    {"name": "medium", "traffic_density": 0.1, "map": 3, "until_step": 1_000_000},
    {"name": "hard",   "traffic_density": 0.2, "map": 4, "until_step": 1_500_000},
]


def build_cmd(seed, phase, is_first):
    cmd = [
        sys.executable, "train.py",
        "--full", "--seed", str(seed),
        "--no-wandb",
        "--steps", str(phase["until_step"]),
        "--traffic-density", str(phase["traffic_density"]),
        "--map", str(phase["map"]),
    ]
    if not is_first:
        cmd.append("--resume")
    return cmd


def run_phase(seed, phase, is_first, gpu, dry_run):
    cmd = build_cmd(seed, phase, is_first)
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Stream child stdout live (train.py block-buffers under redirection
    # otherwise — same issue we hit with nohup).
    env["PYTHONUNBUFFERED"] = "1"

    banner = (f"[CURRICULUM seed={seed}] PHASE '{phase['name']}'  "
              f"traffic={phase['traffic_density']} map={phase['map']} "
              f"-> until step {phase['until_step']:,}"
              f"{'  (resume)' if not is_first else '  (fresh)'}"
              f"{f'  GPU={gpu}' if gpu is not None else ''}")
    print("=" * 78)
    print(banner)
    print("=" * 78)
    print("  $ " + " ".join(cmd), flush=True)

    if dry_run:
        print("  [dry-run] not executed.")
        return

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    dt = (time.time() - t0) / 60.0
    if result.returncode != 0:
        print(f"[CURRICULUM seed={seed}] PHASE '{phase['name']}' FAILED "
              f"(exit {result.returncode}) after {dt:.1f} min — stopping curriculum.")
        sys.exit(result.returncode)
    print(f"[CURRICULUM seed={seed}] PHASE '{phase['name']}' complete in {dt:.1f} min.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None,
                        help="Physical GPU index to pin this seed to (sets "
                             "CUDA_VISIBLE_DEVICES for the train.py subprocesses)")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny phases (0.5M/1M/1.5M) to validate the pipeline cheaply")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the per-phase commands without running them")
    args = parser.parse_args()

    phases = SMOKE_PHASES if args.smoke else PHASES
    mode = "SMOKE" if args.smoke else "FULL"
    print(f"Curriculum [{mode}] | seed={args.seed} | "
          f"gpu={args.gpu if args.gpu is not None else 'default'} | "
          f"{len(phases)} phases | final target "
          f"{phases[-1]['until_step']:,} steps")

    t0 = time.time()
    for i, phase in enumerate(phases):
        run_phase(args.seed, phase, is_first=(i == 0), gpu=args.gpu,
                  dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\n[CURRICULUM seed={args.seed}] ALL {len(phases)} PHASES COMPLETE "
              f"in {(time.time()-t0)/60:.1f} min.")
        print(f"Final model: checkpoints/seed_{args.seed}/ppo_final.pt")


if __name__ == "__main__":
    main()