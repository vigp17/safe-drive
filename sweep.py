"""
Multi-seed sweep launcher — AWS p3.16xlarge (8x V100)

Launches 8 independent PPO runs in parallel, one per GPU.
Each run uses a different seed → proper confidence intervals.

Usage (on AWS):
    python sweep.py

Requires:
    - 8 CUDA GPUs visible
    - W&B logged in: wandb login
"""
import subprocess
import sys
import time


def main():
    n_gpus = 8
    seeds = list(range(n_gpus))
    procs = []

    print(f"Launching {n_gpus} seeds across {n_gpus} GPUs...")

    for seed in seeds:
        env = {
            "CUDA_VISIBLE_DEVICES": str(seed),
        }
        cmd = [
            sys.executable, "train.py",
            "--full",
            "--seed", str(seed),
        ]
        import os
        proc_env = os.environ.copy()
        proc_env.update(env)

        proc = subprocess.Popen(cmd, env=proc_env)
        procs.append((seed, proc))
        print(f"  Launched seed={seed} on GPU {seed} (PID {proc.pid})")
        time.sleep(2)   # stagger startup to avoid race on MetaDrive assets

    print(f"\nAll {n_gpus} processes running. Waiting for completion...")

    for seed, proc in procs:
        ret = proc.wait()
        status = "OK" if ret == 0 else f"FAILED (code {ret})"
        print(f"  Seed {seed}: {status}")

    print("\nSweep complete. Check W&B for aggregated results.")


if __name__ == "__main__":
    main()