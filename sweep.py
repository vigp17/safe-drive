"""
Multi-seed sweep launcher — auto-detects GPU count.

Launches one independent PPO run per visible GPU.
Each run uses a different seed -> proper confidence intervals.

Usage:
    python sweep.py

Requires:
    - At least 1 CUDA GPU visible
    - W&B logged in: wandb login
"""
import os
import subprocess
import sys
import time

import torch


def main():
    # Detect GPU count at runtime instead of hardcoding it. The original
    # version hardcoded n_gpus=8 for an 8x V100 box; when run on a 4-GPU
    # box it still launched 8 processes, and seeds 4-7 silently fell back
    # to CPU (no error — torch.cuda.is_available() per device index just
    # returned False) and ran for 24+ hours before being caught manually.
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("ERROR: No CUDA GPUs detected. Aborting — refusing to launch "
              "CPU-fallback training, which is what caused the runaway "
              "24+ hour processes last time.")
        sys.exit(1)

    seeds = list(range(n_gpus))
    procs = []

    print(f"Detected {n_gpus} GPU(s). Launching {n_gpus} seeds, one per GPU...")

    for seed in seeds:
        cmd = [
            sys.executable, "train.py",
            "--full",
            "--seed", str(seed),
            "--resume",   # Spot-safe: if relaunched after an interruption,
                          # each seed continues from its latest checkpoint
                          # (no-op on a fresh run with no checkpoints).
        ]
        proc_env = os.environ.copy()
        proc_env["CUDA_VISIBLE_DEVICES"] = str(seed)

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
    print("Checkpoints are in checkpoints/seed_<N>/ for each seed.")


if __name__ == "__main__":
    main()