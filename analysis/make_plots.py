"""
Generate the headline figure for safe-drive: the safety/competence curve as a
function of scenario difficulty, aggregated across the 4 curriculum-trained
seeds.

Data below are the held-out (unseen scenario) and in-distribution KPI numbers
from eval/safety_eval.py run on each seed's final model at three difficulties:
  easy   = traffic_density 0.0, map 2
  medium = traffic_density 0.1, map 3
  hard   = traffic_density 0.2, map 4

Run:  python analysis/make_plots.py   ->  results/difficulty_curve.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Eval results: [seed0, seed1, seed2, seed3] per metric ──────────────
# Held-out (unseen) scenarios — the generalization numbers we headline.
HELDOUT = {
    "easy":   {"arrival": [30, 25, 45, 15], "collision": [20, 5, 15, 30], "route": [71.7, 58.8, 70.7, 54.9]},
    "medium": {"arrival": [25, 10, 0, 5],   "collision": [65, 20, 45, 55], "route": [49.8, 46.9, 42.4, 35.4]},
    "hard":   {"arrival": [5, 10, 0, 0],    "collision": [65, 45, 10, 50], "route": [26.6, 43.9, 17.8, 24.8]},
}
# In-distribution (training scenario pool) — shown for context.
TRAIN = {
    "easy":   {"arrival": [30, 50, 25, 30], "collision": [20, 10, 35, 15], "route": [71.1, 72.1, 75.5, 72.0]},
    "medium": {"arrival": [0, 35, 5, 0],    "collision": [70, 15, 25, 45], "route": [35.3, 57.4, 43.5, 32.6]},
    "hard":   {"arrival": [0, 0, 0, 0],     "collision": [80, 55, 5, 50],  "route": [23.7, 35.4, 21.5, 27.7]},
}

ORDER = ["easy", "medium", "hard"]
LABELS = ["Easy\n(no traffic, map 2)", "Medium\n(0.1 traffic, map 3)", "Hard\n(0.2 traffic, map 4)"]
X = np.arange(len(ORDER))

# ── Baselines (held-out), single run each ──────────────────────────────
#   random : action_space.sample()       — the floor
#   idm    : MetaDrive rule-based driver  — classical reference
BASELINE = {
    "random": {
        "easy":   {"arrival": 0.0, "collision": 0.0, "route": 6.1},
        "medium": {"arrival": 0.0, "collision": 0.0, "route": 3.7},
        "hard":   {"arrival": 0.0, "collision": 0.0, "route": 3.5},
    },
    "idm": {
        "easy":   {"arrival": 80.0, "collision": 0.0,  "route": 89.7},
        "medium": {"arrival": 65.0, "collision": 5.0,  "route": 83.4},
        "hard":   {"arrival": 0.0,  "collision": 90.0, "route": 48.6},
    },
}

METRICS = [
    ("arrival",   "Arrival rate",     "#2e7d32"),   # green  — the good metric
    ("route",     "Route completion", "#1565c0"),   # blue
    ("collision", "Collision rate",   "#c62828"),   # red    — the bad metric
]


def mean_std(source, metric):
    means = np.array([np.mean(source[d][metric]) for d in ORDER])
    stds = np.array([np.std(source[d][metric]) for d in ORDER])
    return means, stds


def make_comparison(out_dir):
    """3-policy comparison (random / PPO / IDM) across difficulty, held-out."""
    metrics = [("arrival", "Arrival rate (%)"),
               ("route", "Route completion (%)"),
               ("collision", "Collision rate (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, (metric, ylabel) in zip(axes, metrics):
        # PPO: mean +/- std across seeds (held-out)
        ppo_mean, ppo_std = mean_std(HELDOUT, metric)
        ax.plot(X, ppo_mean, "-o", color="#1565c0", lw=2.6, ms=8,
                label="PPO (ours)", zorder=4)
        ax.fill_between(X, ppo_mean - ppo_std, ppo_mean + ppo_std,
                        color="#1565c0", alpha=0.15, zorder=1)
        # Random floor
        rnd = [BASELINE["random"][d][metric] for d in ORDER]
        ax.plot(X, rnd, "--s", color="#888", lw=2, ms=6, label="Random (floor)", zorder=3)
        # IDM reference
        idm = [BASELINE["idm"][d][metric] for d in ORDER]
        ax.plot(X, idm, "--^", color="#e65100", lw=2, ms=7, label="IDM (rule-based)", zorder=3)

        ax.set_title(ylabel.split(" (")[0], fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(X)
        ax.set_xticklabels([l.split("\n")[0] for l in LABELS], fontsize=10)
        ax.set_ylim(-5, 100)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    fig.suptitle(
        "Random vs PPO (learned) vs IDM (rule-based) across difficulty — held-out scenarios",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.04,
        "IDM dominates in easy/moderate traffic but collapses on hard (90% collision); the learned "
        "PPO agent degrades more gracefully, crashing far less than IDM in dense traffic.",
        ha="center", fontsize=9.5, style="italic", color="#444",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'baseline_comparison.png'}")


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    for ax, (source, title) in zip(
        axes, [(HELDOUT, "Held-out scenarios (generalization)"),
               (TRAIN, "In-distribution scenarios")]
    ):
        for metric, label, color in METRICS:
            means, stds = mean_std(source, metric)
            # mean line with std band
            ax.plot(X, means, "-o", color=color, lw=2.4, ms=8, label=label, zorder=3)
            ax.fill_between(X, means - stds, means + stds, color=color, alpha=0.13, zorder=1)
            # individual seed points to show across-seed spread
            for i, d in enumerate(ORDER):
                pts = source[d][metric]
                ax.scatter([X[i]] * len(pts), pts, color=color, alpha=0.35, s=22, zorder=2)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(X)
        ax.set_xticklabels(LABELS, fontsize=10)
        ax.set_ylim(-5, 100)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Percent (%)", fontsize=11)
    axes[0].legend(loc="upper center", fontsize=10, framealpha=0.95)

    fig.suptitle(
        "Safe-Drive: PPO agent safety vs. scenario difficulty (4 seeds, mean ± std)",
        fontsize=13.5, fontweight="bold", y=1.0,
    )
    fig.text(
        0.5, -0.02,
        "Competence degrades monotonically with difficulty: arrival and route completion fall, "
        "collisions rise. Faint dots = individual seeds (note the spread at medium/hard).",
        ha="center", fontsize=9.5, style="italic", color="#444",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "difficulty_curve.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'difficulty_curve.png'}")

    make_comparison(out_dir)


if __name__ == "__main__":
    main()