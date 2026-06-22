# Model Card — safe-drive PPO policy

## Model description

A reinforcement-learning driving policy trained with PPO in the MetaDrive
simulator. The model maps a 259-dimensional observation (ego kinematics,
navigation reference, and a lidar distance ring) to a 2-dimensional continuous
control `[steering, acceleration]`, each in `[-1, 1]`.

- **Architecture:** MLP actor-critic. Shared backbone `259 → 256 → 256` with Tanh
  activations; a linear actor-mean head (2 outputs) with a learnable per-dimension
  log-std (clamped to `[-5, 0.5]`); a linear scalar value head. ~140K parameters.
- **Algorithm:** PPO with clipped surrogate objective, GAE (γ=0.99, λ=0.95),
  entropy bonus.
- **Training:** ~15M environment steps per seed via a three-stage difficulty
  curriculum (easy → medium → hard), 16 parallel environments per seed, 4 seeds.
- **Checkpoints:** `checkpoints/seed_{0..3}/ppo_final.pt`. Seed 1 is the flagship
  (strongest in the easy/moderate regime).
- **Deployment artifact:** TorchScript export (`export/export_torchscript.py`)
  exposes the deterministic inference path only.

## Intended use

Research and portfolio demonstration of distributed RL and safety evaluation
methodology. The model drives a simulated vehicle in MetaDrive scenarios.

**Out of scope:** this is a simulator policy and is **not** suitable for any
real-world vehicle control, safety-critical deployment, or decision-making about
physical systems. It has no perception stack, no real sensor model, and no
validation against real driving data.

## Training and evaluation data

Procedurally generated MetaDrive scenarios. Difficulty is parameterized by
traffic density (0.0 / 0.1 / 0.2) and map complexity (block count 2 / 3 / 4).
Evaluation uses two disjoint scenario pools — a training-distribution range and
a held-out range the policy never saw during training — to measure
generalization.

## Metrics

Reported on held-out scenarios (mean across 4 seeds):

| Difficulty | Arrival | Collision | Route completion |
|---|---|---|---|
| Easy   | 29% | 18% | 64% |
| Medium | 10% | 46% | 44% |
| Hard   | 4%  | 43% | 28% |

For context, on the same scenarios a random policy reaches 0% arrival
(~3–6% route), and MetaDrive's rule-based IDM driver reaches 80% / 65% / 0%
arrival across easy/medium/hard (collapsing to 90% collision on hard).

## Limitations and biases

- **Modest absolute performance.** The policy is meaningfully above a random
  floor but below a hand-engineered controller in easy/moderate conditions.
- **Robustness vs. peak trade-off.** Its one advantage over IDM is graceful
  degradation: far fewer collisions than IDM in dense traffic.
- **Seed instability.** At higher difficulty, independently trained seeds
  converge to qualitatively different (cautious vs reckless) behaviors.
- **Compute-bounded.** Trained under a fixed, low cloud-cost budget; not trained
  to convergence on the hard regime.
- **Simulation only.** No sim-to-real validity is claimed or implied.

## Ethical considerations

The model must not be used to control any physical vehicle or inform any
real-world safety decision. It exists to study and demonstrate learned-control
behavior and evaluation methodology in simulation.

## Reproducibility

Fixed seeds, version-pinned dependencies (`requirements.txt`), config-driven
hyperparameters and reward (`configs/ppo.yaml`), and a Dockerfile for a clean
environment. See `README.md` for commands.