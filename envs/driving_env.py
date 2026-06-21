"""
MetaDrive Gymnasium wrapper for AV-RL training.

Wraps MetaDriveEnv with:
- Reward shaping (collision penalty, destination bonus)
- Episode-level safety KPI tracking (collision rate, route completion)
- Clean Gymnasium interface for vectorized training
"""
import gymnasium as gym
import numpy as np
from metadrive.envs import MetaDriveEnv as _MetaDriveEnv


class DrivingEnv(gym.Env):
    """
    Thin wrapper around MetaDriveEnv that stays fully Gymnasium-compatible.

    Default observation: lidar-based vector of shape (259,)
    Default action:      continuous Box([-1,-1], [1,1]) → [steering, acceleration]
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: dict = None, render_mode: str = None):
        base_config = {
            "use_render": False,         # no display window
            "num_scenarios": 100,        # procedural map pool size
            "traffic_density": 0.1,      # fraction of road filled with vehicles
            "map": 4,                    # road complexity (number of blocks)
            "start_seed": 0,
            "accident_prob": 0.0,        # no pre-placed accidents during training
            "decision_repeat": 5,        # physics steps per RL action
        }
        if config:
            base_config.update(config)

        self._env = _MetaDriveEnv(config=base_config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = render_mode

        # Episode-level safety tracking
        self._ep_crashes = 0
        self._ep_steps = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self._ep_crashes = 0
        self._ep_steps = 0
        # MetaDrive 0.4.3 manages scenario cycling internally via start_seed
        # in config — do not pass seed directly or it breaks agent init
        obs, info = self._env.reset()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._ep_steps += 1

        # Aggregate crash signal — handle both old and new MetaDrive keys
        crashed = (
            info.get("crash", False)
            or info.get("crash_vehicle", False)
            or info.get("crash_object", False)
            or info.get("crash_building", False)
        )
        if crashed:
            self._ep_crashes += 1

        shaped = self._shape_reward(reward, info, crashed, terminated, truncated)

        # Inject safety KPIs into info for logging
        info["ep_collision_rate"] = self._ep_crashes / max(1, self._ep_steps)
        info["ep_route_completion"] = info.get("route_completion", 0.0)

        return obs, shaped, terminated, truncated, info

    def _shape_reward(self, reward: float, info: dict, crashed: bool,
                      terminated: bool, truncated: bool) -> float:
        """
        On top of MetaDrive's dense reward (progress + speed, which already
        includes its own out-of-road penalty):
          -15.0   on crash
          +200.0  on reaching the destination (the jackpot — was +10.0, only
                  ~4% of episode reward, which is why arrival rate was 0%)
          +30.0 * route_completion  at episode end

        The completion bonus replaces the flat per-step time penalty used in
        the previous version. A per-step penalty can backfire: if leaving the
        road ends the episode and stops the penalty, the agent can learn to
        drive off early to "stop the bleeding." Rewarding route_completion at
        episode end instead pulls the agent toward the goal with no such
        perverse incentive — ending early means low completion means a small
        bonus.
        """
        r = float(reward)
        if crashed:
            r -= 15.0
        if info.get("arrive_dest", False):
            r += 200.0
        if terminated or truncated:
            r += 30.0 * float(info.get("route_completion", 0.0))
        return r

    def close(self):
        self._env.close()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.render(mode="rgb_array")
        return None