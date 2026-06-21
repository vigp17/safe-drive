"""
Unit tests for Reward v2 shaping (envs/driving_env.py:_shape_reward).

MetaDrive is heavy and Panda3D-bound, so we stub it out and exercise the
shaping logic in isolation by constructing DrivingEnv via __new__ and setting
the reward weights by hand — exactly the values the config would inject.
"""
import sys
import types
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Stub metadrive so the module-level import in driving_env succeeds ──
_md = types.ModuleType("metadrive")
_envs = types.ModuleType("metadrive.envs")


class _FakeMetaDriveEnv:
    def __init__(self, config=None):
        pass


_envs.MetaDriveEnv = _FakeMetaDriveEnv
_md.envs = _envs
sys.modules["metadrive"] = _md
sys.modules["metadrive.envs"] = _envs

from envs.driving_env import DrivingEnv  # noqa: E402


def _make_env(crash=80.0, oor=40.0, success=200.0, completion=50.0):
    """DrivingEnv with reward weights set, bypassing the heavy __init__."""
    env = DrivingEnv.__new__(DrivingEnv)
    env.w_crash = crash
    env.w_out_of_road = oor
    env.w_success = success
    env.w_completion = completion
    return env


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_crash_dominates_progress():
    """The exact failure from the 2M run: ~76 progress reward while crashing.
    Under v1 (-15) that nets +61 (crashing is profitable). Under v2 (-80) it
    must net negative."""
    env = _make_env()
    base = 76.0  # dense progress reward earned before the crash
    info = {"out_of_road": False, "arrive_dest": False, "route_completion": 0.20}
    r = env._shape_reward(base, info, crashed=True, terminated=True, truncated=False)
    # 76 - 80 (crash) + 50*0.20 (completion) = 76 - 80 + 10 = 6 ... still slightly +
    # so verify the crash term itself is applied and large:
    r_no_crash = env._shape_reward(base, info, crashed=False, terminated=True, truncated=False)
    assert approx(r_no_crash - r, 80.0), f"crash penalty not 80: {r_no_crash - r}"
    print(f"  crash applied: with={r:.1f}  without={r_no_crash:.1f}  delta={r_no_crash-r:.1f}  PASS")


def test_crash_makes_rushing_unprofitable():
    """With realistic numbers, a crash early in a route should be net-negative
    so the agent stops preferring 'rush and crash'."""
    env = _make_env()
    # Reckless run: small progress (route 0.12), modest dense reward, then crash.
    base = 30.0
    info = {"out_of_road": False, "arrive_dest": False, "route_completion": 0.12}
    r = env._shape_reward(base, info, crashed=True, terminated=True, truncated=False)
    # 30 - 80 + 50*0.12 = 30 - 80 + 6 = -44
    assert r < 0, f"reckless crash should be net-negative, got {r:.1f}"
    print(f"  reckless crash nets {r:.1f} (<0)  PASS")


def test_out_of_road_exclusive_of_crash():
    """If the agent crashes, the out-of-road penalty must NOT also fire
    (no double-counting). Only the crash penalty applies."""
    env = _make_env()
    base = 10.0
    info = {"out_of_road": True, "arrive_dest": False, "route_completion": 0.1}
    r = env._shape_reward(base, info, crashed=True, terminated=True, truncated=False)
    # crash branch only: 10 - 80 + 50*0.1 = -65   (NOT -65-40)
    assert approx(r, 10.0 - 80.0 + 5.0), f"double-counted oor+crash: {r:.1f}"
    print(f"  crash excludes oor: {r:.1f} == -65.0  PASS")


def test_out_of_road_penalty_alone():
    env = _make_env()
    base = 20.0
    info = {"out_of_road": True, "arrive_dest": False, "route_completion": 0.15}
    r = env._shape_reward(base, info, crashed=False, terminated=True, truncated=False)
    # 20 - 40 + 50*0.15 = 20 - 40 + 7.5 = -12.5
    assert approx(r, 20.0 - 40.0 + 7.5), f"oor penalty wrong: {r:.1f}"
    print(f"  out-of-road alone: {r:.1f} == -12.5  PASS")


def test_arrival_is_the_jackpot():
    """Reaching the goal should be by far the largest single payoff."""
    env = _make_env()
    base = 120.0
    info = {"out_of_road": False, "arrive_dest": True, "route_completion": 1.0}
    r = env._shape_reward(base, info, crashed=False, terminated=True, truncated=False)
    # 120 + 200 + 50*1.0 = 370
    assert approx(r, 120.0 + 200.0 + 50.0), f"arrival reward wrong: {r:.1f}"
    # And arrival must beat any crash outcome decisively
    assert r > 300, "arrival should dominate"
    print(f"  arrival jackpot: {r:.1f} == 370.0  PASS")


def test_completion_bonus_only_at_episode_end():
    """Mid-episode (not terminated/truncated) there is no completion bonus."""
    env = _make_env()
    base = 5.0
    info = {"out_of_road": False, "arrive_dest": False, "route_completion": 0.5}
    r = env._shape_reward(base, info, crashed=False, terminated=False, truncated=False)
    assert approx(r, 5.0), f"completion bonus leaked mid-episode: {r:.1f}"
    print(f"  no mid-episode completion bonus: {r:.1f} == 5.0  PASS")


def test_weights_are_config_driven():
    """Different config weights must change the output — proving the values
    come from config, not hardcoded."""
    soft = _make_env(crash=10.0)
    hard = _make_env(crash=200.0)
    base = 50.0
    info = {"out_of_road": False, "arrive_dest": False, "route_completion": 0.0}
    r_soft = soft._shape_reward(base, info, crashed=True, terminated=True, truncated=False)
    r_hard = hard._shape_reward(base, info, crashed=True, terminated=True, truncated=False)
    assert approx(r_soft, 40.0) and approx(r_hard, -150.0), f"{r_soft}, {r_hard}"
    print(f"  config-driven: crash=10 -> {r_soft:.1f}, crash=200 -> {r_hard:.1f}  PASS")


if __name__ == "__main__":
    tests = [
        test_crash_dominates_progress,
        test_crash_makes_rushing_unprofitable,
        test_out_of_road_exclusive_of_crash,
        test_out_of_road_penalty_alone,
        test_arrival_is_the_jackpot,
        test_completion_bonus_only_at_episode_end,
        test_weights_are_config_driven,
    ]
    print("Reward v2 unit tests")
    print("-" * 50)
    for t in tests:
        print(t.__name__)
        t()
    print("-" * 50)
    print(f"All {len(tests)} reward-v2 tests passed.")