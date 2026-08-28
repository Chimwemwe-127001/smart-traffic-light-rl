"""Integration tests for the SUMO environment. Skipped if SUMO is not installed.

They check the traffic rules every controller must obey, straight from the
per-second log of what the signal showed.

    python -m pytest tests
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip('sumo')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic_rl.baselines import FixedTime, RandomPolicy
from traffic_rl.env import MAX_GREEN_S, MIN_GREEN_S, YELLOW_S, TrafficEnv
from traffic_rl.runner import run_episode


def signal_runs(records, tls):
    """Lengths of consecutive seconds with the same signal state: (state, length) pairs.
    The first and last runs are cut by the episode boundary, so they are dropped."""
    states = [r['green'] for r in records if r['tls'] == tls]
    runs, start = [], 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            runs.append((states[start], i - start))
            start = i
    return runs[1:-1]


def random_episode(scenario, seed, episode_s=600):
    env = TrafficEnv(scenario, program='agent', episode_s=episode_s, record=True)
    rng = np.random.default_rng(seed)
    ready, _ = env.reset(seed)
    while ready:
        ready, _, _ = env.step({tls: int(rng.integers(2)) for tls in ready})
    env.close()
    return env


@pytest.mark.parametrize('scenario', ['single', 'corridor'])
def test_agent_greens_respect_min_and_max_green(scenario):
    env = random_episode(scenario, seed=3)
    for tls in env.ids:
        runs = signal_runs(env.records, tls)
        greens = [n for state, n in runs if state in (0, 1)]
        yellows = [n for state, n in runs if state == -1]
        assert greens, 'no complete green phase observed'
        assert all(MIN_GREEN_S <= n <= MAX_GREEN_S for n in greens), greens
        assert all(n == YELLOW_S for n in yellows), yellows


def test_fixed_time_runs_the_42_second_plan():
    env = TrafficEnv('single', program='fixed', episode_s=400, record=True)
    env.reset(seed=1)
    env.run_to_end()
    env.close()
    runs = signal_runs(env.records, 'Node2')
    assert {n for state, n in runs if state in (0, 1)} == {42}
    assert {n for state, n in runs if state == -1} == {3}


class AlwaysArm:
    """Always asks for the same arm. Used to check that 'select' mode obeys the agent."""
    program = 'agent'

    def __init__(self, k):
        self.k = k

    def act(self, tls, obs, explore=False):
        return self.k

    def reward(self, tls, queues):
        return -queues[tls]

    def learn(self, *args):
        return None


def test_select_mode_serves_the_chosen_arm():
    env = TrafficEnv('four_way', program='agent', episode_s=600, record=True)
    ready, _ = env.reset(seed=4)
    applied = []
    while ready:
        ready, _, _ = env.step({tls: 2 for tls in ready})     # always ask for arm S
        applied += env.applied.values()
    env.close()
    greens = [r['green'] for r in env.records if r['green'] != -1]
    assert greens.count(2) / len(greens) > 0.6               # S gets most of the green
    # the only other choices are forced by the 60 s max green, and go to the next arm in cycle order (W)
    assert set(applied) <= {2, 3}
    runs = signal_runs(env.records, 'C')
    assert all(MIN_GREEN_S <= n <= MAX_GREEN_S for s, n in runs if s != -1)
    assert all(n == YELLOW_S for s, n in runs if s == -1)


def test_four_way_fixed_plan_gives_each_arm_30_seconds_in_turn():
    env = TrafficEnv('four_way', program='fixed', episode_s=400, record=True)
    env.reset(seed=1)
    env.run_to_end()
    env.close()
    runs = signal_runs(env.records, 'C')
    greens = [s for s, n in runs if s != -1]
    assert {n for s, n in runs if s != -1} == {30}
    assert greens[:4] in ([0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2])   # N, E, S, W order


@pytest.mark.parametrize('name', ['four_way', 'lusaka'])
def test_new_networks_drive_on_the_left(name):
    """Zambia drives on the left: on every arm, the incoming lanes sit on the driver's left."""
    import sumolib
    net_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'networks', name, f'{name}.net.xml')
    net = sumolib.net.readNet(net_file)
    for arm in 'NESW':
        inc, out = net.getEdge(f'{arm}2C'), net.getEdge(f'C2{arm}')
        (x1, y1), (x2, y2) = inc.getFromNode().getCoord(), inc.getToNode().getCoord()
        lane_in, lane_out = inc.getLanes()[0].getShape(), out.getLanes()[0].getShape()
        (xi, yi), (xo, yo) = lane_in[len(lane_in) // 2], lane_out[len(lane_out) // 2]
        cross = (x2 - x1) * (yi - yo) - (y2 - y1) * (xi - xo)     # > 0: incoming lane is left of travel
        assert cross > 0, f'{name} arm {arm} drives on the right'


def test_same_seed_gives_identical_results():
    a = run_episode('single', RandomPolicy(seed=5), seed=11, episode_s=300)
    b = run_episode('single', RandomPolicy(seed=5), seed=11, episode_s=300)
    del a['td_loss'], b['td_loss']          # NaN for a policy that does not learn, and NaN != NaN
    assert a == b


def test_different_seeds_give_different_traffic():
    a = run_episode('single', FixedTime(), seed=1, episode_s=300)
    b = run_episode('single', FixedTime(), seed=2, episode_s=300)
    assert a['throughput'] != b['throughput'] or a['avg_wait_s'] != b['avg_wait_s']
