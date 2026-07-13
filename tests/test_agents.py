"""Unit tests for the learning code. No SUMO needed.

    python -m pytest tests
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic_rl.dqn import MLP, ReplayBuffer, DQN
from traffic_rl.metrics import bootstrap_ci, paired_difference
from traffic_rl.multi_agent import CoordinatedQLearning
from traffic_rl.q_learning import QLearning
from traffic_rl.state import count_bin, discrete_state, feature_vector, DEFAULT_CONFIG


def obs(eb=0, sb=0, green=0, green_time=15, tls='Node2'):
    lanes = np.array([eb / 3] * 3 + [sb / 3] * 3)
    return {tls: {'lanes': lanes, 'EB': eb, 'SB': sb, 'green': green,
                  'green_time': green_time, 'queue': eb + sb}}


def test_count_bins_follow_edges():
    edges = [2, 4, 7, 11]
    assert [count_bin(n, edges) for n in (0, 1, 2, 3, 4, 6, 7, 10, 11, 50)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_discrete_state_and_features():
    o = obs(eb=6, sb=0, green=1, green_time=45)['Node2']
    assert discrete_state(o, DEFAULT_CONFIG) == (2, 0, 1, 2)
    f = feature_vector(o, DEFAULT_CONFIG)
    assert f.shape == (9,)
    assert f[6:8].tolist() == [0.0, 1.0] and f[8] == 0.75


def test_q_update_matches_formula():
    agent = QLearning(alpha=0.5, gamma=0.9)
    o, o2 = obs(eb=5), obs(sb=5)
    s, s2 = agent.state('Node2', o), agent.state('Node2', o2)
    agent.q('Node2', s2)[:] = [1.0, 3.0]
    agent.learn('Node2', o, 1, {'Node2': 4.0}, o2)
    # Q = 0 + 0.5 * (-4 + 0.9 * 3 - 0) = -0.65
    assert np.isclose(agent.q('Node2', s)[1], -0.65)
    assert agent.q('Node2', s)[0] == 0.0


def test_greedy_action_and_save_load(tmp_path):
    agent = QLearning()
    o = obs(eb=8)
    agent.q('Node2', agent.state('Node2', o))[:] = [-5.0, -1.0]
    assert agent.act('Node2', o) == 1
    path = tmp_path / 'q.json'
    agent.save(path)
    assert QLearning().load(path).act('Node2', o) == 1


def test_coordinated_state_and_reward_use_neighbor():
    agent = CoordinatedQLearning('corridor')
    o = {**obs(eb=3, tls='Node2'), **obs(sb=12, green=1, tls='Node5')}
    assert agent.state('Node2', o)[-2:] == (1, 4)
    assert agent.reward('Node2', {'Node2': 2.0, 'Node5': 6.0}) == -2.0 - 0.5 * 6.0


def test_mlp_gradient_matches_finite_differences():
    net = MLP(n_in=4, n_hidden=5, n_out=2, lr=0.0, seed=1)
    x = np.random.default_rng(0).normal(size=(3, 4))
    a, y = np.array([0, 1, 1]), np.array([0.1, -0.2, 0.05])

    def loss():
        out = net.predict(x)
        err = out[np.arange(3), a] - y
        return np.mean(np.where(np.abs(err) < 1, 0.5 * err ** 2, np.abs(err) - 0.5))

    W1 = net.params[0]
    eps = 1e-6
    W1[0, 0] += eps; up = loss()
    W1[0, 0] -= 2 * eps; down = loss()
    W1[0, 0] += eps
    numeric = (up - down) / (2 * eps)

    out, (xx, z1, a1, z2, a2) = net.forward(x)
    err = out[np.arange(3), a] - y
    d3 = np.zeros_like(out); d3[np.arange(3), a] = np.clip(err, -1, 1) / 3
    d2 = (d3 @ net.params[4].T) * (z2 > 0)
    d1 = (d2 @ net.params[2].T) * (z1 > 0)
    analytic = (xx.T @ d1)[0, 0]
    assert np.isclose(numeric, analytic, atol=1e-7)


def test_mlp_can_fit_a_target():
    net = MLP(n_in=3, n_hidden=16, n_out=2, lr=1e-2, seed=0)
    x = np.random.default_rng(1).uniform(size=(64, 3))
    a = np.zeros(64, dtype=int)
    y = x.sum(axis=1) * 0.5
    first = net.train_step(x, a, y)
    for _ in range(300):
        last = net.train_step(x, a, y)
    assert last < first * 0.1


def test_replay_buffer_wraps_around():
    buf = ReplayBuffer(capacity=5, n_features=2)
    for i in range(8):
        buf.add([i, i], i % 2, -i, [i + 1, i + 1])
    assert buf.size == 5
    assert set(buf.r) == {-3, -4, -5, -6, -7}
    s, a, r, s2 = buf.sample(10)
    assert s.shape == (10, 2) and np.all(s2 == s + 1)


def test_dqn_learns_after_warmup():
    agent = DQN(seed=0)
    agent.WARMUP = 10
    o, o2 = obs(eb=4), obs(eb=2)
    losses = [agent.learn('Node2', o, 0, {'Node2': 3.0}, o2) for _ in range(20)]
    assert losses[0] is None and losses[-1] is not None


def test_bootstrap_and_paired_difference():
    m, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and lo < m < hi
    d, lo, hi = paired_difference([1, 2, 3], [2, 3, 4])
    assert d == -1 and lo == hi == -1
