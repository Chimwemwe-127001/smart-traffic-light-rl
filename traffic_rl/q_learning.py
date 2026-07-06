"""Tabular Q-learning (Watkins and Dayan, 1992).

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

One Q-table per intersection. On a single intersection this is plain
Q-learning. On the corridor it is "independent learners" (Tan, 1993): every
agent learns only from its own detectors and its own queue.
"""
import json

import numpy as np

from traffic_rl.baselines import queue_reward
from traffic_rl.state import discrete_state, load_config


class QLearning:
    program = 'agent'

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, seed=0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.cfg = load_config()
        self.rng = np.random.default_rng(seed)
        self.tables = {}                     # tls -> {state tuple: np.array([Q(keep), Q(switch)])}

    # -- what the agent sees and what it is rewarded for (overridden by the coordinated agent)

    def state(self, tls, obs):
        return discrete_state(obs[tls], self.cfg)

    def reward(self, tls, queues):
        return queue_reward(tls, queues)

    # -- the Q-learning algorithm

    def q(self, tls, s):
        table = self.tables.setdefault(tls, {})
        if s not in table:
            table[s] = np.zeros(2)
        return table[s]

    def act(self, tls, obs, explore=False):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        return int(np.argmax(self.q(tls, self.state(tls, obs))))

    def learn(self, tls, obs, action, queues, next_obs):
        s, s2 = self.state(tls, obs), self.state(tls, next_obs)
        r = self.reward(tls, queues)
        q = self.q(tls, s)
        td_error = r + self.gamma * np.max(self.q(tls, s2)) - q[action]
        q[action] += self.alpha * td_error
        return float(td_error ** 2)

    def n_states(self):
        return sum(len(t) for t in self.tables.values())

    # -- saving as readable JSON

    def save(self, path):
        data = {tls: {','.join(map(str, s)): q.tolist() for s, q in table.items()}
                for tls, table in self.tables.items()}
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.tables = {tls: {tuple(int(x) for x in k.split(',')): np.array(v) for k, v in table.items()}
                       for tls, table in data.items()}
        return self
