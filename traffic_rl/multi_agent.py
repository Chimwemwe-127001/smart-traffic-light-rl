"""Coordinated multi-agent Q-learning: the research idea of this project.

Each intersection is its own Q-learning agent (see q_learning.py), but
neighbors exchange observations over a one-hop link:

    state  = own (EB bin, SB bin, green, green age) + neighbor (green, total count bin)
    reward = -(own queue) - 0.5 * (neighbor queue)

Seeing the neighbor's phase and load lets an agent anticipate the platoon that
is about to arrive. Sharing part of the neighbor's queue in the reward stops an
agent from "solving" its own queue by pushing congestion downstream. This is
the cooperative reward shaping used in networked traffic RL (see e.g.
Chu et al. 2019, MA2C, which uses a spatially discounted neighbor reward).

Independent learners are just the parent class, QLearning, run on the corridor.
"""
from traffic_rl.baselines import queue_reward
from traffic_rl.q_learning import QLearning
from traffic_rl.scenarios import SCENARIOS
from traffic_rl.state import count_bin

NEIGHBOR_WEIGHT = 0.5


class CoordinatedQLearning(QLearning):
    def __init__(self, scenario='corridor', neighbor_weight=NEIGHBOR_WEIGHT, **kwargs):
        super().__init__(**kwargs)
        self.neighbors = {tls: c['neighbor'] for tls, c in SCENARIOS[scenario]['intersections'].items()}
        self.neighbor_weight = neighbor_weight

    def state(self, tls, obs):
        n = obs[self.neighbors[tls]]
        return super().state(tls, obs) + (n['green'], count_bin(n['EB'] + n['SB'], self.cfg['count_bin_edges']))

    def reward(self, tls, queues):
        return queue_reward(tls, queues, self.neighbors[tls], self.neighbor_weight)
