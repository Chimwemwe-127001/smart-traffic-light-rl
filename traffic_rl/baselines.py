"""Non-learning controllers every agent is compared against."""
import numpy as np


def queue_reward(tls, queues, neighbor=None, neighbor_weight=0.0):
    """Reward = minus the mean queue since the last decision (optionally plus a share of the neighbor's)."""
    r = -queues[tls]
    if neighbor is not None:
        r -= neighbor_weight * queues[neighbor]
    return r


class FixedTime:
    """The default fixed cycle from the network file: 42 s green, 3 s yellow per approach."""
    program = 'fixed'


class Actuated:
    """SUMO's gap-based actuated control (networks/*/actuated.add.xml)."""
    program = 'actuated'


class RandomPolicy:
    """Keep or switch with equal probability at every decision. The floor any agent must beat."""
    program = 'agent'

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, tls, obs, explore=False):
        return int(self.rng.integers(2))

    def reward(self, tls, queues):
        return queue_reward(tls, queues)

    def learn(self, *args):
        return None
