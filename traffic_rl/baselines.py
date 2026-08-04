"""Non-learning controllers every agent is compared against."""
import numpy as np

from traffic_rl.scenarios import n_actions


def queue_reward(tls, queues, neighbor=None, neighbor_weight=0.0):
    """Reward = minus the mean queue since the last decision (optionally plus a share of the neighbor's)."""
    r = -queues[tls]
    if neighbor is not None:
        r -= neighbor_weight * queues[neighbor]
    return r


class FixedTime:
    """The fixed cycle stored in the network file."""
    program = 'fixed'


class Actuated:
    """SUMO's gap-based actuated control (networks/*/actuated.add.xml)."""
    program = 'actuated'


class RandomPolicy:
    """A random action at every decision. The floor any agent must beat."""
    program = 'agent'

    def __init__(self, seed=0, scenario='single'):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions(scenario)

    def act(self, tls, obs, explore=False):
        return int(self.rng.integers(self.n_actions))

    def reward(self, tls, queues):
        return queue_reward(tls, queues)

    def learn(self, *args):
        return None
