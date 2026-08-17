"""Non-learning controllers every agent is compared against."""
import numpy as np

from traffic_rl.scenarios import SCENARIOS, arms, n_actions


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


class LongestQueueFirst:
    """Give green to the phase whose arms have the most vehicles waiting.

    A strong adaptive heuristic in the spirit of max-pressure control
    (Varaiya, 2013), using only the detectors the agents also see. It keeps
    the current green while that phase is still the busiest; the environment
    applies the same min and max green rules as for the agents. For 'select'
    scenarios only. A tie goes to the earlier phase, so at Lusaka the
    protected right-turn phase is never chosen and right turns wait for gaps.
    """
    program = 'agent'

    def __init__(self, scenario):
        sc = SCENARIOS[scenario]
        tls = next(iter(sc['intersections']))
        names = arms(scenario, tls)
        # arm indices served by each green, in green order
        self.phase_arms = [[names.index(a) for a in group] for group in sc['phase_arms']] \
            if 'phase_arms' in sc else [[k] for k in range(len(names))]

    def act(self, tls, obs, explore=False):
        counts = obs[tls]['counts']
        pressure = [sum(counts[a] for a in group) for group in self.phase_arms]
        return int(np.argmax(pressure))

    def reward(self, tls, queues):
        return queue_reward(tls, queues)

    def learn(self, *args):
        return None
