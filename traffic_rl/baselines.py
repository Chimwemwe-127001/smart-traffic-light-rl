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
    """The hand-set fixed cycle stored in the network file (e.g. 42 s per arm)."""
    program = 'fixed'


MAX_CYCLE_S = 120           # a common practical upper limit on the cycle length


def webster_cycle(Y, L):
    """Webster's (1958) optimum cycle length in seconds: C0 = (1.5 L + 5) / (1 - Y).
    Y: sum over phases of the critical flow ratio y = demand / saturation flow.
    L: lost time per cycle (start-up plus clearance, summed over the phases)."""
    if Y >= 1:
        raise ValueError(f'Y = {Y:.2f}: demand exceeds what any fixed cycle can serve')
    return (1.5 * L + 5) / (1 - Y)


def webster_greens(y, lost, cycle, yellow=3, min_green=10, max_green=60):
    """Displayed green per phase. The effective green time in the cycle (cycle - L) is
    shared in proportion to y, and displayed green = effective green + the phase's lost
    time - yellow. Rounded to whole seconds and kept within [min_green, max_green]."""
    effective = cycle - sum(lost)
    return [int(min(max_green, max(min_green, round(effective * yi / sum(y) + li - yellow))))
            for yi, li in zip(y, lost)]


class Webster:
    """Fixed-time plan timed with Webster's method from measured saturation flows and the
    scenario's average demand (networks/*/webster*.add.xml, made by experiments/webster.py).
    The fixed-time plan a traffic engineer would install; the fair fixed-time baseline."""
    program = 'webster'


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
    scenarios only. Where two greens serve the same arms (Manda Hill: through
    traffic, then right turns), it counts only the lanes each green serves.
    """
    program = 'agent'

    def __init__(self, scenario):
        sc = SCENARIOS[scenario]
        tls = next(iter(sc['intersections']))
        names = arms(scenario, tls)
        # arm indices served by each green, in green order
        self.phase_arms = [[names.index(a) for a in group] for group in sc['phase_arms']] \
            if 'phase_arms' in sc else [[k] for k in range(len(names))]
        self.phase_lanes = sc.get('phase_lanes')      # detector lanes served by each green, if given

    def act(self, tls, obs, explore=False):
        if self.phase_lanes:
            lanes = obs[tls]['lanes']
            pressure = [sum(lanes[i] for i in group) for group in self.phase_lanes]
        else:
            counts = obs[tls]['counts']
            pressure = [sum(counts[a] for a in group) for group in self.phase_arms]
        return int(np.argmax(pressure))

    def reward(self, tls, queues):
        return queue_reward(tls, queues)

    def learn(self, *args):
        return None
