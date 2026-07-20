"""Run one episode of any controller on any scenario.

A controller is anything with:
    program                     'agent', 'fixed' or 'actuated'
    act(tls, obs, explore)      -> 0 keep | 1 switch
    learn(tls, obs, action, queues, next_obs) -> TD loss or None

The same loop is used for training (explore=True, learn=True) and for
evaluation (explore=False, learn=False), so any difference in results comes
from the policy, not from the harness.
"""
import numpy as np

from traffic_rl.env import TrafficEnv


def run_episode(scenario, controller, seed, explore=False, learn=False,
                episode_s=1200, gui=False):
    env = TrafficEnv(scenario, program=controller.program, episode_s=episode_s, gui=gui)
    ready, obs = env.reset(seed)
    rewards, losses = [], []

    if controller.program != 'agent':
        env.run_to_end()
    else:
        last = {}                                   # tls -> (obs, action) at its last decision
        while ready:
            actions = {}
            for tls in ready:
                if tls in last:
                    prev_obs, prev_action = last[tls]
                    queues = env.interval_queues(tls)
                    rewards.append(controller.reward(tls, queues))
                    if learn:
                        loss = controller.learn(tls, prev_obs, prev_action, queues, obs)
                        if loss is not None:
                            losses.append(loss)
                actions[tls] = controller.act(tls, obs, explore)
            decision_obs = obs
            ready, obs, done = env.step(actions)
            for tls, applied in env.applied.items():       # learn from what was really done
                last[tls] = (decision_obs, applied)

    metrics = env.close()
    metrics['total_reward'] = float(np.sum(rewards)) if rewards else float('nan')
    metrics['td_loss'] = float(np.mean(losses)) if losses else float('nan')
    return metrics
