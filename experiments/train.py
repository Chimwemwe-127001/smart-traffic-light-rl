"""Train one agent on one scenario.

    python experiments/train.py --agent q_learning  --scenario single
    python experiments/train.py --agent dqn         --scenario single
    python experiments/train.py --agent q_learning  --scenario corridor        # independent learners
    python experiments/train.py --agent coordinated --scenario corridor        # neighbor sharing
    (same two for --scenario corridor_heavy)

Every training episode is a new random traffic draw (seed = episode number).
Exploration epsilon decays from 1.0 to 0.05 over the first 60% of episodes.
Every VAL_EVERY episodes the greedy policy is scored on 3 validation seeds;
the checkpoint with the lowest validation queue is kept as the final model.
Evaluation seeds (evaluate.py) are never used here.

Outputs:
    results/logs/train_<agent>_<scenario>.csv   one row per training episode
    results/logs/val_<agent>_<scenario>.csv     greedy validation curve (episode 0 = untrained)
    results/models/<agent>_<scenario>.json|.npz
"""
import argparse
import copy
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic_rl.dqn import DQN
from traffic_rl.multi_agent import CoordinatedQLearning
from traffic_rl.q_learning import QLearning
from traffic_rl.runner import run_episode

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, 'results', 'logs')
MODELS = os.path.join(REPO, 'results', 'models')
VAL_SEEDS = [500, 501, 502]
VAL_EVERY = 10
EPS_END = 0.05


def make_agent(name, scenario, seed=0):
    if name == 'q_learning':
        return QLearning(seed=seed)
    if name == 'dqn':
        return DQN(seed=seed)
    if name == 'coordinated':
        return CoordinatedQLearning(scenario=scenario, seed=seed)
    raise ValueError(name)


def model_path(name, scenario):
    return os.path.join(MODELS, f'{name}_{scenario}' + ('.npz' if name == 'dqn' else '.json'))


def validate(scenario, agent):
    runs = [run_episode(scenario, agent, seed=s) for s in VAL_SEEDS]
    return {k: float(np.mean([r[k] for r in runs])) for k in ('avg_queue', 'avg_wait_s', 'avg_travel_s')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', choices=['q_learning', 'dqn', 'coordinated'], required=True)
    ap.add_argument('--scenario', choices=['single', 'corridor', 'corridor_heavy'], required=True)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.agent == 'dqn' and args.scenario != 'single':
        sys.exit('The DQN input layer is sized for the single intersection.')

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    tag = f'{args.agent}_{args.scenario}'
    agent = make_agent(args.agent, args.scenario, args.seed)
    decay = EPS_END ** (1 / (0.6 * args.episodes))

    train_rows, val_rows = [], []
    best, best_agent = float('inf'), None
    start = time.time()
    for ep in range(args.episodes + 1):
        if ep % VAL_EVERY == 0 or ep == args.episodes:
            v = validate(args.scenario, agent)
            size = agent.n_states() if hasattr(agent, 'n_states') else ''
            val_rows.append([ep, v['avg_queue'], v['avg_wait_s'], v['avg_travel_s'], size])
            flag = ''
            if ep > 0 and v['avg_queue'] < best:
                best, best_agent, flag = v['avg_queue'], copy.deepcopy(agent), '  <- best'
            print(f'[val] ep {ep:4d}  queue {v["avg_queue"]:6.2f}  wait {v["avg_wait_s"]:6.1f}s  '
                  f'travel {v["avg_travel_s"]:6.1f}s{flag}', flush=True)
        if ep == args.episodes:
            break

        agent.epsilon = max(EPS_END, decay ** ep)
        m = run_episode(args.scenario, agent, seed=ep, explore=True, learn=True)
        size = agent.n_states() if hasattr(agent, 'n_states') else ''
        train_rows.append([ep, round(agent.epsilon, 4), m['total_reward'], m['avg_queue'],
                           m['avg_wait_s'], m['avg_travel_s'], m['throughput'], m['switches'],
                           m['td_loss'], size])
        print(f'ep {ep:4d}  eps {agent.epsilon:.2f}  reward {m["total_reward"]:9.1f}  '
              f'queue {m["avg_queue"]:6.2f}  wait {m["avg_wait_s"]:6.1f}s  td {m["td_loss"]:.4f}  '
              f'states {size}  [{time.time() - start:.0f}s]', flush=True)

    with open(os.path.join(LOGS, f'train_{tag}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'epsilon', 'total_reward', 'avg_queue', 'avg_wait_s', 'avg_travel_s',
                    'throughput', 'switches', 'td_loss', 'q_table_states'])
        w.writerows(train_rows)
    with open(os.path.join(LOGS, f'val_{tag}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'avg_queue', 'avg_wait_s', 'avg_travel_s', 'q_table_states'])
        w.writerows(val_rows)
    best_agent.save(model_path(args.agent, args.scenario))
    print(f'saved {model_path(args.agent, args.scenario)} (best validation queue {best:.2f})')


if __name__ == '__main__':
    main()
