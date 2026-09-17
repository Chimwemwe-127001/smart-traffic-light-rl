"""Train one agent on one scenario (one training run).

    python experiments/train.py --agent q_learning  --scenario single --run 0
    python experiments/train.py --agent dqn         --scenario single --run 0
    python experiments/train.py --agent q_learning  --scenario corridor --run 0    # independent learners
    python experiments/train.py --agent coordinated --scenario corridor --run 0    # neighbor sharing
    (experiments/train_all.py runs every agent, scenario and run in parallel)

Every learner is trained in several independent runs (--run 0 to 4), because
one run can be lucky or unlucky. The run number seeds the agent (initial
weights, exploration) and the order in which it meets the 900 training traffic
draws, so the runs differ in everything except the traffic they may see.

Every training episode is a new random traffic draw from seeds 0-899.
Exploration epsilon decays from 1.0 to 0.05 over the first 60% of episodes.
Every VAL_EVERY episodes the greedy policy is scored on 10 validation seeds;
the checkpoint with the best validation score is kept as the final model.

Seed ranges never overlap: training 0-899, SEMMA sample 900-902, final
evaluation 1000-1009, validation 2000-2009.

Outputs:
    results/logs/train_<agent>_<scenario>_run<k>.csv   one row per training episode
    results/logs/val_<agent>_<scenario>_run<k>.csv     greedy validation curve (episode 0 = untrained)
    results/models/<agent>_<scenario>_run<k>.json|.npz
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
from traffic_rl.scenarios import SCENARIOS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, 'results', 'logs')
MODELS = os.path.join(REPO, 'results', 'models')
VAL_SEEDS = list(range(2000, 2010))  # outside training, SEMMA and evaluation seeds
FIRST_RESERVED_SEED = 900           # SEMMA uses 900-902 and evaluation 1000-1009, so training stops at 899
VAL_EVERY = 10
EPS_END = 0.05


def make_agent(name, scenario, seed=0):
    if name == 'q_learning':
        return QLearning(seed=seed, scenario=scenario)
    if name == 'dqn':
        return DQN(seed=seed, scenario=scenario)
    if name == 'coordinated':
        return CoordinatedQLearning(scenario=scenario, seed=seed)
    raise ValueError(name)


def model_path(name, scenario, run):
    return os.path.join(MODELS, f'{name}_{scenario}_run{run}' + ('.npz' if name == 'dqn' else '.json'))


def validate(scenario, agent):
    """Score the greedy policy. Runs on a copy: acting adds unseen states to a
    Q-table, and validation must not change the agent being trained."""
    probe = copy.deepcopy(agent)
    runs = [run_episode(scenario, probe, seed=s) for s in VAL_SEEDS]
    return {k: float(np.mean([r[k] for r in runs])) for k in ('avg_queue', 'avg_wait_s', 'avg_travel_s', 'avg_delay_s')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', choices=['q_learning', 'dqn', 'coordinated'], required=True)
    ap.add_argument('--scenario', choices=list(SCENARIOS), required=True)
    ap.add_argument('--episodes', type=int, default=900)
    ap.add_argument('--run', type=int, default=0, help='training run: seeds the agent and the traffic order')
    args = ap.parse_args()
    if args.episodes > FIRST_RESERVED_SEED:
        sys.exit(f'Training seeds 0-{args.episodes - 1} would reach seeds reserved for SEMMA, '
                 f'evaluation or validation (from {FIRST_RESERVED_SEED}).')
    if args.agent == 'dqn' and len(SCENARIOS[args.scenario]['intersections']) != 1:
        sys.exit('The DQN controls one intersection; use a single-junction scenario.')
    if args.agent == 'coordinated' and len(SCENARIOS[args.scenario]['intersections']) < 2:
        sys.exit('Coordination needs a scenario with neighboring intersections.')

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    tag = f'{args.agent}_{args.scenario}_run{args.run}'
    agent = make_agent(args.agent, args.scenario, args.run)
    traffic = np.random.default_rng(args.run).permutation(args.episodes)     # order of the training draws
    decay = EPS_END ** (1 / (0.6 * args.episodes))

    # Checkpoint selection: the new junctions can back up past the network edge, where the
    # in-network queue would not see it, so they select on total delay instead.
    select_by = SCENARIOS[args.scenario].get('select_by', 'avg_queue')
    train_rows, val_rows = [], []
    best, best_agent = float('inf'), None
    start = time.time()
    for ep in range(args.episodes + 1):
        if ep % VAL_EVERY == 0 or ep == args.episodes:
            v = validate(args.scenario, agent)
            size = agent.n_states() if hasattr(agent, 'n_states') else ''
            val_rows.append([ep, v['avg_queue'], v['avg_wait_s'], v['avg_travel_s'], size, v['avg_delay_s']])
            flag = ''
            if ep > 0 and v[select_by] < best:
                best, best_agent, flag = v[select_by], copy.deepcopy(agent), '  <- best'
            print(f'[val] ep {ep:4d}  queue {v["avg_queue"]:6.2f}  wait {v["avg_wait_s"]:6.1f}s  '
                  f'delay {v["avg_delay_s"]:6.1f}s  travel {v["avg_travel_s"]:6.1f}s{flag}', flush=True)
        if ep == args.episodes:
            break

        agent.epsilon = max(EPS_END, decay ** ep)
        m = run_episode(args.scenario, agent, seed=int(traffic[ep]), explore=True, learn=True)
        size = agent.n_states() if hasattr(agent, 'n_states') else ''
        train_rows.append([ep, round(agent.epsilon, 4), m['total_reward'], m['avg_queue'],
                           m['avg_wait_s'], m['avg_travel_s'], m['throughput'], m['switches'],
                           m['td_loss'], size, m['avg_delay_s'], int(traffic[ep])])
        print(f'ep {ep:4d}  eps {agent.epsilon:.2f}  reward {m["total_reward"]:9.1f}  '
              f'queue {m["avg_queue"]:6.2f}  wait {m["avg_wait_s"]:6.1f}s  td {m["td_loss"]:.4f}  '
              f'states {size}  [{time.time() - start:.0f}s]', flush=True)

    with open(os.path.join(LOGS, f'train_{tag}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'epsilon', 'total_reward', 'avg_queue', 'avg_wait_s', 'avg_travel_s',
                    'throughput', 'switches', 'td_loss', 'q_table_states', 'avg_delay_s', 'traffic_seed'])
        w.writerows(train_rows)
    with open(os.path.join(LOGS, f'val_{tag}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'avg_queue', 'avg_wait_s', 'avg_travel_s', 'q_table_states', 'avg_delay_s'])
        w.writerows(val_rows)
    best_agent.save(model_path(args.agent, args.scenario, args.run))
    print(f'saved {model_path(args.agent, args.scenario, args.run)} (best validation {select_by} {best:.2f})')


if __name__ == '__main__':
    main()
