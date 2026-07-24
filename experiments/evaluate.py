"""Evaluate every controller on held-out traffic and score the rubric.

Protocol
    - 10 evaluation seeds (1000-1009), never used in training or validation
    - every controller runs on exactly the same seeds (paired comparison)
    - learners act greedily (no exploration, no learning)
    - "untrained" = the same agent class with its initial parameters,
      so before vs after isolates what training added

Outputs:
    results/evaluation.csv   one row per (scenario, controller, seed)
    results/summary.json     everything below as data
    results/RESULTS.md       tables, probes and rubric

Usage:
    python experiments/evaluate.py
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic_rl.baselines import Actuated, FixedTime, RandomPolicy
from traffic_rl.dqn import DQN
from traffic_rl.metrics import bootstrap_ci, paired_difference, percent_change
from traffic_rl.multi_agent import CoordinatedQLearning
from traffic_rl.q_learning import QLearning
from traffic_rl.runner import run_episode

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'results')
MODELS = os.path.join(RESULTS, 'models')
LOGS = os.path.join(RESULTS, 'logs')
EVAL_SEEDS = list(range(1000, 1010))
METRICS = ['avg_queue', 'avg_wait_s', 'avg_travel_s', 'throughput', 'backlog', 'switches']
LABELS = {'avg_queue': 'Avg queue (veh)', 'avg_wait_s': 'Avg wait (s)', 'avg_travel_s': 'Avg travel time (s)',
          'throughput': 'Throughput (veh)', 'backlog': 'Backlog (veh)', 'switches': 'Switches'}


def model(name):
    return os.path.join(MODELS, name)


def controllers(scenario):
    base = {'Fixed-time': FixedTime(), 'Actuated': Actuated(), 'Random': RandomPolicy(seed=7)}
    if scenario == 'single':
        return {**base,
                'Q-learning (untrained)': QLearning(),
                'Q-learning (trained)': QLearning().load(model('q_learning_single.json')),
                'DQN (untrained)': DQN(),
                'DQN (trained)': DQN().load(model('dqn_single.npz'))}
    return {**base,
            'Independent QL (untrained)': QLearning(),
            'Independent QL (trained)': QLearning().load(model(f'q_learning_{scenario}.json')),
            'Coordinated QL (untrained)': CoordinatedQLearning(scenario),
            'Coordinated QL (trained)': CoordinatedQLearning(scenario).load(model(f'coordinated_{scenario}.json'))}


# ------------------------------------------------------------------ probes

def probe_obs(eb, sb, green):
    per_lane = lambda n: [n / 3] * 3
    return {'Node2': {'lanes': np.array(per_lane(eb) + per_lane(sb)), 'EB': float(eb), 'SB': float(sb),
                      'green': green, 'green_time': 20.0, 'queue': float(eb + sb)}}


PROBES = [
    ('SB busy, EB empty, EB has green', probe_obs(0, 15, 0), 1),
    ('EB busy, SB empty, SB has green', probe_obs(15, 0, 1), 1),
    ('EB busy, SB empty, EB has green', probe_obs(15, 0, 0), 0),
    ('SB busy, EB empty, SB has green', probe_obs(0, 15, 1), 0),
]


def run_probes(agent):
    """A probe only counts if the agent clearly prefers the right action.
    A tie (e.g. a state the Q-table never visited, Q = 0, 0) is not an answer."""
    out = []
    for desc, obs, expected in PROBES:
        if isinstance(agent, DQN):
            q = agent.net.predict(agent.features('Node2', obs))[0]
            seen = True
        else:
            s = agent.state('Node2', obs)
            seen = s in agent.tables.get('Node2', {})
            q = agent.tables['Node2'][s] if seen else np.zeros(2)
        out.append({'probe': desc, 'expected': 'switch' if expected else 'keep', 'seen': seen,
                    'q_keep': float(q[0]), 'q_switch': float(q[1]),
                    'ok': bool(q[0] != q[1] and int(np.argmax(q)) == expected)})
    return out


# -------------------------------------------------------------- training logs

def td_loss_trend(tag, k=20):
    """Mean TD loss over the first and last k logged episodes (after replay warm-up)."""
    with open(os.path.join(LOGS, f'train_{tag}.csv')) as f:
        loss = [float(r['td_loss']) for r in csv.DictReader(f) if r['td_loss'] not in ('', 'nan')]
    return float(np.mean(loss[:k])), float(np.mean(loss[-k:]))


# -------------------------------------------------------------------- main

def main():
    per_seed = []
    runs = {}
    for scenario in ['single', 'corridor', 'corridor_heavy']:
        runs[scenario] = {}
        for name, ctrl in controllers(scenario).items():
            ms = [run_episode(scenario, ctrl, seed=s) for s in EVAL_SEEDS]
            runs[scenario][name] = {k: [m[k] for m in ms] for k in METRICS}
            for s, m in zip(EVAL_SEEDS, ms):
                per_seed.append({'scenario': scenario, 'controller': name, 'seed': s,
                                 **{k: m[k] for k in METRICS}})
            print(f'{scenario:15s} {name:28s} queue {np.mean(runs[scenario][name]["avg_queue"]):6.2f}  '
                  f'wait {np.mean(runs[scenario][name]["avg_wait_s"]):6.1f}s', flush=True)

    with open(os.path.join(RESULTS, 'evaluation.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'controller', 'seed'] + METRICS)
        w.writeheader()
        w.writerows(per_seed)

    summary = {sc: {name: {k: bootstrap_ci(v) for k, v in d.items()} for name, d in r.items()}
               for sc, r in runs.items()}

    # Before vs after training (same agent, same seeds)
    learners = [('single', 'Q-learning'), ('single', 'DQN'),
                ('corridor', 'Independent QL'), ('corridor', 'Coordinated QL'),
                ('corridor_heavy', 'Independent QL'), ('corridor_heavy', 'Coordinated QL')]
    before_after = []
    for sc, name in learners:
        b, a = runs[sc][f'{name} (untrained)'], runs[sc][f'{name} (trained)']
        row = {'scenario': sc, 'learner': name}
        for k in ('avg_queue', 'avg_wait_s', 'avg_travel_s'):
            row[k] = {'before': float(np.mean(b[k])), 'after': float(np.mean(a[k])),
                      'change_pct': percent_change(np.mean(a[k]), np.mean(b[k])),
                      'diff_ci': paired_difference(a[k], b[k])}
        before_after.append(row)

    probes = {'Q-learning': run_probes(QLearning().load(model('q_learning_single.json'))),
              'DQN': run_probes(DQN().load(model('dqn_single.npz')))}
    td = {tag: td_loss_trend(tag) for tag in
          ['q_learning_single', 'dqn_single', 'q_learning_corridor', 'coordinated_corridor',
           'q_learning_corridor_heavy', 'coordinated_corridor_heavy']}

    # Every trained learner against the strongest baseline, seed by seed
    vs_actuated = []
    for sc, name in learners:
        row = {'scenario': sc, 'learner': name}
        for k in ('avg_wait_s', 'avg_travel_s', 'avg_queue'):
            row[k] = paired_difference(runs[sc][f'{name} (trained)'][k], runs[sc]['Actuated'][k])
        vs_actuated.append(row)

    rubric = score_rubric(runs, probes, td)
    out = {'summary': summary, 'before_after': before_after, 'vs_actuated': vs_actuated, 'probes': probes,
           'td_loss': td, 'rubric': rubric, 'eval_seeds': EVAL_SEEDS}
    with open(os.path.join(RESULTS, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    write_markdown(out)
    print('wrote results/evaluation.csv, results/summary.json, results/RESULTS.md')


# ------------------------------------------------------------------ rubric

def score_rubric(runs, probes, td):
    rows = []

    def add(criterion, measure, threshold, result, passed):
        rows.append({'criterion': criterion, 'measure': measure, 'threshold': threshold,
                     'result': result, 'pass': bool(passed)})

    trained = [('single', 'Q-learning'), ('single', 'DQN'), ('corridor', 'Independent QL'),
               ('corridor', 'Coordinated QL'), ('corridor_heavy', 'Independent QL'),
               ('corridor_heavy', 'Coordinated QL')]
    for sc, name in trained:
        a = runs[sc][f'{name} (trained)']['avg_wait_s']
        b = runs[sc][f'{name} (untrained)']['avg_wait_s']
        m, lo, hi = paired_difference(a, b)
        add(f'{name} learned ({sc})', 'avg wait, trained minus untrained, 95% CI',
            'CI below 0', f'{m:+.1f} s [{lo:+.1f}, {hi:+.1f}]', hi < 0)
    for sc, name in trained:
        a = runs[sc][f'{name} (trained)']['avg_wait_s']
        for base in ('Fixed-time', 'Random'):
            m, lo, hi = paired_difference(a, runs[sc][base]['avg_wait_s'])
            add(f'{name} beats {base} ({sc})', f'avg wait, trained minus {base}, 95% CI',
                'CI below 0', f'{m:+.1f} s [{lo:+.1f}, {hi:+.1f}]', hi < 0)
    for sc, name in trained:
        a = runs[sc][f'{name} (trained)']['avg_wait_s']
        act = runs[sc]['Actuated']['avg_wait_s']
        pct = percent_change(np.mean(a), np.mean(act))
        m, lo, hi = paired_difference(a, act)
        add(f'{name} competitive with Actuated ({sc})', 'avg wait vs Actuated, paired 95% CI',
            'within +10%', f'{pct:+.1f}% ({m:+.2f} s [{lo:+.2f}, {hi:+.2f}])', pct <= 10)
    for name, res in probes.items():
        n = sum(p['ok'] for p in res)
        add(f'{name} learned traffic logic', 'hand-made probe states answered correctly',
            '4 of 4', f'{n} of 4', n == 4)
    for tag, (first, last) in td.items():
        add(f'Stable learning ({tag})', 'TD loss, last 20 vs first 20 episodes',
            'last <= 1.5 x first', f'{first:.4f} -> {last:.4f}', last <= 1.5 * first)
    for sc in ('corridor', 'corridor_heavy'):
        m, lo, hi = paired_difference(runs[sc]['Coordinated QL (trained)']['avg_wait_s'],
                                      runs[sc]['Independent QL (trained)']['avg_wait_s'])
        add(f'Coordination helps ({sc})', 'avg wait, coordinated minus independent, 95% CI',
            'CI below 0', f'{m:+.1f} s [{lo:+.1f}, {hi:+.1f}]', hi < 0)
    for sc, r in runs.items():
        for name, d in r.items():
            if '(trained)' in name:
                ft = np.mean(r['Fixed-time']['throughput'])
                pct = 100 * np.mean(d['throughput']) / ft
                add(f'{name.replace(" (trained)", "")} serves all demand ({sc})',
                    'vehicles completed vs Fixed-time', 'at least 98%', f'{pct:.1f}%', pct >= 98)
    return rows


# ---------------------------------------------------------------- markdown

def fmt(ci, digits=2):
    m, lo, hi = ci
    return f'{m:.{digits}f} ± {(hi - lo) / 2:.{digits}f}'


def write_markdown(out):
    L = ['# Results', '',
         f'Held-out evaluation on {len(out["eval_seeds"])} traffic seeds '
         f'({out["eval_seeds"][0]}-{out["eval_seeds"][-1]}), never seen in training. '
         'Values are mean ± half-width of the 95% bootstrap CI. Generated by `experiments/evaluate.py`.', '']
    titles = {'single': 'Single intersection (shifting demand)',
              'corridor': 'Corridor, 2 intersections, normal demand',
              'corridor_heavy': 'Corridor, 2 intersections, heavy demand'}
    cols = ['avg_queue', 'avg_wait_s', 'avg_travel_s', 'throughput', 'switches']
    for sc, table in out['summary'].items():
        L += [f'## {titles[sc]}', '', '| Controller | ' + ' | '.join(LABELS[c] for c in cols) + ' |',
              '|---|' + '---|' * len(cols)]
        for name, d in table.items():
            cells = []
            for c in cols:
                if c == 'switches' and name in ('Fixed-time', 'Actuated'):
                    cells.append('n/a')          # SUMO runs these programs; switches are not counted
                else:
                    cells.append(fmt(d[c], 0 if c in ('throughput', 'switches') else 2))
            L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        L.append('')
    L += ['## Before vs after training', '',
          '| Scenario | Learner | Wait before (s) | Wait after (s) | Change | Queue before | Queue after | Change |',
          '|---|---|---|---|---|---|---|---|']
    for r in out['before_after']:
        w, q = r['avg_wait_s'], r['avg_queue']
        L.append(f'| {r["scenario"]} | {r["learner"]} | {w["before"]:.1f} | {w["after"]:.1f} | '
                 f'{w["change_pct"]:+.0f}% | {q["before"]:.2f} | {q["after"]:.2f} | {q["change_pct"]:+.0f}% |')
    L += ['', '## Trained learners vs Actuated control', '',
          'Paired difference per seed (learner minus Actuated), mean and 95% bootstrap CI. '
          'Negative is better for the learner. "tie" means the CI includes zero.', '',
          '| Scenario | Learner | Wait (s) | Travel time (s) | Queue (veh) |', '|---|---|---|---|---|']

    def verdict(ci):
        m, lo, hi = ci
        tag = 'better' if hi < 0 else ('worse' if lo > 0 else 'tie')
        return f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}] {tag}'
    for r in out['vs_actuated']:
        L.append(f'| {r["scenario"]} | {r["learner"]} | {verdict(r["avg_wait_s"])} | '
                 f'{verdict(r["avg_travel_s"])} | {verdict(r["avg_queue"])} |')
    L += ['', '## Q-value probes (single intersection)', '',
          '| Agent | Probe | Expected | Q(keep) | Q(switch) | Correct |', '|---|---|---|---|---|---|']
    for agent, res in out['probes'].items():
        for p in res:
            verdict = 'yes' if p['ok'] else ('no' if p['seen'] else 'no (state never visited)')
            L.append(f'| {agent} | {p["probe"]} | {p["expected"]} | {p["q_keep"]:.3f} | '
                     f'{p["q_switch"]:.3f} | {verdict} |')
    L += ['', '## Rubric', '', '| Criterion | Measure | Threshold | Result | Pass |', '|---|---|---|---|---|']
    for r in out['rubric']:
        L.append(f'| {r["criterion"]} | {r["measure"]} | {r["threshold"]} | {r["result"]} | '
                 f'{"PASS" if r["pass"] else "FAIL"} |')
    n_pass = sum(r['pass'] for r in out['rubric'])
    L += ['', f'**{n_pass} of {len(out["rubric"])} rubric checks pass.**', '']
    with open(os.path.join(RESULTS, 'RESULTS.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    main()
