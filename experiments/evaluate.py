"""Evaluate every controller on held-out traffic and score the rubric.

Protocol
    - 10 evaluation seeds (1000-1009), never used in training or validation
    - every controller runs on exactly the same seeds (paired comparison)
    - learners act greedily (no exploration, no learning)
    - "untrained" = the same agent class with its initial parameters,
      so before vs after isolates what training added

Scenarios
    v1:   single, corridor, corridor_heavy
    v1.1: four_way (one lane each way, split phasing) and lusaka
          (Great East Road / Lufubu Road, plus a demand sweep x0.75 / x1.25)

Outputs:
    results/evaluation.csv   one row per (scenario, controller, seed)
    results/summary.json     everything below as data
    results/RESULTS.md       tables, fairness, probes, sweep and rubric

Usage:
    python experiments/evaluate.py
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic_rl.baselines import Actuated, FixedTime, LongestQueueFirst, RandomPolicy
from traffic_rl.dqn import DQN
from traffic_rl.metrics import bootstrap_ci, paired_difference, percent_change
from traffic_rl.multi_agent import CoordinatedQLearning
from traffic_rl.q_learning import QLearning
from traffic_rl.runner import run_episode
from traffic_rl.scenarios import SCENARIOS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'results')
MODELS = os.path.join(RESULTS, 'models')
LOGS = os.path.join(RESULTS, 'logs')
EVAL_SEEDS = list(range(1000, 1010))
V1 = ['single', 'corridor', 'corridor_heavy']
V11 = ['four_way', 'lusaka']
METRICS = ['avg_queue', 'avg_wait_s', 'avg_travel_s', 'throughput', 'backlog', 'switches', 'avg_delay_s', 'max_delay_s']
LABELS = {'avg_queue': 'Avg queue (veh)', 'avg_wait_s': 'Avg wait (s)', 'avg_travel_s': 'Avg travel time (s)',
          'throughput': 'Throughput (veh)', 'backlog': 'Backlog (veh)', 'switches': 'Switches',
          'avg_delay_s': 'Avg delay (s)', 'max_delay_s': 'Worst delay (s)'}
TITLES = {'single': 'Single intersection (shifting demand)',
          'corridor': 'Corridor, 2 intersections, normal demand',
          'corridor_heavy': 'Corridor, 2 intersections, heavy demand',
          'four_way': 'Four-way junction, one lane each way, split phasing',
          'lusaka': 'Lusaka: Great East Road / Lufubu Road, estimated morning peak'}
LEARNERS = [('single', 'Q-learning'), ('single', 'DQN'),
            ('corridor', 'Independent QL'), ('corridor', 'Coordinated QL'),
            ('corridor_heavy', 'Independent QL'), ('corridor_heavy', 'Coordinated QL'),
            ('four_way', 'Q-learning'), ('four_way', 'DQN'),
            ('lusaka', 'Q-learning'), ('lusaka', 'DQN')]


def model(name):
    return os.path.join(MODELS, name)


def key(scenario):
    """Headline metric. The new junctions can back up past the network edge, so they use
    total delay (stopped + queued before entering); v1 keeps average wait, as published."""
    return 'avg_delay_s' if scenario in V11 else 'avg_wait_s'


def controllers(scenario):
    base = {'Fixed-time': FixedTime(), 'Actuated': Actuated(), 'Random': RandomPolicy(seed=7, scenario=scenario)}
    if scenario in V11:
        return {**base, 'Longest queue first': LongestQueueFirst(scenario),
                'Q-learning (untrained)': QLearning(scenario=scenario),
                'Q-learning (trained)': QLearning(scenario=scenario).load(model(f'q_learning_{scenario}.json')),
                'DQN (untrained)': DQN(scenario=scenario),
                'DQN (trained)': DQN(scenario=scenario).load(model(f'dqn_{scenario}.npz'))}
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

def make_obs(tls, lanes_per_arm, counts, green, n_greens):
    """A hand-made observation: each arm's count spread evenly over its lanes."""
    lanes = [c / n for c, n in zip(counts, lanes_per_arm) for _ in range(n)]
    return {tls: {'lanes': np.array(lanes, dtype=float), 'counts': [float(c) for c in counts],
                  'green': green, 'n_greens': n_greens, 'green_time': 20.0, 'queue': float(sum(counts))}}


# (description, observation, right action). v1: action 0 keep, 1 switch; EB green = 0, SB green = 1.
PROBES = {
    'single': ('Node2', [
        ('SB busy, EB empty, EB has green', make_obs('Node2', [3, 3], [0, 15], 0, 2), 1),
        ('EB busy, SB empty, SB has green', make_obs('Node2', [3, 3], [15, 0], 1, 2), 1),
        ('EB busy, SB empty, EB has green', make_obs('Node2', [3, 3], [15, 0], 0, 2), 0),
        ('SB busy, EB empty, SB has green', make_obs('Node2', [3, 3], [0, 15], 1, 2), 0),
    ]),
    # four_way: arms N, E, S, W; action k = give green to arm k
    'four_way': ('C', [
        ('N busy, others empty, E has green', make_obs('C', [1] * 4, [12, 0, 0, 0], 1, 4), 0),
        ('W busy, others empty, N has green', make_obs('C', [1] * 4, [0, 0, 0, 12], 0, 4), 3),
        ('E busy, others empty, E has green', make_obs('C', [1] * 4, [0, 12, 0, 0], 1, 4), 1),
        ('S busy, W light, S has green', make_obs('C', [1] * 4, [0, 0, 12, 2], 2, 4), 2),
    ]),
    # lusaka: arms W, E, N, S; greens 0 main road, 1 main-road right turns, 2 side roads
    'lusaka': ('C', [
        ('Main road busy, side roads empty, side roads have green',
         make_obs('C', [2, 2, 1, 1], [30, 40, 0, 0], 2, 3), 0),
        ('Side roads busy, main road light, main road has green',
         make_obs('C', [2, 2, 1, 1], [2, 3, 12, 6], 0, 3), 2),
        ('Main road busy, side roads empty, main road has green',
         make_obs('C', [2, 2, 1, 1], [30, 40, 0, 0], 0, 3), 0),
    ]),
}


def action_name(scenario, k):
    if scenario == 'single':
        return ['keep', 'switch'][k]
    if scenario == 'lusaka':
        return ['main road', 'main-road right turns', 'side roads'][k]
    return f'serve {"NESW"[k]}'


def run_probes(scenario, agent):
    """A probe only counts if the agent clearly prefers the right action.
    A tie (e.g. a state the Q-table never visited, all Q = 0) is not an answer."""
    tls, probes = PROBES[scenario]
    out = []
    for desc, obs, expected in probes:
        if isinstance(agent, DQN):
            q = agent.net.predict(agent.features(tls, obs))[0]
            seen = True
        else:
            s = agent.state(tls, obs)
            seen = s in agent.tables.get(tls, {})
            q = agent.tables[tls][s] if seen else np.zeros(agent.n_actions)
        best = int(np.argmax(q))
        out.append({'probe': desc, 'expected': action_name(scenario, expected), 'seen': seen,
                    'chosen': action_name(scenario, best), 'q': [float(v) for v in q],
                    'ok': bool(best == expected and np.sum(q == q[best]) == 1)})
    return out


# -------------------------------------------------------------- training logs

def td_loss_trend(tag, k=20):
    """Mean TD loss over the first and last k logged episodes (after replay warm-up)."""
    with open(os.path.join(LOGS, f'train_{tag}.csv')) as f:
        loss = [float(r['td_loss']) for r in csv.DictReader(f) if r['td_loss'] not in ('', 'nan')]
    return float(np.mean(loss[:k])), float(np.mean(loss[-k:]))


# -------------------------------------------------------------------- main

def evaluate(scenario, ctrls, per_seed, label=None, routes=None):
    label = label or scenario
    runs = {}
    for name, ctrl in ctrls.items():
        ms = [run_episode(scenario, ctrl, seed=s, routes=routes) for s in EVAL_SEEDS]
        runs[name] = {k: [m[k] for m in ms] for k in METRICS}
        runs[name]['arm_delay'] = [m['arm_delay'] for m in ms]
        for s, m in zip(EVAL_SEEDS, ms):
            per_seed.append({'scenario': label, 'controller': name, 'seed': s, **{k: m[k] for k in METRICS}})
        print(f'{label:15s} {name:28s} queue {np.mean(runs[name]["avg_queue"]):6.2f}  '
              f'wait {np.mean(runs[name]["avg_wait_s"]):6.1f}s  delay {np.mean(runs[name]["avg_delay_s"]):6.1f}s  '
              f'worst {np.mean(runs[name]["max_delay_s"]):6.1f}s', flush=True)
    return runs


def main():
    per_seed = []
    runs = {sc: evaluate(sc, controllers(sc), per_seed) for sc in V1 + V11}

    # Lusaka demand sweep: the estimated peak is uncertain, so repeat at x0.75 and x1.25
    sweep = {}
    for scale, routes in SCENARIOS['lusaka']['demand_sweep'].items():
        ctrls = {k: v for k, v in controllers('lusaka').items()
                 if k in ('Fixed-time', 'Actuated', 'Longest queue first', 'Q-learning (trained)', 'DQN (trained)')}
        sweep[scale] = evaluate('lusaka', ctrls, per_seed, label=f'lusaka_x{scale:.2f}', routes=routes)
    sweep[1.0] = {k: v for k, v in runs['lusaka'].items() if k in sweep[0.75]}

    with open(os.path.join(RESULTS, 'evaluation.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'controller', 'seed'] + METRICS)
        w.writeheader()
        w.writerows(per_seed)

    summary = {sc: {name: {k: bootstrap_ci(v) for k, v in d.items() if k in METRICS} for name, d in r.items()}
               for sc, r in runs.items()}

    # Before vs after training (same agent, same seeds)
    before_after = []
    for sc, name in LEARNERS:
        b, a = runs[sc][f'{name} (untrained)'], runs[sc][f'{name} (trained)']
        row = {'scenario': sc, 'learner': name}
        for k in ('avg_queue', 'avg_wait_s', 'avg_travel_s', 'avg_delay_s'):
            row[k] = {'before': float(np.mean(b[k])), 'after': float(np.mean(a[k])),
                      'change_pct': percent_change(np.mean(a[k]), np.mean(b[k])),
                      'diff_ci': paired_difference(a[k], b[k])}
        before_after.append(row)

    # Every trained learner against the strongest baselines, seed by seed
    def versus(baseline, pairs):
        out = []
        for sc, name in pairs:
            row = {'scenario': sc, 'learner': name}
            for k in ('avg_wait_s', 'avg_delay_s', 'max_delay_s', 'avg_travel_s', 'avg_queue'):
                row[k] = paired_difference(runs[sc][f'{name} (trained)'][k], runs[sc][baseline][k])
            out.append(row)
        return out
    vs_actuated = versus('Actuated', LEARNERS)
    vs_lqf = versus('Longest queue first', [(sc, n) for sc, n in LEARNERS if sc in V11])

    # Fairness on the new junctions: worst single wait and average wait per arm
    fairness = {}
    for sc in V11:
        fairness[sc] = {}
        for name, d in runs[sc].items():
            arm_names = SCENARIOS[sc]['intersections']['C']['approaches']
            fairness[sc][name] = {'max_delay_s': bootstrap_ci(d['max_delay_s']),
                                  'arm_delay': {a: float(np.mean([ad.get(f'C:{a}', 0.0) for ad in d['arm_delay']]))
                                                for a in arm_names}}

    sweep_out = {f'{s:.2f}': {name: {k: float(np.mean(d[k])) for k in
                                     ('avg_delay_s', 'max_delay_s', 'avg_wait_s', 'throughput', 'backlog')}
                              for name, d in sweep[s].items()} for s in sorted(sweep)}

    probes = {f'{name} ({sc})': run_probes(sc, controllers(sc)[f'{name} (trained)'])
              for sc, name in LEARNERS if sc in PROBES}
    td = {f'{tag}_{sc}': td_loss_trend(f'{tag}_{sc}') for sc, tag in
          [('single', 'q_learning'), ('single', 'dqn'), ('corridor', 'q_learning'), ('corridor', 'coordinated'),
           ('corridor_heavy', 'q_learning'), ('corridor_heavy', 'coordinated'),
           ('four_way', 'q_learning'), ('four_way', 'dqn'), ('lusaka', 'q_learning'), ('lusaka', 'dqn')]}

    rubric = score_rubric(runs, probes, td)
    out = {'summary': summary, 'before_after': before_after, 'vs_actuated': vs_actuated, 'vs_lqf': vs_lqf,
           'fairness': fairness, 'sweep': sweep_out, 'probes': probes, 'td_loss': td, 'rubric': rubric,
           'eval_seeds': EVAL_SEEDS}
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

    label = lambda sc: 'avg delay' if sc in V11 else 'avg wait'
    for sc, name in LEARNERS:
        a = runs[sc][f'{name} (trained)'][key(sc)]
        b = runs[sc][f'{name} (untrained)'][key(sc)]
        m, lo, hi = paired_difference(a, b)
        add(f'{name} learned ({sc})', f'{label(sc)}, trained minus untrained, 95% CI',
            'CI below 0', f'{m:+.1f} s [{lo:+.1f}, {hi:+.1f}]', hi < 0)
    for sc, name in LEARNERS:
        a = runs[sc][f'{name} (trained)'][key(sc)]
        for base in ('Fixed-time', 'Random'):
            m, lo, hi = paired_difference(a, runs[sc][base][key(sc)])
            add(f'{name} beats {base} ({sc})', f'{label(sc)}, trained minus {base}, 95% CI',
                'CI below 0', f'{m:+.1f} s [{lo:+.1f}, {hi:+.1f}]', hi < 0)
    for sc, name in LEARNERS:
        a = runs[sc][f'{name} (trained)'][key(sc)]
        act = runs[sc]['Actuated'][key(sc)]
        pct = percent_change(np.mean(a), np.mean(act))
        m, lo, hi = paired_difference(a, act)
        add(f'{name} competitive with Actuated ({sc})', f'{label(sc)} vs Actuated, paired 95% CI',
            'within +10%', f'{pct:+.1f}% ({m:+.2f} s [{lo:+.2f}, {hi:+.2f}])', pct <= 10)
    for sc, name in LEARNERS:
        if sc not in V11:
            continue
        m, lo, hi = paired_difference(runs[sc][f'{name} (trained)'][key(sc)],
                                      runs[sc]['Longest queue first'][key(sc)])
        add(f'{name} beats longest queue first ({sc})', f'{label(sc)}, trained minus LQF, 95% CI',
            'CI below 0', f'{m:+.1f} s [{lo:+.1f}, {hi:+.1f}]', hi < 0)
        worst = np.mean(runs[sc][f'{name} (trained)']['max_delay_s'])
        act_worst = np.mean(runs[sc]['Actuated']['max_delay_s'])
        add(f'{name} starves no one ({sc})', 'worst single-vehicle delay vs Actuated',
            'at most 1.25 x Actuated', f'{worst:.0f} s vs {act_worst:.0f} s', worst <= 1.25 * act_worst)
    for name, res in probes.items():
        n = sum(p['ok'] for p in res)
        add(f'{name} learned traffic logic', 'hand-made probe states answered correctly',
            f'{len(res)} of {len(res)}', f'{n} of {len(res)}', n == len(res))
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


def verdict(ci):
    m, lo, hi = ci
    tag = 'better' if hi < 0 else ('worse' if lo > 0 else 'tie')
    return f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}] {tag}'


def write_markdown(out):
    L = ['# Results', '',
         f'Held-out evaluation on {len(out["eval_seeds"])} traffic seeds '
         f'({out["eval_seeds"][0]}-{out["eval_seeds"][-1]}), never seen in training. '
         'Values are mean ± half-width of the 95% bootstrap CI. Generated by `experiments/evaluate.py`.', '']
    L += ['**Metrics.** Wait = time stopped, for vehicles that entered the network. Delay = time stopped '
          'plus time queued before entering, for every vehicle, including those still outside at the end. '
          'The new junctions (four-way, Lusaka) are judged on delay, because a controller that starves an arm '
          'can push its queue outside the network, where wait would not see it.', '']
    for sc, table in out['summary'].items():
        cols = (['avg_queue', 'avg_wait_s', 'avg_travel_s', 'throughput', 'switches'] if sc in V1 else
                ['avg_delay_s', 'max_delay_s', 'avg_wait_s', 'avg_queue', 'throughput', 'backlog', 'switches'])
        L += [f'## {TITLES[sc]}', '', '| Controller | ' + ' | '.join(LABELS[c] for c in cols) + ' |',
              '|---|' + '---|' * len(cols)]
        for name, d in table.items():
            cells = []
            for c in cols:
                if c == 'switches' and name in ('Fixed-time', 'Actuated'):
                    cells.append('n/a')          # SUMO runs these programs; switches are not counted
                else:
                    cells.append(fmt(d[c], 0 if c in ('throughput', 'switches', 'backlog', 'max_delay_s') else 2))
            L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        L.append('')
    L += ['## Before vs after training', '',
          'v1 scenarios use average wait; the new junctions use average delay.', '',
          '| Scenario | Learner | Wait or delay before (s) | After (s) | Change | Queue before | Queue after | Change |',
          '|---|---|---|---|---|---|---|---|']
    for r in out['before_after']:
        w, q = r[key(r['scenario'])], r['avg_queue']
        L.append(f'| {r["scenario"]} | {r["learner"]} | {w["before"]:.1f} | {w["after"]:.1f} | '
                 f'{w["change_pct"]:+.0f}% | {q["before"]:.2f} | {q["after"]:.2f} | {q["change_pct"]:+.0f}% |')
    for section, title, base in (('vs_actuated', 'Trained learners vs Actuated control', 'Actuated'),
                                 ('vs_lqf', 'Trained learners vs longest queue first', 'longest queue first')):
        L += ['', f'## {title}', '',
              f'Paired difference per seed (learner minus {base}), mean and 95% bootstrap CI. '
              'Negative is better for the learner. "tie" means the CI includes zero.', '',
              '| Scenario | Learner | Wait (s) | Delay (s) | Worst delay (s) | Travel time (s) | Queue (veh) |',
              '|---|---|---|---|---|---|---|']
        for r in out[section]:
            L.append(f'| {r["scenario"]} | {r["learner"]} | {verdict(r["avg_wait_s"])} | {verdict(r["avg_delay_s"])} | '
                     f'{verdict(r["max_delay_s"])} | {verdict(r["avg_travel_s"])} | {verdict(r["avg_queue"])} |')
    L += ['', '## Fairness on the new junctions', '',
          'Worst single-vehicle delay (mean ± CI half-width over seeds) and average delay per arm.', '']
    for sc, table in out['fairness'].items():
        arms = list(next(iter(table.values()))['arm_delay'])
        L += [f'### {TITLES[sc]}', '',
              '| Controller | Worst delay (s) | ' + ' | '.join(f'Delay {a} (s)' for a in arms) + ' |',
              '|---|---|' + '---|' * len(arms)]
        for name, d in table.items():
            L.append(f'| {name} | {fmt(d["max_delay_s"], 0)} | '
                     + ' | '.join(f'{d["arm_delay"].get(a, 0):.1f}' for a in arms) + ' |')
        L.append('')
    L += ['## Lusaka demand sweep', '',
          'The peak volumes are estimates, so every controller is also tested at 75% and 125% of them.', '',
          '| Demand | Controller | Avg delay (s) | Worst delay (s) | Avg wait (s) | Throughput (veh) | Backlog (veh) |',
          '|---|---|---|---|---|---|---|']
    for scale, table in out['sweep'].items():
        for name, d in table.items():
            L.append(f'| x{scale} | {name} | {d["avg_delay_s"]:.1f} | {d["max_delay_s"]:.0f} | '
                     f'{d["avg_wait_s"]:.1f} | {d["throughput"]:.0f} | {d["backlog"]:.0f} |')
    L += ['', '## Q-value probes', '',
          '| Agent | Probe | Right answer | Agent chose | Q-values | Correct |', '|---|---|---|---|---|---|']
    for agent, res in out['probes'].items():
        for p in res:
            ok = 'yes' if p['ok'] else ('no' if p['seen'] else 'no (state never visited)')
            q = ', '.join(f'{v:.2f}' for v in p['q'])
            L.append(f'| {agent} | {p["probe"]} | {p["expected"]} | {p["chosen"]} | {q} | {ok} |')
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
