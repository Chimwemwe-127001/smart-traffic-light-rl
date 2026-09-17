"""Evaluate every controller on held-out traffic and score the rubric.

Protocol
    - 10 evaluation seeds (1000-1009), never used in training or validation
    - every controller runs on exactly the same seeds (paired comparison)
    - every learner was trained in 5 independent runs; all 5 are evaluated, and
      the confidence intervals resample both runs and seeds (traffic_rl/metrics.py)
    - learners act greedily (no exploration, no learning)
    - "untrained" = the same agent with its initial parameters (per run),
      so before vs after isolates what training added
    - two fixed-time plans: the hand-set one, and one timed with Webster's method
      (experiments/webster.py), the fair fixed-time baseline

Scenarios
    v1:   single, corridor, corridor_heavy
    v1.1: four_way (one lane each way, split phasing) and lusaka
          (Great East Road / Lufubu Road, plus a demand sweep x0.75 / x1.25)

Outputs:
    results/evaluation.csv   one row per (scenario, controller, run, seed)
    results/summary.json     everything below as data
    results/RESULTS.md       tables, run-to-run spread, fairness, probes, sweep and rubric

Usage:
    python experiments/evaluate.py [--jobs N]
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from traffic_rl.baselines import Actuated, FixedTime, LongestQueueFirst, RandomPolicy, Webster
from traffic_rl.dqn import DQN
from traffic_rl.metrics import bootstrap_ci_runs, paired_difference_runs, percent_change
from traffic_rl.runner import run_episode
from traffic_rl.scenarios import SCENARIOS
from train import make_agent, model_path

REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, 'results')
LOGS = os.path.join(RESULTS, 'logs')
EVAL_SEEDS = list(range(1000, 1010))
RUNS = list(range(5))
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
TAG = {'Q-learning': 'q_learning', 'Independent QL': 'q_learning', 'DQN': 'dqn', 'Coordinated QL': 'coordinated'}
WEBSTER, HAND_SET = 'Fixed-time (Webster)', 'Fixed-time (hand-set)'
NOT_COUNTED = (HAND_SET, WEBSTER, 'Actuated')        # SUMO runs these programs; switches are not counted


def key(scenario):
    """Headline metric. The new junctions can back up past the network edge, so they use
    total delay (stopped + queued before entering); v1 keeps average wait, as published."""
    return 'avg_delay_s' if scenario in V11 else 'avg_wait_s'


def baseline_names(scenario):
    names = [HAND_SET, WEBSTER, 'Actuated', 'Random']
    return names + ['Longest queue first'] if scenario in V11 else names


def learner_names(scenario):
    return [name for sc, name in LEARNERS if sc == scenario]


def make_controller(scenario, spec):
    """spec: ('baseline', name) or ('learner', name, run, trained)."""
    if spec[0] == 'baseline':
        return {HAND_SET: FixedTime, WEBSTER: Webster, 'Actuated': Actuated,
                'Random': lambda: RandomPolicy(seed=7, scenario=scenario),
                'Longest queue first': lambda: LongestQueueFirst(scenario)}[spec[1]]()
    _, name, run, trained = spec
    agent = make_agent(TAG[name], scenario, run)
    if trained:
        path = model_path(TAG[name], scenario, run)
        if not os.path.exists(path):
            sys.exit(f'missing {os.path.relpath(path, REPO)}: run experiments/train_all.py first')
        agent.load(path)
    return agent


def run_seeds(task):
    """One controller on all evaluation seeds, in order (worker process)."""
    scenario, spec, routes = task
    ctrl = make_controller(scenario, spec)
    return [run_episode(scenario, ctrl, seed=s, routes=routes) for s in EVAL_SEEDS]


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

def evaluate(pool, scenario, names, per_seed, label=None, routes=None):
    """Run the named controllers. Baselines give arrays of shape (seeds,); learners
    '<name> (trained)' and '<name> (untrained)' give (runs, seeds)."""
    label = label or scenario
    tasks, keys = [], []
    for name in names:
        base = name.split(' (')[0]
        if base in learner_names(scenario):
            for run in RUNS:
                tasks.append((scenario, ('learner', base, run, name.endswith('(trained)')), routes))
                keys.append((name, run))
        else:
            tasks.append((scenario, ('baseline', name), routes))
            keys.append((name, None))
    results = list(pool.map(run_seeds, tasks))
    runs = {}
    for (name, run), ms in zip(keys, results):
        for s, m in zip(EVAL_SEEDS, ms):
            per_seed.append({'scenario': label, 'controller': name, 'run': '' if run is None else run,
                             'seed': s, **{k: m[k] for k in METRICS}})
        d = runs.setdefault(name, {k: [] for k in METRICS + ['arm_delay']})
        for k in METRICS:
            d[k].append([m[k] for m in ms])
        d['arm_delay'] += [m['arm_delay'] for m in ms]
    for name, d in runs.items():
        for k in METRICS:
            d[k] = np.array(d[k][0] if len(d[k]) == 1 and name in baseline_names(scenario) else d[k], dtype=float)
        print(f'{label:15s} {name:28s} queue {d["avg_queue"].mean():6.2f}  wait {d["avg_wait_s"].mean():6.1f}s  '
              f'delay {d["avg_delay_s"].mean():6.1f}s  worst {d["max_delay_s"].mean():6.1f}s', flush=True)
    return runs


def all_names(scenario):
    return baseline_names(scenario) + [f'{n} ({s})' for n in learner_names(scenario) for s in ('untrained', 'trained')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    per_seed = []
    with ProcessPoolExecutor(args.jobs) as pool:
        runs = {sc: evaluate(pool, sc, all_names(sc), per_seed) for sc in V1 + V11}
        # Lusaka demand sweep: the estimated peak is uncertain, so repeat at x0.75 and x1.25.
        # Every plan and agent stays as designed or trained for x1.00.
        sweep_names = [HAND_SET, WEBSTER, 'Actuated', 'Longest queue first', 'Q-learning (trained)', 'DQN (trained)']
        sweep = {scale: evaluate(pool, 'lusaka', sweep_names, per_seed, label=f'lusaka_x{scale:.2f}', routes=routes)
                 for scale, routes in SCENARIOS['lusaka']['demand_sweep'].items()}
    sweep[1.0] = {k: v for k, v in runs['lusaka'].items() if k in sweep_names}

    with open(os.path.join(RESULTS, 'evaluation.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'controller', 'run', 'seed'] + METRICS)
        w.writeheader()
        w.writerows(per_seed)

    summary = {sc: {name: {k: bootstrap_ci_runs(v) for k, v in d.items() if k in METRICS} for name, d in r.items()}
               for sc, r in runs.items()}

    # Before vs after training (same runs, same seeds: run k before vs run k after)
    before_after = []
    for sc, name in LEARNERS:
        b, a = runs[sc][f'{name} (untrained)'], runs[sc][f'{name} (trained)']
        row = {'scenario': sc, 'learner': name}
        for k in ('avg_queue', 'avg_wait_s', 'avg_travel_s', 'avg_delay_s'):
            row[k] = {'before': float(b[k].mean()), 'after': float(a[k].mean()),
                      'change_pct': percent_change(a[k].mean(), b[k].mean()),
                      'diff_ci': paired_difference_runs(a[k], b[k])}
        before_after.append(row)

    # Every trained learner against the strongest baselines, seed by seed
    def versus(baseline, pairs):
        out = []
        for sc, name in pairs:
            row = {'scenario': sc, 'learner': name}
            for k in ('avg_wait_s', 'avg_delay_s', 'max_delay_s', 'avg_travel_s', 'avg_queue'):
                row[k] = paired_difference_runs(runs[sc][f'{name} (trained)'][k], runs[sc][baseline][k])
            out.append(row)
        return out
    vs_webster = versus(WEBSTER, LEARNERS)
    vs_actuated = versus('Actuated', LEARNERS)
    vs_lqf = versus('Longest queue first', [(sc, n) for sc, n in LEARNERS if sc in V11])

    # Run to run: each training run's own mean on the headline metric
    spread = []
    for sc, name in LEARNERS:
        per_run = runs[sc][f'{name} (trained)'][key(sc)].mean(axis=1)
        spread.append({'scenario': sc, 'learner': name, 'metric': key(sc), 'runs': [float(v) for v in per_run],
                       'webster': float(runs[sc][WEBSTER][key(sc)].mean()),
                       'actuated': float(runs[sc]['Actuated'][key(sc)].mean())})

    # Fairness on the new junctions: worst single delay and average delay per arm
    fairness = {}
    for sc in V11:
        fairness[sc] = {}
        arm_names = SCENARIOS[sc]['intersections']['C']['approaches']
        for name, d in runs[sc].items():
            fairness[sc][name] = {'max_delay_s': bootstrap_ci_runs(d['max_delay_s']),
                                  'arm_delay': {a: float(np.mean([ad.get(f'C:{a}', 0.0) for ad in d['arm_delay']]))
                                                for a in arm_names}}

    sweep_out = {f'{s:.2f}': {name: {k: float(d[k].mean()) for k in
                                     ('avg_delay_s', 'max_delay_s', 'avg_wait_s', 'throughput', 'backlog')}
                              for name, d in sweep[s].items()} for s in sorted(sweep)}

    probes = {}
    for sc, name in LEARNERS:
        if sc in PROBES:
            per_run = [run_probes(sc, make_controller(sc, ('learner', name, r, True))) for r in RUNS]
            probes[f'{name} ({sc})'] = [
                {'probe': p['probe'], 'expected': p['expected'], 'n_ok': sum(res[i]['ok'] for res in per_run),
                 'runs': [{k: res[i][k] for k in ('chosen', 'q', 'seen', 'ok')} for res in per_run]}
                for i, p in enumerate(per_run[0])]
    td = {f'{TAG[name]}_{sc}': [td_loss_trend(f'{TAG[name]}_{sc}_run{r}') for r in RUNS] for sc, name in LEARNERS}

    rubric = score_rubric(runs, probes, td)
    out = {'summary': summary, 'before_after': before_after, 'vs_webster': vs_webster, 'vs_actuated': vs_actuated,
           'vs_lqf': vs_lqf, 'spread': spread, 'fairness': fairness, 'sweep': sweep_out, 'probes': probes,
           'td_loss': td, 'rubric': rubric, 'eval_seeds': EVAL_SEEDS, 'runs': RUNS}
    with open(os.path.join(RESULTS, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    write_markdown(out)
    print('wrote results/evaluation.csv, results/summary.json, results/RESULTS.md')


# ------------------------------------------------------------------ rubric

MAJORITY = 3            # a per-run check passes if at least 3 of the 5 runs pass it


def score_rubric(runs, probes, td):
    rows = []

    def add(criterion, measure, threshold, result, passed):
        rows.append({'criterion': criterion, 'measure': measure, 'threshold': threshold,
                     'result': result, 'pass': bool(passed)})

    label = lambda sc: 'avg delay' if sc in V11 else 'avg wait'
    n = len(RUNS)
    for sc, name in LEARNERS:
        m, lo, hi = paired_difference_runs(runs[sc][f'{name} (trained)'][key(sc)],
                                           runs[sc][f'{name} (untrained)'][key(sc)])
        add(f'{name} learned ({sc})', f'{label(sc)}, trained minus untrained, 95% CI over runs and seeds',
            'CI below 0', f'{m:+.1f} s [{lo:+.1f}, {hi:+.1f}]', hi < 0)
    for sc, name in LEARNERS:
        a = runs[sc][f'{name} (trained)'][key(sc)]
        for base in (WEBSTER, 'Random'):
            m, lo, hi = paired_difference_runs(a, runs[sc][base][key(sc)])
            add(f'{name} beats {base} ({sc})', f'{label(sc)}, trained minus {base}, 95% CI',
                'CI below 0', f'{m:+.1f} s [{lo:+.1f}, {hi:+.1f}]', hi < 0)
    for sc, name in LEARNERS:
        a = runs[sc][f'{name} (trained)'][key(sc)]
        act = runs[sc]['Actuated'][key(sc)]
        pct = percent_change(a.mean(), act.mean())
        m, lo, hi = paired_difference_runs(a, act)
        add(f'{name} competitive with Actuated ({sc})', f'{label(sc)} vs Actuated, 95% CI',
            'within +10%', f'{pct:+.1f}% ({m:+.2f} s [{lo:+.2f}, {hi:+.2f}])', pct <= 10)
    for sc, name in LEARNERS:
        if sc not in V11:
            continue
        m, lo, hi = paired_difference_runs(runs[sc][f'{name} (trained)'][key(sc)],
                                           runs[sc]['Longest queue first'][key(sc)])
        add(f'{name} beats longest queue first ({sc})', f'{label(sc)}, trained minus LQF, 95% CI',
            'CI below 0', f'{m:+.1f} s [{lo:+.1f}, {hi:+.1f}]', hi < 0)
        worst = runs[sc][f'{name} (trained)']['max_delay_s'].mean()
        act_worst = runs[sc]['Actuated']['max_delay_s'].mean()
        add(f'{name} starves no one ({sc})', 'worst single-vehicle delay vs Actuated',
            'at most 1.25 x Actuated', f'{worst:.0f} s vs {act_worst:.0f} s', worst <= 1.25 * act_worst)
    for name, res in probes.items():
        all_ok = sum(all(p['runs'][r]['ok'] for p in res) for r in range(n))
        add(f'{name} learned traffic logic', f'runs answering all {len(res)} hand-made probe states',
            f'at least {MAJORITY} of {n} runs', f'{all_ok} of {n}', all_ok >= MAJORITY)
    for tag, trends in td.items():
        stable = sum(last <= 1.5 * first for first, last in trends)
        add(f'Stable learning ({tag})', 'TD loss, last 20 vs first 20 episodes, per run',
            f'last <= 1.5 x first in at least {MAJORITY} of {n} runs', f'{stable} of {n}', stable >= MAJORITY)
    for sc in ('corridor', 'corridor_heavy'):
        m, lo, hi = paired_difference_runs(runs[sc]['Coordinated QL (trained)']['avg_wait_s'],
                                           runs[sc]['Independent QL (trained)']['avg_wait_s'], pair_runs=False)
        add(f'Coordination helps ({sc})', 'avg wait, coordinated minus independent, 95% CI',
            'CI below 0', f'{m:+.1f} s [{lo:+.1f}, {hi:+.1f}]', hi < 0)
    for sc, r in runs.items():
        for name, d in r.items():
            if '(trained)' in name:
                pct = 100 * d['throughput'].mean() / r[WEBSTER]['throughput'].mean()
                add(f'{name.replace(" (trained)", "")} serves all demand ({sc})',
                    'vehicles completed vs Webster fixed-time', 'at least 98%', f'{pct:.1f}%', pct >= 98)
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
    n_runs = len(out['runs'])
    L = ['# Results', '',
         f'Held-out evaluation on {len(out["eval_seeds"])} traffic seeds '
         f'({out["eval_seeds"][0]}-{out["eval_seeds"][-1]}), never seen in training. Every learner was trained in '
         f'{n_runs} independent runs, and all {n_runs} are evaluated. Values are mean ± half-width of the 95% '
         'bootstrap CI; for learners the bootstrap resamples both training runs and traffic seeds. '
         'Every episode has a 300 s warm-up that the metrics leave out. Generated by `experiments/evaluate.py`.', '']
    L += ['**Metrics.** Wait = time stopped, for vehicles that entered the network. Delay = time stopped '
          'plus time queued before entering, for every vehicle, including those still outside at the end. '
          'The new junctions (four-way, Lusaka) are judged on delay, because a controller that starves an arm '
          'can push its queue outside the network, where wait would not see it.', '',
          '**Fixed-time plans.** "Hand-set" is the plan stored in the network. "Webster" is timed with '
          "Webster's method from saturation flows measured in SUMO (`experiments/webster.py`, "
          '`results/webster.json`); it is the fair fixed-time baseline.', '']
    for sc, table in out['summary'].items():
        cols = (['avg_queue', 'avg_wait_s', 'avg_travel_s', 'throughput', 'switches'] if sc in V1 else
                ['avg_delay_s', 'max_delay_s', 'avg_wait_s', 'avg_queue', 'throughput', 'backlog', 'switches'])
        L += [f'## {TITLES[sc]}', '', '| Controller | ' + ' | '.join(LABELS[c] for c in cols) + ' |',
              '|---|' + '---|' * len(cols)]
        for name, d in table.items():
            cells = []
            for c in cols:
                if c == 'switches' and name in NOT_COUNTED:
                    cells.append('n/a')
                else:
                    cells.append(fmt(d[c], 0 if c in ('throughput', 'switches', 'backlog', 'max_delay_s') else 2))
            L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        L.append('')
    L += ['## Before vs after training', '',
          f'Mean over {n_runs} runs and all seeds. v1 scenarios use average wait; the new junctions use average delay.',
          '', '| Scenario | Learner | Wait or delay before (s) | After (s) | Change | Queue before | Queue after | Change |',
          '|---|---|---|---|---|---|---|---|']
    for r in out['before_after']:
        w, q = r[key(r['scenario'])], r['avg_queue']
        L.append(f'| {r["scenario"]} | {r["learner"]} | {w["before"]:.1f} | {w["after"]:.1f} | '
                 f'{w["change_pct"]:+.0f}% | {q["before"]:.2f} | {q["after"]:.2f} | {q["change_pct"]:+.0f}% |')
    L += ['', '## Run-to-run spread', '',
          'Each training run scored on its own (mean over the 10 seeds). A wide spread means the result of a '
          'single run would not be reliable.', '',
          '| Scenario | Learner | Metric | ' + ' | '.join(f'Run {r}' for r in out['runs'])
          + ' | Min to max | Webster | Actuated |', '|---|---|---|' + '---|' * (n_runs + 3)]
    for r in out['spread']:
        v = r['runs']
        L.append(f'| {r["scenario"]} | {r["learner"]} | {"delay" if r["metric"] == "avg_delay_s" else "wait"} (s) | '
                 + ' | '.join(f'{x:.1f}' for x in v)
                 + f' | {min(v):.1f} to {max(v):.1f} | {r["webster"]:.1f} | {r["actuated"]:.1f} |')
    for section, title, base in (('vs_webster', 'Trained learners vs Webster fixed-time', 'Webster fixed-time'),
                                 ('vs_actuated', 'Trained learners vs Actuated control', 'Actuated'),
                                 ('vs_lqf', 'Trained learners vs longest queue first', 'longest queue first')):
        L += ['', f'## {title}', '',
              f'Paired difference per seed (learner minus {base}), mean and 95% CI over runs and seeds. '
              'Negative is better for the learner. "tie" means the CI includes zero.', '',
              '| Scenario | Learner | Wait (s) | Delay (s) | Worst delay (s) | Travel time (s) | Queue (veh) |',
              '|---|---|---|---|---|---|---|']
        for r in out[section]:
            L.append(f'| {r["scenario"]} | {r["learner"]} | {verdict(r["avg_wait_s"])} | {verdict(r["avg_delay_s"])} | '
                     f'{verdict(r["max_delay_s"])} | {verdict(r["avg_travel_s"])} | {verdict(r["avg_queue"])} |')
    L += ['', '## Fairness on the new junctions', '',
          'Worst single-vehicle delay (mean ± CI half-width) and average delay per arm.', '']
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
          'The peak volumes are estimates, so every controller is also tested at 75% and 125% of them. '
          'The Webster plan and the agents stay as designed or trained for 100%.', '',
          '| Demand | Controller | Avg delay (s) | Worst delay (s) | Avg wait (s) | Throughput (veh) | Backlog (veh) |',
          '|---|---|---|---|---|---|---|']
    for scale, table in out['sweep'].items():
        for name, d in table.items():
            L.append(f'| x{scale} | {name} | {d["avg_delay_s"]:.1f} | {d["max_delay_s"]:.0f} | '
                     f'{d["avg_wait_s"]:.1f} | {d["throughput"]:.0f} | {d["backlog"]:.0f} |')
    L += ['', '## Q-value probes', '',
          f'Each trained run is asked for its preferred action in hand-made states. A tie (for example a state the '
          f'Q-table never visited) is not an answer. The Q-values of every run are in `summary.json`.', '',
          '| Agent | Probe | Right answer | Runs correct | Choice by run |', '|---|---|---|---|---|']
    for agent, res in out['probes'].items():
        for p in res:
            choices = ', '.join(r['chosen'] if r['seen'] else 'unvisited' for r in p['runs'])
            L.append(f'| {agent} | {p["probe"]} | {p["expected"]} | {p["n_ok"]} of {n_runs} | {choices} |')
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
