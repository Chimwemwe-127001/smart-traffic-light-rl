"""SEMMA data mining on the detector data, before any agent is trained.

SEMMA (SAS Institute): Sample, Explore, Modify, Model, Assess.
This script does the first three steps. Its output decides how the agents
see the world, so the state design comes from data and not from guesswork.

    Sample  - run the baseline controllers on every scenario and log what the
              detectors report every second
    Explore - distributions, demand over time, lane correlation, queue by phase
    Modify  - derive the Q-learning bin edges and the DQN input scale from the data

Model and Assess are the training and evaluation scripts.

Outputs (results/semma/):
    detector_samples.csv.gz   the sampled dataset
    summary.json              the numbers quoted in docs/SEMMA.md
    state_config.json         bins + scale read by the agents
    *.png                     the exploration figures

Usage:
    python experiments/semma.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic_rl.env import TrafficEnv
from traffic_rl.scenarios import SCENARIOS as ALL_SCENARIOS, arms, n_actions

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'results', 'semma')
SCENARIOS = ['single', 'corridor', 'corridor_heavy']           # v1 study, one shared config ("default")
NEW_SCENARIOS = ['four_way', 'lusaka']                           # v1.1, one config each
PROGRAMS = ['fixed', 'actuated', 'random']
SAMPLE_SEEDS = [900, 901, 902]          # not used for training or evaluation


# ------------------------------------------------------------------ Sample

def sample_episode(scenario, program, seed):
    env = TrafficEnv(scenario, program='agent' if program == 'random' else program, record=True)
    rng = np.random.default_rng(seed)
    n = n_actions(scenario)
    ready, _ = env.reset(seed)
    if program == 'random':
        while ready:
            ready, _, _ = env.step({tls: int(rng.integers(n)) for tls in ready})
    else:
        env.run_to_end()
    env.close()
    for r in env.records:
        r.update(scenario=scenario, program=program, seed=seed)
    return env.records


def sample(scenarios=SCENARIOS):
    rows = []
    for sc in scenarios:
        for prog in PROGRAMS:
            for seed in SAMPLE_SEEDS:
                rows += sample_episode(sc, prog, seed)
                print(f'sampled {sc:15s} {prog:9s} seed {seed}  ({len(rows)} rows)')
    return rows


def save_csv(rows, path):
    import csv
    import gzip
    keys = sorted({k for r in rows for k in r})
    import io
    # mtime=0: the file only changes when the data does
    with io.TextIOWrapper(gzip.GzipFile(path, 'wb', mtime=0), newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------- Explore

def column(rows, key, **where):
    return np.array([r[key] for r in rows
                     if all(r.get(k) == v for k, v in where.items()) and key in r], dtype=float)


def explore(rows):
    summary = {}

    # 1. How many vehicles does one approach camera see?
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, sc in zip(axes, SCENARIOS):
        counts = np.concatenate([column(rows, 'EB', scenario=sc), column(rows, 'SB', scenario=sc)])
        ax.hist(counts, bins=np.arange(0, counts.max() + 2) - 0.5, color='tab:blue', alpha=0.8)
        ax.set_title(sc)
        ax.set_xlabel('vehicles seen on one approach')
        summary[f'{sc}_approach_count'] = {
            'mean': float(counts.mean()), 'median': float(np.median(counts)),
            'p90': float(np.percentile(counts, 90)), 'max': float(counts.max()),
            'share_zero': float(np.mean(counts == 0)),
        }
    axes[0].set_ylabel('seconds')
    fig.suptitle('Explore 1: approach counts are right-skewed with many empty seconds')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'explore_approach_counts.png'), dpi=120)
    plt.close(fig)

    # 2. Demand over time on the single intersection (fixed-time, averaged over seeds)
    fig, ax = plt.subplots(figsize=(10, 4))
    for d, c in (('EB', 'tab:blue'), ('SB', 'tab:orange')):
        series = np.array([[r[d] for r in rows if r['scenario'] == 'single' and r['program'] == 'fixed'
                            and r['seed'] == s] for s in SAMPLE_SEEDS])
        smooth = np.convolve(series.mean(0), np.ones(60) / 60, mode='same')
        ax.plot(np.arange(1, len(smooth) + 1), smooth, color=c, label=d)
    for x in (400, 800):
        ax.axvline(x, color='0.5', ls='--')
    ax.set_xlabel('time (s)'); ax.set_ylabel('vehicles seen (60 s mean)')
    ax.set_title('Explore 2: demand shifts from EB to SB over the episode (single, fixed-time)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'explore_demand_over_time.png'), dpi=120)
    plt.close(fig)

    # 3. Are lanes of the same approach redundant?
    single = [r for r in rows if r['scenario'] == 'single']
    lanes = np.array([[r[f'lane_{k}'] for k in range(6)] for r in single])
    corr = np.corrcoef(lanes.T)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    names = ['EB0', 'EB1', 'EB2', 'SB0', 'SB1', 'SB2']
    ax.set_xticks(range(6), names); ax.set_yticks(range(6), names)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im)
    ax.set_title('Explore 3: lane count correlation (single)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'explore_lane_correlation.png'), dpi=120)
    plt.close(fig)
    within = [corr[i, j] for a in (range(3), range(3, 6)) for i in a for j in a if i < j]
    across = [corr[i, j] for i in range(3) for j in range(3, 6)]
    summary['lane_corr_within_approach'] = float(np.mean(within))
    summary['lane_corr_across_approach'] = float(np.mean(across))

    # 4. Queue on each approach depending on who has green
    fig, ax = plt.subplots(figsize=(7, 4))
    data, labels = [], []
    for g, gname in ((0, 'EB green'), (1, 'SB green')):
        for d in ('EB', 'SB'):
            data.append([r[d] for r in single if r['green'] == g])
            labels.append(f'{d} count\n{gname}')
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, 5), labels)
    ax.set_ylabel('vehicles seen')
    ax.set_title('Explore 4: the red approach builds up, the green one drains')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'explore_queue_by_phase.png'), dpi=120)
    plt.close(fig)

    # 5. Do the cameras see the real queue? Detector queue vs all stopped vehicles on the approach.
    #    A low ratio means the detectors miss part of the queue (this is how we found
    #    detectors that ended 10-16 m before the stop line).
    for sc in SCENARIOS:
        det = column(rows, 'queue', scenario=sc)
        edge = column(rows, 'edge_queue', scenario=sc)
        summary[f'{sc}_detector_coverage'] = float(det.sum() / max(1.0, edge.sum()))

    # 6. Baseline performance straight from the sample (reference for Assess)
    for sc in SCENARIOS:
        for prog in PROGRAMS:
            q = column(rows, 'edge_queue', scenario=sc, program=prog)
            summary[f'{sc}_{prog}_mean_queue_per_intersection'] = float(q.mean())
    return summary


# ------------------------------------------------------------------ Modify

def modify(rows, summary):
    """Turn the exploration into state design decisions."""
    counts = np.concatenate([column(rows, d) for d in ('EB', 'SB')])
    # Q-learning: bin approach counts at data quantiles so every bin is actually visited.
    # Quantiles over non-empty seconds; "empty" gets its own bin because it is so common.
    nonzero = counts[counts > 0]
    edges = sorted({int(np.ceil(np.percentile(nonzero, p))) for p in (25, 50, 75, 90)})
    # DQN: scale each lane count by its 99th percentile so inputs sit roughly in [0, 1].
    lane_vals = np.concatenate([column(rows, f'lane_{k}', scenario='single') for k in range(6)])
    lane_scale = float(np.percentile(lane_vals, 99))
    # Reward scale for the DQN: the 99th percentile of the detector queue.
    queue_scale = float(np.percentile(column(rows, 'queue'), 99))

    config = {
        'count_bin_edges': edges,          # bin = how many edges the count has reached (0 = empty)
        'lane_scale': lane_scale,
        'queue_scale': queue_scale,
        'green_time_edges': [20, 40],      # short / medium / long green, from the 10 s min and 42 s fixed plan
    }
    summary['modify'] = config
    return config


# ------------------------------------------------- v1.1: four-way and Lusaka

def explore_arms(rows, sc):
    """Per-arm exploration for one of the new junctions."""
    names = arms(sc, 'C')
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
    summary = {}

    # 1. Demand seen by each arm over time (fixed-time, mean over seeds, 60 s smoothing)
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, c in zip(names, colors):
        series = np.array([[r[name] for r in rows if r['scenario'] == sc and r['program'] == 'fixed'
                            and r['seed'] == s] for s in SAMPLE_SEEDS])
        smooth = np.convolve(series.mean(0), np.ones(60) / 60, mode='same')
        ax.plot(np.arange(1, len(smooth) + 1), smooth, color=c, lw=2, label=name)
    ax.set_xlabel('time (s)'); ax.set_ylabel('vehicles seen (60 s mean)')
    ax.set_title(f'{sc}: vehicles seen per arm under fixed-time control')
    ax.legend(ncol=len(names)); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'{sc}_explore_arm_demand.png'), dpi=120)
    plt.close(fig)

    # 2. Real queue per arm under each baseline controller
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    for k, prog in enumerate(PROGRAMS):
        means = [column(rows, f'queue_{n}', scenario=sc, program=prog).mean() for n in names]
        ax.bar(np.arange(len(names)) + (k - 1) * width, means, width, label=prog,
               color=['#8a8983', '#52514e', '#c3c2b7'][k], edgecolor='white')
        summary.update({f'{prog}_mean_queue_{n}': float(m) for n, m in zip(names, means)})
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel('stopped vehicles on the arm (mean)')
    ax.set_title(f'{sc}: which arm queues, by baseline controller')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'{sc}_explore_arm_queue.png'), dpi=120)
    plt.close(fig)

    counts = np.concatenate([column(rows, n, scenario=sc) for n in names])
    for n in names:
        c = column(rows, n, scenario=sc)
        summary[f'arm_count_{n}'] = {'mean': float(c.mean()), 'p90': float(np.percentile(c, 90)),
                                     'max': float(c.max()), 'share_zero': float(np.mean(c == 0))}
    summary['approach_count'] = {'mean': float(counts.mean()), 'median': float(np.median(counts)),
                                 'p90': float(np.percentile(counts, 90)), 'max': float(counts.max())}
    det, edge = column(rows, 'queue', scenario=sc), column(rows, 'edge_queue', scenario=sc)
    summary['detector_coverage'] = float(det.sum() / max(1.0, edge.sum()))
    return summary


def modify_for(rows, sc, summary):
    """Same rules as the v1 Modify step, applied to one junction's own data."""
    names = arms(sc, 'C')
    counts = np.concatenate([column(rows, n, scenario=sc) for n in names])
    nonzero = counts[counts > 0]
    edges = sorted({int(np.ceil(np.percentile(nonzero, p))) for p in (25, 50, 75, 90)})
    n_lanes = sum(len(a['detectors']) for a in ALL_SCENARIOS[sc]['intersections']['C']['approaches'].values())
    lanes = np.concatenate([column(rows, f'lane_{k}', scenario=sc) for k in range(n_lanes)])
    config = {'count_bin_edges': edges,
              'lane_scale': float(np.percentile(lanes, 99)),
              'queue_scale': float(np.percentile(column(rows, 'queue', scenario=sc), 99)),
              'green_time_edges': [20, 40]}
    summary['modify'] = config
    return config


def explore_counts_and_lanes(rows, sc, summary, config):
    """Two views that back the Modify decisions: count distributions against the
    chosen bin edges, and lane correlation (are lanes of one arm redundant?)."""
    names = arms(sc, 'C')
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']

    # 1. How many vehicles each arm's detectors see, with the Q-learning bin edges
    fig, ax = plt.subplots(figsize=(9, 4))
    top = max(column(rows, n, scenario=sc).max() for n in names)
    bins = np.arange(0, top + 2) - 0.5
    for name, c in zip(names, colors):
        ax.hist(column(rows, name, scenario=sc), bins=bins, histtype='step', lw=2, color=c, label=name)
    for e in config['count_bin_edges']:
        ax.axvline(e - 0.5, color='#52514e', ls='--', lw=1)
    ax.set_yscale('log')
    ax.set_xlabel('vehicles seen on one arm (dashed lines: Q-learning bin edges)')
    ax.set_ylabel('seconds (log scale)')
    ax.set_title(f'{sc}: vehicles seen per arm, all baseline controllers')
    ax.legend(ncol=len(names), loc='upper center', bbox_to_anchor=(0.5, -0.2)); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'{sc}_explore_arm_counts.png'), dpi=120)
    plt.close(fig)

    # 2. Lane correlation, labelled by arm
    approaches = ALL_SCENARIOS[sc]['intersections']['C']['approaches']
    lane_arm = [n for n, a in approaches.items() for _ in a['detectors']]
    lanes = np.array([[float(r[f'lane_{k}']) for k in range(len(lane_arm))] for r in rows if r['scenario'] == sc])
    corr = np.corrcoef(lanes.T)
    labels = [f'{a}{i}' for a, i in zip(lane_arm, [lane_arm[:k + 1].count(a) - 1 for k, a in enumerate(lane_arm)])]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im)
    ax.set_title(f'{sc}: lane count correlation')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'{sc}_explore_lane_correlation.png'), dpi=120)
    plt.close(fig)
    pairs = [(i, j) for i in range(len(lane_arm)) for j in range(i + 1, len(lane_arm))]
    within = [corr[i, j] for i, j in pairs if lane_arm[i] == lane_arm[j]]
    across = [corr[i, j] for i, j in pairs if lane_arm[i] != lane_arm[j]]
    if within:
        summary['lane_corr_within_arm'] = float(np.mean(within))
    summary['lane_corr_across_arms'] = float(np.mean(across))


def coverage_check(sc, reach_m):
    """Detector coverage (detector queue / real queue on the arms) with the main-road
    detectors reaching reach_m metres back. Uses a temporary detector file outside the
    repo, so the committed network is not touched. Fixed-time control, sample seeds."""
    import tempfile
    sys.path.insert(0, os.path.join(REPO, 'networks'))
    import build_networks as bn
    from traffic_rl.env import traci
    import sumolib

    cfg = ALL_SCENARIOS[sc]
    names = arms(sc, 'C')
    net_file = cfg['cfg'].replace('.sumocfg', '.net.xml')
    tmp = tempfile.mkdtemp(prefix='coverage_')
    reach = {f'{a}2C': reach_m for a in bn.MAIN_ARMS} if sc == 'lusaka' else {f'{a}2C': reach_m for a in names}
    bn.write_detectors(sumolib.net.readNet(net_file), tmp, [[f'{a}2C'] for a in names], reach=reach)
    det_file = os.path.join(tmp, 'det.add.xml')
    edges = [e for a in cfg['intersections']['C']['approaches'].values() for e in a['edges']]
    det_q = edge_q = 0.0
    for seed in SAMPLE_SEEDS:
        traci.start([sumolib.checkBinary('sumo'), '-c', cfg['cfg'], '-a', det_file, '--seed', str(seed),
                     '--step-length', '1', '--time-to-teleport', '-1', '--no-warnings', '--no-step-log'])
        dets = traci.lanearea.getIDList()
        for _ in range(1200):
            traci.simulationStep()
            det_q += sum(traci.lanearea.getLastStepHaltingNumber(d) for d in dets)
            edge_q += sum(traci.edge.getLastStepHaltingNumber(e) for e in edges)
        traci.close()
    return float(det_q / max(1.0, edge_q))


def load_json(name, default):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', nargs='+', choices=['v1', 'v1.1'], default=['v1', 'v1.1'])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    config = load_json('state_config.json', {})
    if 'count_bin_edges' in config:                 # older single-config file
        config = {'default': config}
    summary = load_json('summary.json', {})

    if 'v1' in args.groups:
        rows = sample()
        save_csv(rows, os.path.join(OUT, 'detector_samples.csv.gz'))
        v1 = explore(rows)
        config['default'] = modify(rows, v1)
        v1['n_rows'] = len(rows)
        summary.update(v1)
    if 'v1.1' in args.groups:
        rows = sample(NEW_SCENARIOS)
        save_csv(rows, os.path.join(OUT, 'detector_samples_v1_1.csv.gz'))
        for sc in NEW_SCENARIOS:
            s = explore_arms(rows, sc)
            config[sc] = modify_for(rows, sc, s)
            explore_counts_and_lanes(rows, sc, s, config[sc])
            s['n_rows'] = sum(1 for r in rows if r['scenario'] == sc)
            summary[sc] = s
        # The Lusaka detector decision, reproduced: coverage with the original 105 m
        # detectors vs the 250 m detectors now used on Great East Road
        summary['lusaka']['coverage_check'] = {f'{m}m': coverage_check('lusaka', m) for m in (105, 250)}

    with open(os.path.join(OUT, 'state_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in NEW_SCENARIOS if k in summary}, indent=2))


if __name__ == '__main__':
    main()
