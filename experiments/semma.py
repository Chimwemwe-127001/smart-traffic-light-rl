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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'results', 'semma')
SCENARIOS = ['single', 'corridor', 'corridor_heavy']
PROGRAMS = ['fixed', 'actuated', 'random']
SAMPLE_SEEDS = [900, 901, 902]          # not used for training or evaluation


# ------------------------------------------------------------------ Sample

def sample_episode(scenario, program, seed):
    env = TrafficEnv(scenario, program='agent' if program == 'random' else program, record=True)
    rng = np.random.default_rng(seed)
    ready, _ = env.reset(seed)
    if program == 'random':
        while ready:
            ready, _, _ = env.step({tls: int(rng.integers(2)) for tls in ready})
    else:
        env.run_to_end()
    env.close()
    for r in env.records:
        r.update(scenario=scenario, program=program, seed=seed)
    return env.records


def sample():
    rows = []
    for sc in SCENARIOS:
        for prog in PROGRAMS:
            for seed in SAMPLE_SEEDS:
                rows += sample_episode(sc, prog, seed)
                print(f'sampled {sc:15s} {prog:9s} seed {seed}  ({len(rows)} rows)')
    return rows


def save_csv(rows, path):
    import csv
    import gzip
    keys = sorted({k for r in rows for k in r})
    with gzip.open(path, 'wt', newline='') as f:
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


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = sample()
    save_csv(rows, os.path.join(OUT, 'detector_samples.csv.gz'))
    summary = explore(rows)
    config = modify(rows, summary)
    summary['n_rows'] = len(rows)
    with open(os.path.join(OUT, 'state_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
