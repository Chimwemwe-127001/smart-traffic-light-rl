"""Build every figure used in the README from the logs and the evaluation.

    python experiments/make_figures.py

Colors follow the entity, not the chart: baselines are grays, and each
learner keeps one hue everywhere (blue = Q-learning / independent,
orange = DQN, aqua = coordinated). Untrained = the same hue, lighter.
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'results')
LOGS = os.path.join(RESULTS, 'logs')
FIG = os.path.join(RESULTS, 'figures')

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
COLORS = {'Fixed-time': '#8a8983', 'Actuated': '#52514e', 'Random': '#c3c2b7',
          'Longest queue first': '#7d93ad',
          'Q-learning': '#2a78d6', 'Independent QL': '#2a78d6',
          'DQN': '#eb6834', 'Coordinated QL': '#1baf7a'}
TITLES = {'single': 'Single intersection', 'corridor': 'Corridor, normal demand',
          'corridor_heavy': 'Corridor, heavy demand', 'four_way': 'Four-way, one lane each way',
          'lusaka': 'Lusaka, Great East Rd / Lufubu Rd'}
LEARNERS = {'single': [('q_learning', 'Q-learning'), ('dqn', 'DQN')],
            'corridor': [('q_learning', 'Independent QL'), ('coordinated', 'Coordinated QL')],
            'corridor_heavy': [('q_learning', 'Independent QL'), ('coordinated', 'Coordinated QL')]}
LEARNERS_V11 = {'four_way': [('q_learning', 'Q-learning'), ('dqn', 'DQN')],
                'lusaka': [('q_learning', 'Q-learning'), ('dqn', 'DQN')]}

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.8, 'axes.axisbelow': True, 'legend.frameon': False})


def plain_log_axis(ax, ticks=(2, 3, 4, 6, 10, 15, 20, 30, 50, 100, 200, 400, 800)):
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def lighten(hex_color, amount=0.55):
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(rgb + (1 - rgb) * amount)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', name)


def learning_curve(summary, sc, learners, baselines, metric):
    """One scenario per figure: the greedy policy on validation traffic every 10
    episodes, with the baselines' held-out test score as dashed reference lines."""
    label = {'avg_wait_s': 'avg waiting time per vehicle', 'avg_delay_s': 'avg delay per vehicle'}[metric]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ys = []
    for tag, name in learners:
        rows = read_csv(os.path.join(LOGS, f'val_{tag}_{sc}.csv'))
        y = [float(r[metric]) for r in rows]
        ys += y
        ax.plot([int(r['episode']) for r in rows], y, '-o', ms=3.5, lw=2, color=COLORS[name], label=name)
    for base in baselines:            # in the legend, not on the lines: some sit close together
        y = summary['summary'][sc][base][metric][0]
        ys.append(y)
        ax.axhline(y, color=COLORS[base], lw=1.5, ls='--', label=f'{base} (test)')
    candidates = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 80, 150, 300, 600)
    plain_log_axis(ax, ticks=[t for t in candidates if min(ys) * 0.8 <= t <= max(ys) * 1.2])
    ax.set_xlim(-15, 915)
    ax.set_xlabel('training episode (episode 0 = before training)')
    ax.set_ylabel(f'{label} (s, log scale)')
    ax.set_title(f'Learning curve: {TITLES[sc]}', color=INK)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=len(learners) + len(baselines), fontsize=9)
    fig.tight_layout()
    save(fig, f'learning_curve_{sc}.png')


def learning_curves(summary):
    for sc, learners in LEARNERS.items():
        learning_curve(summary, sc, learners, ('Fixed-time', 'Actuated'), 'avg_wait_s')
    for sc, learners in LEARNERS_V11.items():
        learning_curve(summary, sc, learners, ('Fixed-time', 'Actuated', 'Longest queue first'), 'avg_delay_s')


def before_after(summary):
    rows = summary['before_after']
    fig, ax = plt.subplots(figsize=(2.2 * len(rows), 5))
    x = np.arange(len(rows))
    w = 0.38
    short = {'single': 'Single\nintersection', 'corridor': 'Corridor\nnormal', 'corridor_heavy': 'Corridor\nheavy',
             'four_way': 'Four-way\n(delay)', 'lusaka': 'Lusaka\n(delay)'}
    for i, r in enumerate(rows):
        c = COLORS[r['learner']]
        metric = 'avg_delay_s' if r['scenario'] in LEARNERS_V11 else 'avg_wait_s'
        b, a = r[metric]['before'], r[metric]['after']
        ax.bar(i - w / 2 - 0.01, b, w, color=lighten(c), edgecolor='white', linewidth=2)
        ax.bar(i + w / 2 + 0.01, a, w, color=c, edgecolor='white', linewidth=2)
        ax.annotate(f'{b:.1f}s', (i - w / 2, b), xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=8, color=MUTED)
        ax.annotate(f'{a:.1f}s', (i + w / 2, a), xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=8, color=INK)
    plain_log_axis(ax)
    ax.set_xticks(x, [f'{r["learner"]}\n{short[r["scenario"]]}' for r in rows], fontsize=8.5)
    ax.set_ylabel('avg wait, or avg delay on the new junctions (s, log scale)')
    ax.legend(handles=[Patch(color='#c3c2b7', label='before training (lighter shade, initial parameters)'),
                       Patch(color=MUTED, label='after training (full color)')],
              loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.set_title('Before vs after training, held-out traffic (same seeds)', color=INK, pad=32)
    save(fig, 'before_after.png')


def head_to_head(summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    for ax, sc in zip(axes, LEARNERS):
        table = summary['summary'][sc]
        names = ['Fixed-time', 'Random', 'Actuated'] + [n for n in table if n.endswith('(trained)')]
        means = [table[n]['avg_wait_s'][0] for n in names]
        err = [[table[n]['avg_wait_s'][0] - table[n]['avg_wait_s'][1] for n in names],
               [table[n]['avg_wait_s'][2] - table[n]['avg_wait_s'][0] for n in names]]
        colors = [COLORS[n.replace(' (trained)', '')] for n in names]
        ax.bar(range(len(names)), means, 0.7, color=colors, edgecolor='white', linewidth=2,
               yerr=err, capsize=3, error_kw={'ecolor': INK, 'elinewidth': 1})
        for i, m in enumerate(means):
            ax.annotate(f'{m:.1f}', (i, m + err[1][i]), xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=8.5, color=INK)
        ax.set_xticks(range(len(names)), [n.replace(' (trained)', '').replace(' ', '\n', 1) for n in names],
                      fontsize=8.5)
        ax.set_title(TITLES[sc], color=INK)
    axes[0].set_ylabel('avg waiting time per vehicle (s)')
    fig.suptitle('Head to head on held-out traffic (mean and 95% CI over 10 seeds)', color=INK)
    fig.tight_layout()
    save(fig, 'head_to_head.png')


def training_diagnostics():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for sc, learners in LEARNERS.items():
        for tag, label in learners:
            rows = read_csv(os.path.join(LOGS, f'train_{tag}_{sc}.csv'))
            ep = [int(r['episode']) for r in rows]
            ls = {'single': '-.', 'corridor': '-', 'corridor_heavy': ':'}[sc]
            name = f'{label} ({TITLES[sc].lower()})'
            if tag == 'dqn':
                loss = [float(r['td_loss']) for r in rows]
                axes[1].plot(ep, loss, lw=2, color=COLORS[label], label=name)
            else:
                axes[2].plot(ep, [int(r['q_table_states']) for r in rows], lw=2, ls=ls,
                             color=COLORS[label], label=name)
            reward = np.array([float(r['total_reward']) for r in rows])
            if sc == 'single':
                smooth = np.convolve(reward, np.ones(10) / 10, mode='valid')
                axes[0].plot(ep[9:], smooth, lw=2, color=COLORS[label], label=label)
    axes[0].set_title('Training reward, single intersection (10-episode mean)', color=INK)
    axes[0].set_xlabel('episode'); axes[0].legend()
    axes[1].set_title('DQN TD loss (Huber)', color=INK)
    axes[1].set_xlabel('episode'); axes[1].legend()
    axes[2].set_title('Q-table growth: states discovered', color=INK)
    axes[2].set_xlabel('episode'); axes[2].legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'training_diagnostics.png')


# ------------------------------------------------------ v1.1: four-way and Lusaka

def bars_with_ci(ax, table, names, metric):
    means = [table[n][metric][0] for n in names]
    err = [[table[n][metric][0] - table[n][metric][1] for n in names],
           [table[n][metric][2] - table[n][metric][0] for n in names]]
    colors = [COLORS[n.replace(' (trained)', '')] for n in names]
    ax.bar(range(len(names)), means, 0.7, color=colors, edgecolor='white', linewidth=2,
           yerr=err, capsize=3, error_kw={'ecolor': INK, 'elinewidth': 1})
    for i, m in enumerate(means):
        ax.annotate(f'{m:.0f}' if m >= 100 else f'{m:.1f}', (i, m + err[1][i]), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=8.5, color=INK)
    short = {'Longest queue first': 'Longest\nqueue first'}
    ax.set_xticks(range(len(names)), [short.get(n, n.replace(' (trained)', '')) for n in names], fontsize=8.5)


V11_ORDER = ['Fixed-time', 'Random', 'Actuated', 'Longest queue first', 'Q-learning (trained)', 'DQN (trained)']


def head_to_head_v11(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, sc in zip(axes, LEARNERS_V11):
        bars_with_ci(ax, summary['summary'][sc], V11_ORDER, 'avg_delay_s')
        ax.set_title(TITLES[sc], color=INK)
    axes[0].set_ylabel('avg delay per vehicle (s)')
    fig.suptitle('Head to head on the new junctions (mean and 95% CI over 10 held-out seeds)', color=INK)
    fig.tight_layout()
    save(fig, 'head_to_head_v11.png')


def fairness(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, sc in zip(axes, LEARNERS_V11):
        bars_with_ci(ax, summary['summary'][sc], V11_ORDER, 'max_delay_s')
        ax.set_title(TITLES[sc], color=INK)
    axes[0].set_ylabel('worst single-vehicle delay (s)')
    fig.suptitle('Fairness: the longest delay any one driver had (mean and 95% CI over 10 seeds)', color=INK)
    fig.tight_layout()
    save(fig, 'fairness_v11.png')


def lusaka_sweep(summary):
    sweep = summary['sweep']
    scales = sorted(sweep, key=float)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name in sweep[scales[0]]:
        label = name.replace(' (trained)', '')
        ys = [sweep[s][name]['avg_delay_s'] for s in scales]
        ax.plot([float(s) for s in scales], ys, '-o', lw=2, ms=6, color=COLORS[label], label=label)
    ax.set_xticks([float(s) for s in scales], [f'x{float(s):.2f}' for s in scales])
    ax.set_xlabel('demand, relative to the estimated morning peak')
    ax.set_ylabel('avg delay per vehicle (s)')
    ax.set_title('Lusaka: how each controller copes as demand grows', color=INK)
    ax.legend(loc='upper left', fontsize=9)
    save(fig, 'lusaka_sweep.png')


def main():
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(RESULTS, 'summary.json')) as f:
        summary = json.load(f)
    learning_curves(summary)
    before_after(summary)
    head_to_head(summary)
    training_diagnostics()
    head_to_head_v11(summary)
    fairness(summary)
    lusaka_sweep(summary)


if __name__ == '__main__':
    main()
