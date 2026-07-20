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
          'Q-learning': '#2a78d6', 'Independent QL': '#2a78d6',
          'DQN': '#eb6834', 'Coordinated QL': '#1baf7a'}
TITLES = {'single': 'Single intersection', 'corridor': 'Corridor, normal demand',
          'corridor_heavy': 'Corridor, heavy demand'}
LEARNERS = {'single': [('q_learning', 'Q-learning'), ('dqn', 'DQN')],
            'corridor': [('q_learning', 'Independent QL'), ('coordinated', 'Coordinated QL')],
            'corridor_heavy': [('q_learning', 'Independent QL'), ('coordinated', 'Coordinated QL')]}

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.8, 'axes.axisbelow': True, 'legend.frameon': False})


def plain_log_axis(ax, ticks=(2, 3, 4, 6, 10, 15, 20, 30)):
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


def validation_curves(summary):
    """Greedy validation wait time over training, with the baselines as reference lines."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (sc, learners) in zip(axes, LEARNERS.items()):
        for tag, label in learners:
            rows = read_csv(os.path.join(LOGS, f'val_{tag}_{sc}.csv'))
            ax.plot([int(r['episode']) for r in rows], [float(r['avg_wait_s']) for r in rows],
                    '-o', ms=4, lw=2, color=COLORS[label], label=label)
        for base in ('Fixed-time', 'Actuated'):
            y = summary['summary'][sc][base]['avg_wait_s'][0]
            ax.axhline(y, color=COLORS[base], lw=1.5, ls='--')
            ax.annotate(base, (1.0, y), xycoords=('axes fraction', 'data'), xytext=(4, 0),
                        textcoords='offset points', va='center', fontsize=9, color=MUTED)
        plain_log_axis(ax)
        ax.set_title(TITLES[sc], color=INK)
        ax.set_xlabel('training episode')
    axes[0].set_ylabel('avg waiting time per vehicle (s, log scale)')
    for ax in axes:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.suptitle('Learning curves: greedy policy on validation traffic (episode 0 = untrained)', color=INK)
    fig.tight_layout()
    save(fig, 'learning_curves.png')


def before_after(summary):
    rows = summary['before_after']
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(rows))
    w = 0.38
    for i, r in enumerate(rows):
        c = COLORS[r['learner']]
        b, a = r['avg_wait_s']['before'], r['avg_wait_s']['after']
        ax.bar(i - w / 2 - 0.01, b, w, color=lighten(c), edgecolor='white', linewidth=2)
        ax.bar(i + w / 2 + 0.01, a, w, color=c, edgecolor='white', linewidth=2)
        ax.annotate(f'{b:.1f}s', (i - w / 2, b), xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=8, color=MUTED)
        ax.annotate(f'{a:.1f}s', (i + w / 2, a), xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=8, color=INK)
    plain_log_axis(ax)
    ax.set_xticks(x, [f'{r["learner"]}\n{TITLES[r["scenario"]].replace(", ", chr(10))}' for r in rows], fontsize=8.5)
    ax.set_ylabel('avg waiting time per vehicle (s, log scale)')
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


def main():
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(RESULTS, 'summary.json')) as f:
        summary = json.load(f)
    validation_curves(summary)
    before_after(summary)
    head_to_head(summary)
    training_diagnostics()


if __name__ == '__main__':
    main()
