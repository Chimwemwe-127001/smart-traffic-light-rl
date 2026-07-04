"""Turning raw detector observations into agent inputs.

The bin edges and scales come from the SEMMA step (results/semma/state_config.json),
so the state design is backed by the data the agents will actually see.
"""
import json
import os

import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results', 'semma', 'state_config.json')

# Used only if SEMMA has not been run yet.
DEFAULT_CONFIG = {'count_bin_edges': [2, 4, 7, 11], 'lane_scale': 8.0,
                  'queue_scale': 19.0, 'green_time_edges': [20, 40]}


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return dict(DEFAULT_CONFIG)


def count_bin(count, edges):
    """0 = fewer vehicles than the first edge, len(edges) = at or above the last edge."""
    return int(np.searchsorted(edges, count, side='right'))


def discrete_state(o, cfg):
    """Tabular state for one intersection: (EB bin, SB bin, green direction, green age bin)."""
    return (count_bin(o['EB'], cfg['count_bin_edges']),
            count_bin(o['SB'], cfg['count_bin_edges']),
            o['green'],
            count_bin(o['green_time'], cfg['green_time_edges']))


def feature_vector(o, cfg):
    """DQN input: per-lane counts, one-hot green direction, green age. All roughly in [0, 1]."""
    lanes = np.asarray(o['lanes'], dtype=float) / cfg['lane_scale']
    green = np.array([o['green'] == 0, o['green'] == 1], dtype=float)
    age = np.array([min(o['green_time'] / 60.0, 1.0)])
    return np.concatenate([lanes, green, age])
