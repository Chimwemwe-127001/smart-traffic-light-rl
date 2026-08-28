"""Turning raw detector observations into agent inputs.

The bin edges and scales come from the SEMMA step (results/semma/state_config.json),
so the state design is backed by the data the agents will actually see. The file
holds one config per scenario, plus "default" for the v1 networks.
"""
import json
import os

import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results', 'semma', 'state_config.json')

# Used only if SEMMA has not been run yet.
DEFAULT_CONFIG = {'count_bin_edges': [2, 4, 7, 11], 'lane_scale': 8.0,
                  'queue_scale': 19.0, 'green_time_edges': [20, 40]}


def load_config(scenario=None):
    if not os.path.exists(CONFIG_PATH):
        return dict(DEFAULT_CONFIG)
    with open(CONFIG_PATH) as f:
        data = json.load(f)
    if 'count_bin_edges' in data:            # older single-config file
        return data
    return data.get(scenario, data['default'])


def count_bin(count, edges):
    """0 = fewer vehicles than the first edge, len(edges) = at or above the last edge."""
    return int(np.searchsorted(edges, count, side='right'))


def discrete_state(o, cfg):
    """Tabular state for one intersection: a count bin per arm, the arm with green, green age bin."""
    return (tuple(count_bin(c, cfg['count_bin_edges']) for c in o['counts'])
            + (o['green'], count_bin(o['green_time'], cfg['green_time_edges'])))


def feature_vector(o, cfg):
    """DQN input: per-lane counts, one-hot green phase, green age. All roughly in [0, 1]."""
    lanes = np.asarray(o['lanes'], dtype=float) / cfg['lane_scale']
    green = np.array([o['green'] == k for k in range(o.get('n_greens', len(o['counts'])))], dtype=float)
    age = np.array([min(o['green_time'] / 60.0, 1.0)])
    return np.concatenate([lanes, green, age])


def n_features(n_lanes, n_greens):
    return n_lanes + n_greens + 1
