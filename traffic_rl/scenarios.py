"""The traffic scenarios used in every experiment.

A scenario says which SUMO files to load, which traffic lights are agents,
and, for each intersection, its approaches ("arms"). Each arm lists its
incoming edge(s) and the lane-area detectors (the "cameras") on them.
The order of the arms matters: it is the order of the state, and in
'select' mode action k means "give green to arm k".

action_mode
    'switch' - 2 arms, action 0 keep / 1 switch to the other arm (v1 scenarios)
    'select' - any number of arms, one arm green at a time (split phasing),
               action k = serve arm k next
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORKS = os.path.join(REPO, 'networks')


def arm(detectors, edges):
    return {'detectors': list(detectors), 'edges': list(edges)}


SCENARIOS = {
    # One intersection, demand shifts from EB-heavy to SB-heavy over the episode.
    'single': {
        'cfg': os.path.join(NETWORKS, 'single', 'RL.sumocfg'),
        'routes': os.path.join(NETWORKS, 'single', 'RL.rou.xml'),
        'actuated': os.path.join(NETWORKS, 'single', 'actuated.add.xml'),
        'webster': os.path.join(NETWORKS, 'single', 'webster.add.xml'),
        'action_mode': 'switch',
        'intersections': {
            'Node2': {
                'approaches': {
                    'EB': arm(['Node1_2_EB_0', 'Node1_2_EB_1', 'Node1_2_EB_2'], ['Node1_2_EB']),
                    'SB': arm(['Node2_7_SB_0', 'Node2_7_SB_1', 'Node2_7_SB_2'], ['Node2_7_SB']),
                },
                'neighbor': None,
            },
        },
    },
    # Two intersections on an EB corridor, moderate demand.
    'corridor': {
        'cfg': os.path.join(NETWORKS, 'corridor', 'MultiAgent.sumocfg'),
        'routes': os.path.join(NETWORKS, 'corridor', 'multiagent.rou.xml'),
        'actuated': os.path.join(NETWORKS, 'corridor', 'actuated.add.xml'),
        'webster': os.path.join(NETWORKS, 'corridor', 'webster.add.xml'),
        'action_mode': 'switch',
        'intersections': {
            'Node2': {
                'approaches': {
                    'EB': arm(['Node1_2_EB_0', 'Node1_2_EB_1', 'Node1_2_EB_2'], ['Node1_2_EB']),
                    'SB': arm(['Node7_2_SB_0', 'Node7_2_SB_1'], ['Node7_2_SB']),
                },
                'neighbor': 'Node5',
            },
            'Node5': {
                'approaches': {
                    'EB': arm(['Node2_5_EB_0', 'Node2_5_EB_1', 'Node2_5_EB_2'], ['Node2_5_EB']),
                    'SB': arm(['Node9_5_SB_0', 'Node9_5_SB_1'], ['Node9_5_SB']),
                },
                'neighbor': 'Node2',
            },
        },
    },
}

# Same corridor, EB demand close to saturation. Coordination should matter more here.
SCENARIOS['corridor_heavy'] = dict(
    SCENARIOS['corridor'],
    routes=os.path.join(NETWORKS, 'corridor', 'multiagent_heavy.rou.xml'),
    webster=os.path.join(NETWORKS, 'corridor', 'webster_heavy.add.xml'),      # timed for the heavy demand
)


# A realistic four-way junction: every road two-way, one lane each direction,
# left-hand traffic, split phasing (the agent picks which arm gets green).
# Built by networks/build_networks.py.
SCENARIOS['four_way'] = {
    'cfg': os.path.join(NETWORKS, 'four_way', 'four_way.sumocfg'),
    'routes': os.path.join(NETWORKS, 'four_way', 'four_way.rou.xml'),
    'actuated': os.path.join(NETWORKS, 'four_way', 'actuated.add.xml'),
    'webster': os.path.join(NETWORKS, 'four_way', 'webster.add.xml'),
    'action_mode': 'select',
    'select_by': 'avg_delay_s',       # checkpoint selection metric in train.py
    'intersections': {
        'C': {
            'approaches': {a: arm([f'{a}2C_0'], [f'{a}2C']) for a in ('N', 'E', 'S', 'W')},
            'neighbor': None,
        },
    },
}


# Great East Road / Manda Hill, Lusaka: the signalized junction of Great East Road
# with Manchinchi Road and Addis Ababa Drive, built from its OpenStreetMap data by
# networks/build_networks.py. Four greens chosen by position in the program, every
# turn protected: 0 Great East Road through and left, 1 Great East Road right turns,
# 2 Manchinchi Road, 3 Addis Ababa Drive. Free left turns use slip roads that the
# signal does not control. The arms, their edges and detectors are written by the
# builder to arms.json.
MANDA = os.path.join(NETWORKS, 'manda_hill')
with open(os.path.join(MANDA, 'arms.json')) as _f:
    _MANDA_ARMS = json.load(_f)
SCENARIOS['manda_hill'] = {
    'cfg': os.path.join(MANDA, 'manda_hill.sumocfg'),
    'routes': os.path.join(MANDA, 'manda_hill.rou.xml'),
    'actuated': os.path.join(MANDA, 'actuated.add.xml'),
    'webster': os.path.join(MANDA, 'webster.add.xml'),        # timed for x1.00, also used in the sweep
    'action_mode': 'select',
    'greens': 'order',
    'n_greens': 4,
    'select_by': 'avg_delay_s',
    'phase_arms': [['W', 'E'], ['W', 'E'], ['N'], ['S']],     # arms each green serves
    'demand_sweep': {s: os.path.join(MANDA, f'manda_hill_x{s:.2f}.rou.xml') for s in (0.75, 1.25)},
    'intersections': {
        'C': {
            # W: Great East Road from the city, E: from Manda Hill Mall, N: Manchinchi Road, S: Addis Ababa Drive
            'approaches': {a: arm(v['detectors'], v['approach']) for a, v in _MANDA_ARMS.items()},
            'neighbor': None,
        },
    },
}
# The detector lanes each green serves, as positions in the observation (arms W, E, N, S in
# order, lanes kerb first). On Great East Road the median-side lane is the right-turn lane.
_n = [len(v['detectors']) for v in _MANDA_ARMS.values()]
_w, _e = list(range(0, _n[0])), list(range(_n[0], _n[0] + _n[1]))
SCENARIOS['manda_hill']['phase_lanes'] = [_w[:-1] + _e[:-1], [_w[-1], _e[-1]],
                                          list(range(sum(_n[:2]), sum(_n[:3]))), list(range(sum(_n[:3]), sum(_n)))]


def arms(scenario, tls):
    return list(SCENARIOS[scenario]['intersections'][tls]['approaches'])


def n_actions(scenario):
    """2 for keep/switch scenarios, one action per arm for select scenarios."""
    sc = SCENARIOS[scenario]
    if sc['action_mode'] == 'switch':
        return 2
    return n_greens(scenario)


def n_greens(scenario):
    """Number of green phases: one per arm, unless the scenario's program says otherwise."""
    sc = SCENARIOS[scenario]
    return sc.get('n_greens') or max(len(i['approaches']) for i in sc['intersections'].values())
