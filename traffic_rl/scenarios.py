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
)


# A realistic four-way junction: every road two-way, one lane each direction,
# left-hand traffic, split phasing (the agent picks which arm gets green).
# Built by networks/build_networks.py.
SCENARIOS['four_way'] = {
    'cfg': os.path.join(NETWORKS, 'four_way', 'four_way.sumocfg'),
    'routes': os.path.join(NETWORKS, 'four_way', 'four_way.rou.xml'),
    'actuated': os.path.join(NETWORKS, 'four_way', 'actuated.add.xml'),
    'action_mode': 'select',
    'intersections': {
        'C': {
            'approaches': {a: arm([f'{a}2C_0'], [f'{a}2C']) for a in ('N', 'E', 'S', 'W')},
            'neighbor': None,
        },
    },
}


def arms(scenario, tls):
    return list(SCENARIOS[scenario]['intersections'][tls]['approaches'])


def n_actions(scenario):
    """2 for keep/switch scenarios, one action per arm for select scenarios."""
    sc = SCENARIOS[scenario]
    if sc['action_mode'] == 'switch':
        return 2
    return max(len(i['approaches']) for i in sc['intersections'].values())
