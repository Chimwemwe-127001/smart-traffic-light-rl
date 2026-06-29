"""The three traffic scenarios used in every experiment.

A scenario says which SUMO files to load, which traffic lights are agents,
and which lane-area detectors (the "cameras") each agent can see.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORKS = os.path.join(REPO, 'networks')

SCENARIOS = {
    # One intersection, demand shifts from EB-heavy to SB-heavy over the episode.
    'single': {
        'cfg': os.path.join(NETWORKS, 'single', 'RL.sumocfg'),
        'routes': os.path.join(NETWORKS, 'single', 'RL.rou.xml'),
        'actuated': os.path.join(NETWORKS, 'single', 'actuated.add.xml'),
        'intersections': {
            'Node2': {
                'EB': ['Node1_2_EB_0', 'Node1_2_EB_1', 'Node1_2_EB_2'],
                'SB': ['Node2_7_SB_0', 'Node2_7_SB_1', 'Node2_7_SB_2'],
                'edges': ['Node1_2_EB', 'Node2_7_SB'],
                'neighbor': None,
            },
        },
    },
    # Two intersections on an EB corridor, moderate demand.
    'corridor': {
        'cfg': os.path.join(NETWORKS, 'corridor', 'MultiAgent.sumocfg'),
        'routes': os.path.join(NETWORKS, 'corridor', 'multiagent.rou.xml'),
        'actuated': os.path.join(NETWORKS, 'corridor', 'actuated.add.xml'),
        'intersections': {
            'Node2': {
                'EB': ['Node1_2_EB_0', 'Node1_2_EB_1', 'Node1_2_EB_2'],
                'SB': ['Node7_2_SB_0', 'Node7_2_SB_1'],
                'edges': ['Node1_2_EB', 'Node7_2_SB'],
                'neighbor': 'Node5',
            },
            'Node5': {
                'EB': ['Node2_5_EB_0', 'Node2_5_EB_1', 'Node2_5_EB_2'],
                'SB': ['Node9_5_SB_0', 'Node9_5_SB_1'],
                'edges': ['Node2_5_EB', 'Node9_5_SB'],
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
