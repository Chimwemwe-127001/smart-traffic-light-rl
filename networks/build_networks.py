"""Build the v1.1 networks reproducibly with SUMO's netconvert.

    python networks/build_networks.py four_way

Zambia drives on the left, so every network is built with --lefthand: vehicles
keep left and the right turn is the one that crosses oncoming traffic.

Each signalized junction gets split phasing: one arm has green at a time
(green for all its movements, then 3 s yellow), which is common in Zambia.
The script writes, next to the network:
    <name>.tll.xml       the fixed-time program (30 s green + 3 s yellow per arm)
    actuated.add.xml     the same phases as gap-based actuated control (10 to 60 s)
    det.add.xml          one lane-area detector per approach lane, ending at the stop line
"""
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

if 'SUMO_HOME' not in os.environ:
    import sumo
    os.environ['SUMO_HOME'] = sumo.SUMO_HOME
import sumolib

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED_GREEN_S, YELLOW_S = 30, 3
MIN_GREEN_S, MAX_GREEN_S = 10, 60
DETECTOR_M = 105


def netconvert(*args):
    subprocess.run([sumolib.checkBinary('netconvert'), '--lefthand', '--no-turnarounds',
                    '--no-warnings', *args], check=True)


def split_phases(net, tls_id, arm_edges):
    """Link-state strings for split phasing: arm k green, then arm k yellow, for every arm in order."""
    conns = net.getTLS(tls_id).getConnections()          # [in_lane, out_lane, link_index]
    n_links = max(c[2] for c in conns) + 1
    phases = []
    for edges in arm_edges:
        mine = {c[2] for c in conns if c[0].getEdge().getID() in edges}
        phases.append(''.join('G' if i in mine else 'r' for i in range(n_links)))
        phases.append(''.join('y' if i in mine else 'r' for i in range(n_links)))
    return phases


def write_programs(net, folder, name, tls_id, arm_edges):
    phases = split_phases(net, tls_id, arm_edges)
    fixed = [f'        <phase duration="{FIXED_GREEN_S if "G" in s else YELLOW_S}" state="{s}"/>' for s in phases]
    with open(os.path.join(folder, f'{name}.tll.xml'), 'w') as f:
        f.write('<tlLogics>\n    <!-- Split phasing, fixed time: 30 s green + 3 s yellow per arm -->\n'
                f'    <tlLogic id="{tls_id}" type="static" programID="0" offset="0">\n'
                + '\n'.join(fixed) + '\n    </tlLogic>\n</tlLogics>\n')
    act = [f'        <phase duration="{FIXED_GREEN_S}" minDur="{MIN_GREEN_S}" maxDur="{MAX_GREEN_S}" state="{s}"/>'
           if 'G' in s else f'        <phase duration="{YELLOW_S}" state="{s}"/>' for s in phases]
    with open(os.path.join(folder, 'actuated.add.xml'), 'w') as f:
        f.write('<additional>\n    <!-- Same split phases as gap-based actuated control: a green is extended\n'
                '         while vehicles keep arriving, from 10 s up to 60 s. -->\n'
                f'    <tlLogic id="{tls_id}" type="actuated" programID="actuated" offset="0">\n'
                '        <param key="max-gap" value="3.0"/>\n        <param key="detector-gap" value="2.0"/>\n'
                + '\n'.join(act) + '\n    </tlLogic>\n</additional>\n')


def write_detectors(net, folder, arm_edges):
    """One detector per approach lane: the last 105 m (or the whole lane if shorter), ending at the stop line."""
    lines, names = [], []
    for edges in arm_edges:
        arm = []
        for e in edges:
            for lane in net.getEdge(e).getLanes():
                length = lane.getLength()
                start = max(0.1, length - DETECTOR_M)
                lines.append(f'    <laneAreaDetector id="{lane.getID()}" lane="{lane.getID()}" pos="{start:.1f}" '
                             f'endPos="{length - 0.1:.1f}" friendlyPos="true" file="/dev/null" period="1000"/>')
                arm.append(lane.getID())
        names.append(arm)
    with open(os.path.join(folder, 'det.add.xml'), 'w') as f:
        f.write('<additional>\n    <!-- One lane-area detector per approach lane, ending at the stop line. -->\n'
                + '\n'.join(lines) + '\n</additional>\n')
    return names


def write_sumocfg(folder, name):
    with open(os.path.join(folder, f'{name}.sumocfg'), 'w') as f:
        f.write(f'<configuration>\n    <input>\n        <net-file value="{name}.net.xml"/>\n'
                f'        <route-files value="{name}.rou.xml"/>\n'
                '        <additional-files value="det.add.xml"/>\n    </input>\n'
                '    <report>\n        <no-step-log value="true"/>\n'
                '        <duration-log.disable value="true"/>\n    </report>\n</configuration>\n')


# ------------------------------------------------------------------- four_way

FOUR_WAY_ARMS = ['N', 'E', 'S', 'W']
ARM_XY = {'N': (0, 200), 'E': (200, 0), 'S': (0, -200), 'W': (-200, 0)}
OPPOSITE = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}
# Turning from each arm in left-hand traffic. Coming from N you drive south:
# your left is E and your right (across oncoming traffic) is W.
LEFT = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
RIGHT = {'N': 'W', 'E': 'N', 'S': 'E', 'W': 'S'}
TURN_SPLIT = {'straight': 0.6, 'left': 0.2, 'right': 0.2}
# Arrivals per arm (veh/h) in three 400 s periods: N-S heavy, balanced, E-W heavy.
FOUR_WAY_DEMAND = [
    (0, 400, {'N': 450, 'S': 350, 'E': 150, 'W': 150}),
    (400, 800, {'N': 275, 'S': 275, 'E': 275, 'W': 275}),
    (800, 1200, {'N': 150, 'S': 150, 'E': 450, 'W': 350}),
]


def build_four_way():
    folder = os.path.join(HERE, 'four_way')
    os.makedirs(folder, exist_ok=True)
    name = 'four_way'
    with open(os.path.join(folder, f'{name}.nod.xml'), 'w') as f:
        f.write('<nodes>\n    <node id="C" x="0" y="0" type="traffic_light"/>\n'
                + ''.join(f'    <node id="{a}" x="{x}" y="{y}" type="priority"/>\n' for a, (x, y) in ARM_XY.items())
                + '</nodes>\n')
    with open(os.path.join(folder, f'{name}.edg.xml'), 'w') as f:
        f.write('<edges>\n    <!-- Every road is two-way with one lane in each direction, 50 km/h -->\n'
                + ''.join(f'    <edge id="{a}2C" from="{a}" to="C" numLanes="1" speed="13.89"/>\n'
                          f'    <edge id="C2{a}" from="C" to="{a}" numLanes="1" speed="13.89"/>\n'
                          for a in FOUR_WAY_ARMS)
                + '</edges>\n')
    base = ['--node-files', os.path.join(folder, f'{name}.nod.xml'),
            '--edge-files', os.path.join(folder, f'{name}.edg.xml')]
    out = os.path.join(folder, f'{name}.net.xml')
    arm_edges = [[f'{a}2C'] for a in FOUR_WAY_ARMS]

    netconvert(*base, '-o', out)                                    # pass 1: get the link order
    write_programs(sumolib.net.readNet(out), folder, name, 'C', arm_edges)
    netconvert(*base, '--tllogic-files', os.path.join(folder, f'{name}.tll.xml'), '-o', out)   # pass 2
    net = sumolib.net.readNet(out)
    detectors = write_detectors(net, folder, arm_edges)
    write_sumocfg(folder, name)
    write_four_way_routes(folder, name)
    print(f'built {out}\n  detectors per arm: {detectors}')


def write_four_way_routes(folder, name):
    lines = ['<routes>',
             '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" '
             'maxSpeed="13.89" guiShape="passenger"/>',
             '    <!-- Random arrivals. Turning split 60% straight, 20% left, 20% right.',
             '         0-400 s N-S heavy, 400-800 s balanced, 800-1200 s E-W heavy. -->']
    for begin, end, per_arm in FOUR_WAY_DEMAND:
        for a in FOUR_WAY_ARMS:
            for turn, to in (('straight', OPPOSITE[a]), ('left', LEFT[a]), ('right', RIGHT[a])):
                p = per_arm[a] * TURN_SPLIT[turn] / 3600
                lines.append(f'    <flow id="{a}_{turn}_{begin}" type="car" begin="{begin}" end="{end}" '
                             f'probability="{p:.4f}" from="{a}2C" to="C2{to}" departLane="best"/>')
    lines.append('</routes>')
    with open(os.path.join(folder, f'{name}.rou.xml'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    targets = sys.argv[1:] or ['four_way']
    if 'four_way' in targets:
        build_four_way()
