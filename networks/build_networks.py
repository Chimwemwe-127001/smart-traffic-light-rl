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
import re
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


def strip_header(path):
    """netconvert writes the build time and absolute local paths in a header comment.
    Drop it, so the file only changes when the network does."""
    with open(path, encoding='utf-8') as f:
        text = f.read()
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(re.sub(r'<!-- generated on .*?-->\s*', '', text, count=1, flags=re.S))


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


def write_programs(folder, name, tls_id, phases, greens_s, note):
    """phases: green, yellow, green, yellow, ... state strings. greens_s: fixed-time green per green phase."""
    durations = iter(greens_s)
    fixed = [f'        <phase duration="{next(durations) if "G" in s else YELLOW_S}" state="{s}"/>' for s in phases]
    with open(os.path.join(folder, f'{name}.tll.xml'), 'w') as f:
        f.write(f'<tlLogics>\n    <!-- {note} -->\n'
                f'    <tlLogic id="{tls_id}" type="static" programID="0" offset="0">\n'
                + '\n'.join(fixed) + '\n    </tlLogic>\n</tlLogics>\n')
    durations = iter(greens_s)
    act = [f'        <phase duration="{next(durations)}" minDur="{MIN_GREEN_S}" maxDur="{MAX_GREEN_S}" state="{s}"/>'
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
    phases = split_phases(sumolib.net.readNet(out), 'C', arm_edges)
    write_programs(folder, name, 'C', phases, [FIXED_GREEN_S] * len(arm_edges),
                   'Split phasing, fixed time: 30 s green + 3 s yellow per arm')
    netconvert(*base, '--tllogic-files', os.path.join(folder, f'{name}.tll.xml'), '-o', out)   # pass 2
    strip_header(out)
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


# --------------------------------------------------------------------- lusaka
#
# Great East Road / Lufubu Road, next to East Park Mall, Lusaka. The signals here
# were switched off in 2021 because they caused peak-hour congestion, and the
# median was later closed. We rebuild the junction as the signalized crossroads
# it was, on its real geometry from OpenStreetMap (networks/lusaka/east_park.osm.xml,
# (c) OpenStreetMap contributors, ODbL).

LUSAKA_OSM = os.path.join(HERE, 'lusaka', 'east_park.osm.xml')
# Where Lufubu Road meets the eastbound carriageway, and the two nodes of the
# westbound carriageway facing it. The crossroads centre is their mean.
LUSAKA_CENTRE_NODES = ['1163797210', '9299695473', '7116045845']
# arm: (description, OSM way + node index that give the arm's direction,
#       arm length in m (None = real length), OSM way for traffic coming in, OSM way for traffic going out)
LUSAKA_ARMS = {
    'W': ('Great East Road from the west (eastbound)', ('402266062', 0), 300, '402266062', '493457962'),
    'E': ('Great East Road from the east (westbound)', ('402267891', 3), 300, '1084679499', '402267891'),
    'N': ('Lufubu Road', ('674442399', 1), 300, '674442399', '674442399'),
    'S': ('East Park Mall access', ('744084367', -1), None, '761630966', '744084367'),
}
MAIN_ARMS, SIDE_ARMS = ('W', 'E'), ('N', 'S')
SIDE_ROAD_SPEED_KMH = 40            # assumption: no maxspeed tag on the side roads
# Fixed-time plan: main road 40 s, protected right turns 10 s, side roads 20 s (3 s yellow after each).
LUSAKA_FIXED_GREENS = [40, 10, 20]

# Morning peak hour. Great East Road carries about 31,000 veh/day (UNZA study).
# Peak hour = K x AADT with K = 0.09, split D = 0.6 towards the city (westbound),
# standard planning factors (Highway Capacity Manual). Side-road volumes and all
# turning shares are assumptions, listed in the README.
AADT, K_FACTOR, D_FACTOR = 31_000, 0.09, 0.6
LUSAKA_ARRIVALS = {'E': AADT * K_FACTOR * D_FACTOR, 'W': AADT * K_FACTOR * (1 - D_FACTOR), 'N': 300, 'S': 200}
LUSAKA_TURNS = {       # share of each arm's traffic by movement (left-hand traffic)
    'W': {'straight': 0.85, 'left': 0.08, 'right': 0.07},
    'E': {'straight': 0.85, 'left': 0.08, 'right': 0.07},
    'N': {'straight': 0.10, 'left': 0.30, 'right': 0.60},
    'S': {'straight': 0.10, 'left': 0.50, 'right': 0.40},
}
DEMAND_SWEEP = [0.75, 1.0, 1.25]


def read_osm(path):
    root = ET.parse(path).getroot()
    nodes = {n.get('id'): (float(n.get('lat')), float(n.get('lon'))) for n in root.findall('node')}
    ways = {w.get('id'): ({t.get('k'): t.get('v') for t in w.findall('tag')}, [nd.get('ref') for nd in w.findall('nd')])
            for w in root.findall('way')}
    return nodes, ways


def lanes_per_direction(tags):
    lanes = int(tags.get('lanes', 1))
    return lanes if tags.get('oneway') == 'yes' else max(1, lanes // 2)


def road_speeds(ways):
    """Speed limit per road name, from any OSM segment of that road that is tagged."""
    return {t['name']: float(t['maxspeed']) for t, _ in ways.values() if 'name' in t and 'maxspeed' in t}


def speed_ms(tags, speeds):
    kmh = tags.get('maxspeed') or speeds.get(tags.get('name')) or SIDE_ROAD_SPEED_KMH
    return float(kmh) / 3.6


def build_lusaka():
    import math
    folder = os.path.join(HERE, 'lusaka')
    name = 'lusaka'
    nodes, ways = read_osm(LUSAKA_OSM)
    speeds = road_speeds(ways)
    lat0 = sum(nodes[n][0] for n in LUSAKA_CENTRE_NODES) / len(LUSAKA_CENTRE_NODES)
    lon0 = sum(nodes[n][1] for n in LUSAKA_CENTRE_NODES) / len(LUSAKA_CENTRE_NODES)

    def xy(node):                      # metres east / north of the junction centre
        lat, lon = nodes[node]
        return ((lon - lon0) * 111_320 * math.cos(math.radians(lat0)), (lat - lat0) * 110_574)

    node_lines, edge_lines = ['    <node id="C" x="0" y="0" type="traffic_light"/>'], []
    for a, (desc, (way, idx), length, in_way, out_way) in LUSAKA_ARMS.items():
        x, y = xy(ways[way][1][idx])
        d = math.hypot(x, y)
        L = length or d
        node_lines.append(f'    <node id="{a}" x="{x / d * L:.1f}" y="{y / d * L:.1f}" type="priority"/>  <!-- {desc} -->')
        for eid, frm, to, w in ((f'{a}2C', a, 'C', in_way), (f'C2{a}', 'C', a, out_way)):
            tags = ways[w][0]
            edge_lines.append(f'    <edge id="{eid}" from="{frm}" to="{to}" numLanes="{lanes_per_direction(tags)}" '
                              f'speed="{speed_ms(tags, speeds):.2f}"/>')
    with open(os.path.join(folder, f'{name}.nod.xml'), 'w') as f:
        f.write('<nodes>\n' + '\n'.join(node_lines) + '\n</nodes>\n')
    with open(os.path.join(folder, f'{name}.edg.xml'), 'w') as f:
        f.write('<edges>\n    <!-- Lanes and speeds from the OSM tags of each way -->\n'
                + '\n'.join(edge_lines) + '\n</edges>\n')

    base = ['--node-files', os.path.join(folder, f'{name}.nod.xml'),
            '--edge-files', os.path.join(folder, f'{name}.edg.xml')]
    out = os.path.join(folder, f'{name}.net.xml')
    netconvert(*base, '-o', out)                                    # pass 1: get the link order
    phases = lusaka_phases(out)
    write_programs(folder, name, 'C', phases, LUSAKA_FIXED_GREENS,
                   'Main road 40 s, protected right turns 10 s, side roads 20 s, 3 s yellow after each')
    netconvert(*base, '--tllogic-files', os.path.join(folder, f'{name}.tll.xml'), '-o', out)   # pass 2
    strip_header(out)
    detectors = write_detectors(sumolib.net.readNet(out), folder, [[f'{a}2C'] for a in LUSAKA_ARMS])
    write_sumocfg(folder, name)
    for scale in DEMAND_SWEEP:
        suffix = '' if scale == 1.0 else f'_x{scale:.2f}'
        write_lusaka_routes(os.path.join(folder, f'{name}{suffix}.rou.xml'), scale)
    print(f'built {out}\n  detectors per arm: {detectors}\n  phases: {phases[::2]}')


def lusaka_phases(net_file):
    """Three greens: main road (right turns yield), protected main-road right turns, side roads (right turns yield)."""
    conns = [c for c in ET.parse(net_file).getroot().findall('connection') if c.get('tl') == 'C']
    n_links = max(int(c.get('linkIndex')) for c in conns) + 1
    move = {int(c.get('linkIndex')): (c.get('from')[0], c.get('dir').lower()) for c in conns}

    def state(rule):
        return ''.join(rule(*move[i]) for i in range(n_links))
    main = state(lambda arm, d: ('g' if d == 'r' else 'G') if arm in MAIN_ARMS else 'r')
    rights = state(lambda arm, d: 'G' if arm in MAIN_ARMS and d == 'r' else 'r')
    side = state(lambda arm, d: ('g' if d == 'r' else 'G') if arm in SIDE_ARMS else 'r')
    yellow = lambda s: ''.join('y' if ch in 'Gg' else 'r' for ch in s)
    return [main, yellow(main), rights, yellow(rights), side, yellow(side)]


def lusaka_destination(arm, turn):
    return {'straight': OPPOSITE[arm], 'left': LEFT[arm], 'right': RIGHT[arm]}[turn]


def write_lusaka_routes(path, scale):
    lines = ['<routes>',
             '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" '
             'maxSpeed="22.22" guiShape="passenger"/>',
             f'    <!-- Morning peak, random arrivals, demand x{scale:.2f}. See networks/build_networks.py',
             '         for how the volumes were derived. -->']
    for a, per_hour in LUSAKA_ARRIVALS.items():
        for turn, share in LUSAKA_TURNS[a].items():
            p = per_hour * scale * share / 3600
            lines.append(f'    <flow id="{a}_{turn}" type="car" begin="0" end="3600" probability="{p:.4f}" '
                         f'from="{a}2C" to="C2{lusaka_destination(a, turn)}" departLane="best"/>')
    lines.append('</routes>')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    targets = sys.argv[1:] or ['four_way', 'lusaka']
    if 'four_way' in targets:
        build_four_way()
    if 'lusaka' in targets:
        build_lusaka()
