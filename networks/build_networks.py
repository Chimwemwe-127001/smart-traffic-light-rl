"""Build the four-way and Manda Hill networks reproducibly with SUMO's netconvert.

    python networks/build_networks.py four_way
    python networks/build_networks.py manda_hill

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


def write_detectors(net, folder, arm_edges, reach=None):
    """One detector per approach lane: the last 105 m (or the whole lane if shorter), ending at the stop line.
    reach: optional {edge: metres} for arms that need to see further back."""
    lines, names = [], []
    for edges in arm_edges:
        arm = []
        for e in edges:
            for lane in net.getEdge(e).getLanes():
                length = lane.getLength()
                start = max(0.1, length - (reach or {}).get(e, DETECTOR_M))
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
# Arrivals per arm (veh/h): a 300 s warm-up (not measured) at the first period's
# rates, then three 400 s periods: N-S heavy, balanced, E-W heavy.
FOUR_WAY_DEMAND = [
    (0, 300, {'N': 450, 'S': 350, 'E': 150, 'W': 150}),
    (300, 700, {'N': 450, 'S': 350, 'E': 150, 'W': 150}),
    (700, 1100, {'N': 275, 'S': 275, 'E': 275, 'W': 275}),
    (1100, 1500, {'N': 150, 'S': 150, 'E': 450, 'W': 350}),
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
             '         0-300 s warm-up at the first-period rates, then 300-700 s N-S heavy,',
             '         700-1100 s balanced, 1100-1500 s E-W heavy. -->']
    for begin, end, per_arm in FOUR_WAY_DEMAND:
        for a in FOUR_WAY_ARMS:
            for turn, to in (('straight', OPPOSITE[a]), ('left', LEFT[a]), ('right', RIGHT[a])):
                p = per_arm[a] * TURN_SPLIT[turn] / 3600
                lines.append(f'    <flow id="{a}_{turn}_{begin}" type="car" begin="{begin}" end="{end}" '
                             f'probability="{p:.4f}" from="{a}2C" to="C2{to}" departLane="best"/>')
    lines.append('</routes>')
    with open(os.path.join(folder, f'{name}.rou.xml'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ----------------------------------------------------------------- manda_hill
#
# Great East Road / Manchinchi Road / Addis Ababa Drive, next to Manda Hill Mall,
# Lusaka: a working signalized junction on the city's main arterial. Every number
# here is read from OpenStreetMap (networks/manda_hill/manda_hill.osm.xml,
# (c) OpenStreetMap contributors, ODbL): arm bearings, lane counts, turn lanes and
# speed limits. The geometry is drawn clean: every arm is a straight dual
# carriageway with a median (Great East Road is one straight line through the
# junction), and the free left turns use curved slip roads that give way where
# they merge, as on the real road.
#
# Per arm: '<arm>_up' is the normal road, '<arm>2C' the stop-line section with its
# turn lanes, '<arm>_slip' the left slip road (if any), 'C2<arm>' (+ '<arm>_out'
# after a slip merges in) the exit.

MANDA_OSM = os.path.join(HERE, 'manda_hill', 'manda_hill.osm.xml')
MANDA_SIGNAL_NODES = ['314441861', '675703414', '675718067', '675718074']    # the junction centre is their mean
# arm: description, OSM way giving the arm's bearing, arm length (m), way for the normal
# approach, way for the stop-line section (its turn:lanes), left slip way or None, way for the exit
MANDA_ARMS = {
    'W': ('Great East Road from the city (eastbound)', '494998690', 300, '495014393', '495014379', None, '495014381'),
    'E': ('Great East Road from Manda Hill Mall (westbound)', '495015152', 240, '617583865', '495014380', '495014375',
          '617583852'),
    'N': ('Manchinchi Road', '28802708', 300, '28802708', '437003429', '1293813900', '437003428'),
    'S': ('Addis Ababa Drive', '693574590', 300, '693574590', '1271757317', '1293813899', '53456893'),
}
MAIN_ARMS, SIDE_ARMS = ('W', 'E'), ('N', 'S')
# Where each turn goes, in left-hand traffic (left = the kerb side, right = across traffic)
MANDA_TURNS_TO = {'W': {'left': 'N', 'through': 'E', 'right': 'S'},
                  'E': {'left': 'S', 'through': 'W', 'right': 'N'},
                  'N': {'left': 'E', 'through': 'S', 'right': 'W'},
                  'S': {'left': 'W', 'through': 'N', 'right': 'E'}}
HALF_MEDIAN_M = {'W': 4.0, 'E': 4.0, 'N': 2.5, 'S': 2.5}     # half the width of the central median
LANE_W = 3.2
MERGE_M = 35                        # where a slip road joins the exit, metres from the junction
SLIP_GAP_M = 1.0                    # space between a slip road and the carriageway beside it
BEARING_M = 120                     # arm directions are measured this far from the junction
SIDE_ROAD_SPEED_KMH = 50            # assumption: no maxspeed tag on Manchinchi Road
MAIN_ROAD_DETECTOR_M = 105          # checked by the SEMMA coverage study
MANDA_FIXED_GREENS = [40, 12, 18, 20]   # assumed plan: main road, main-road rights, Manchinchi, Addis Ababa

# Morning peak hour. Great East Road carried about 31,000 veh/day in 2009, as quoted
# by Choongo (2020, UNZA); the counting place and method are not stated. Peak hour =
# K x daily volume, split D towards the city (westbound). K = 0.09 and D = 0.6 are
# assumed planning values, not measured in Lusaka. The side-road volumes and all
# turning shares are assumptions, guided by how many lanes each movement has.
DAILY_VEH, K_FACTOR, D_FACTOR = 31_000, 0.09, 0.6
MANDA_ARRIVALS = {'E': DAILY_VEH * K_FACTOR * D_FACTOR, 'W': DAILY_VEH * K_FACTOR * (1 - D_FACTOR),
                  'N': 600, 'S': 700}
MANDA_TURNS = {       # share of each arm's traffic by movement
    'W': {'left': 0.10, 'through': 0.75, 'right': 0.15},
    'E': {'left': 0.15, 'through': 0.70, 'right': 0.15},
    'N': {'left': 0.30, 'through': 0.70},                 # no right-turn lane is mapped
    'S': {'left': 0.25, 'through': 0.40, 'right': 0.35},
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


def stop_line_turns(turn_lanes, has_slip):
    """Movements per stop-line lane, kerb lane first (lane 0 in a left-hand network):
    OSM lists turn lanes left to right in the driving direction, the same order.
    A kerb lane that only turns left, where a slip road exists, is the slip road itself."""
    lanes = [[{'slight_left': 'left', 'slight_right': 'right'}.get(m, m) for m in lane.split(';')]
             for lane in turn_lanes.split('|')]
    if has_slip and lanes[0] == ['left']:
        lanes = lanes[1:]
    return lanes


def build_manda_hill():
    import json
    import math
    folder = os.path.join(HERE, 'manda_hill')
    name = 'manda_hill'
    nodes, ways = read_osm(MANDA_OSM)
    speeds = road_speeds(ways)
    lat0 = sum(nodes[n][0] for n in MANDA_SIGNAL_NODES) / len(MANDA_SIGNAL_NODES)
    lon0 = sum(nodes[n][1] for n in MANDA_SIGNAL_NODES) / len(MANDA_SIGNAL_NODES)

    def xy(node):                      # metres east / north of the junction centre
        lat, lon = nodes[node]
        return ((lon - lon0) * 111_320 * math.cos(math.radians(lat0)), (lat - lat0) * 110_574)

    def way_length(way):
        pts = [xy(n) for n in ways[way][1]]
        return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))

    # bearings from the map near the junction (as in the OSM standard layer, not where the roads
    # bend further out); each road is drawn as one straight line through the junction
    unit = {}
    for a, arm in MANDA_ARMS.items():
        near = [xy(n) for w in (arm[3], arm[4], arm[6]) for n in ways[w][1]]
        x, y = min((p for p in near if math.hypot(*p) > 20), key=lambda p: abs(math.hypot(*p) - BEARING_M))
        unit[a] = (x / math.hypot(x, y), y / math.hypot(x, y))
    for a, b in (('E', 'W'), ('S', 'N')):
        gx, gy = unit[a][0] - unit[b][0], unit[a][1] - unit[b][1]
        unit[a] = (gx / math.hypot(gx, gy), gy / math.hypot(gx, gy))
        unit[b] = (-unit[a][0], -unit[a][1])

    def pt(a, t, side):
        """Point t metres out along arm a, `side` metres to the right of the outward direction."""
        ux, uy = unit[a]
        return (ux * t + uy * side, uy * t - ux * side)

    def fmt(points):
        return ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)

    arm = {}
    for a, (desc, _, length, up_way, bay_way, slip_way, out_way) in MANDA_ARMS.items():
        up, bay, out = ways[up_way][0], ways[bay_way][0], ways[out_way][0]
        turns = stop_line_turns(bay.get('turn:lanes', '|'.join(['through'] * lanes_per_direction(bay))),
                                slip_way is not None)
        arm[a] = dict(desc=desc, length=length, bay_len=min(max(way_length(bay_way), 40.0), 100.0),
                      n_up=lanes_per_direction(up), n_bay=len(turns), n_out=lanes_per_direction(out),
                      v_up=speed_ms(up, speeds), v_bay=speed_ms(bay, speeds), v_out=speed_ms(out, speeds),
                      turns=turns, slip=slip_way is not None, h=HALF_MEDIAN_M[a])
    gets_slip = {MANDA_TURNS_TO[a]['left']: a for a in arm if arm[a]['slip']}
    # every road stops at the edge of the junction box: where the crossing road's outer lanes begin
    half_width = {a: arm[a]['h'] + max(arm[a]['n_bay'], arm[a]['n_out']) * LANE_W for a in arm}
    edge_m = {a: max(half_width[b] for b in (SIDE_ARMS if a in MAIN_ARMS else MAIN_ARMS)) + 3 for a in arm}

    node_lines = ['    <node id="C" x="0" y="0" type="traffic_light"/>']
    edge_lines, con = [], []
    for a, m in arm.items():
        h = m['h']
        # inbound carriageway: to the right of the outward direction; the edge line is its median side
        far, split = pt(a, m['length'], h), pt(a, m['bay_len'], h)
        node_lines += [f'    <node id="{a}" x="{far[0]:.2f}" y="{far[1]:.2f}" type="priority"/>  <!-- {m["desc"]} -->',
                       f'    <node id="{a}b" x="{split[0]:.2f}" y="{split[1]:.2f}" type="priority"/>']
        edge_lines += [
            f'    <edge id="{a}_up" from="{a}" to="{a}b" numLanes="{m["n_up"]}" speed="{m["v_up"]:.2f}" priority="3"/>',
            f'    <edge id="{a}2C" from="{a}b" to="C" numLanes="{m["n_bay"]}" speed="{m["v_bay"]:.2f}" priority="3" '
            f'shape="{fmt([split, pt(a, edge_m[a], h)])}"/>']
        # outbound carriageway: to the left; a slip road merges in MERGE_M from the junction
        out_far = pt(a, m['length'], -h)
        node_lines.append(f'    <node id="{a}o" x="{out_far[0]:.2f}" y="{out_far[1]:.2f}" type="priority"/>')
        if a in gets_slip:
            mrg = pt(a, MERGE_M, -h)
            node_lines.append(f'    <node id="{a}m" x="{mrg[0]:.2f}" y="{mrg[1]:.2f}" type="priority"/>')
            edge_lines += [
                f'    <edge id="C2{a}" from="C" to="{a}m" numLanes="{m["n_out"]}" speed="{m["v_out"]:.2f}" priority="3" '
                f'shape="{fmt([pt(a, edge_m[a], -h), mrg])}"/>',
                f'    <edge id="{a}_out" from="{a}m" to="{a}o" numLanes="{m["n_out"]}" speed="{m["v_out"]:.2f}" '
                f'priority="3"/>']
            con += [f'    <connection from="C2{a}" to="{a}_out" fromLane="{i}" toLane="{i}"/>' for i in range(m['n_out'])]
        else:
            edge_lines.append(f'    <edge id="C2{a}" from="C" to="{a}o" numLanes="{m["n_out"]}" speed="{m["v_out"]:.2f}" '
                              f'priority="3" shape="{fmt([pt(a, edge_m[a], -h), out_far])}"/>')
        # the widening before the stop line: every stop-line lane is fed from the nearest normal lane
        for j in range(m['n_bay']):
            con.append(f'    <connection from="{a}_up" to="{a}2C" fromLane="{feeding_lane(j, m["n_up"], m["n_bay"])}" '
                       f'toLane="{j}"/>')

    # slip roads: their own lane, running just outside the approach kerb, a tight curve round
    # the corner of the junction, then just outside the exit kerb until they merge
    for a, m in arm.items():
        if not m['slip']:
            continue
        b = MANDA_TURNS_TO[a]['left']
        k1 = m['h'] + m['n_bay'] * LANE_W + SLIP_GAP_M
        k2 = arm[b]['h'] + arm[b]['n_out'] * LANE_W + SLIP_GAP_M
        p0, p2 = pt(a, edge_m[a], k1), pt(b, edge_m[b], -k2)
        # control point: where the two kerb lines cross
        (ax, ay), (bx, by) = unit[a], unit[b]
        c0, c1 = pt(a, 0, k1), pt(b, 0, -k2)
        det = ax * (-by) - ay * (-bx)
        t = ((c1[0] - c0[0]) * (-by) - (c1[1] - c0[1]) * (-bx)) / det
        p1 = (c0[0] + ax * t, c0[1] + ay * t)
        corner = [((1 - s) ** 2 * p0[0] + 2 * (1 - s) * s * p1[0] + s ** 2 * p2[0],
                   (1 - s) ** 2 * p0[1] + 2 * (1 - s) * s * p1[1] + s ** 2 * p2[1]) for s in [i / 10 for i in range(11)]]
        curve = [pt(a, m['bay_len'], k1)] + corner + [pt(b, MERGE_M, -k2)]
        edge_lines.append(f'    <edge id="{a}_slip" from="{a}b" to="{b}m" numLanes="1" speed="{min(m["v_bay"], 30 / 3.6):.2f}" '
                          f'priority="1" shape="{fmt(curve)}"/>')
        con += [f'    <connection from="{a}_up" to="{a}_slip" fromLane="0" toLane="0"/>',
                f'    <connection from="{a}_slip" to="{b}_out" fromLane="0" toLane="0"/>']

    # turns at the signal, lane by lane, from the OSM turn lanes
    for a, m in arm.items():
        through = [k for k, mv in enumerate(m['turns']) if 'through' in mv]
        for lane, moves in enumerate(m['turns']):
            for mv in moves:
                if mv not in MANDA_TURNS_TO[a]:
                    continue
                b = MANDA_TURNS_TO[a][mv]
                n_out = arm[b]['n_out']
                if mv == 'left':
                    to_lane = 0
                elif mv == 'right':
                    to_lane = n_out - 1
                else:
                    k = through.index(lane)
                    to_lane = min(n_out - 1, round(k * (n_out - 1) / max(1, len(through) - 1)))
                con.append(f'    <connection from="{a}2C" to="C2{b}" fromLane="{lane}" toLane="{to_lane}"/>')

    with open(os.path.join(folder, f'{name}.nod.xml'), 'w') as f:
        f.write('<nodes>\n' + '\n'.join(node_lines) + '\n</nodes>\n')
    with open(os.path.join(folder, f'{name}.edg.xml'), 'w') as f:
        f.write('<edges>\n    <!-- Lanes and speeds from the OSM tags; straight arms, dual carriageways -->\n'
                + '\n'.join(edge_lines) + '\n</edges>\n')
    with open(os.path.join(folder, f'{name}.con.xml'), 'w') as f:
        f.write('<connections>\n    <!-- Lane-by-lane turns from the OSM turn:lanes tags; slip roads give way -->\n'
                + '\n'.join(con) + '\n</connections>\n')

    base = ['--node-files', os.path.join(folder, f'{name}.nod.xml'),
            '--edge-files', os.path.join(folder, f'{name}.edg.xml'),
            '--connection-files', os.path.join(folder, f'{name}.con.xml')]
    out = os.path.join(folder, f'{name}.net.xml')
    netconvert(*base, '-o', out)                                    # pass 1: get the link order
    phases = main_side_phases(out, 'C', {f'{a}2C': a for a in arm})
    write_programs(folder, name, 'C', phases, MANDA_FIXED_GREENS,
                   'Assumed plan: main road 40 s, main-road rights 12 s, Manchinchi 18 s, Addis Ababa 20 s, 3 s yellow after each')
    netconvert(*base, '--tllogic-files', os.path.join(folder, f'{name}.tll.xml'), '-o', out)   # pass 2
    strip_header(out)

    net = sumolib.net.readNet(out)
    approach = [[f'{a}_up', f'{a}2C'] for a in arm]
    detectors = write_chain_detectors(net, folder, [(a, arm[a]['n_up'], arm[a]['n_bay']) for a in arm],
                                      MAIN_ROAD_DETECTOR_M)
    with open(os.path.join(folder, 'arms.json'), 'w') as f:
        json.dump({a: {'description': arm[a]['desc'], 'approach': approach[i],
                       'exit': [f'C2{a}'] + ([f'{a}_out'] if a in gets_slip else []), 'detectors': detectors[i]}
                   for i, a in enumerate(arm)}, f, indent=2)
    write_sumocfg(folder, name)
    for scale in DEMAND_SWEEP:
        suffix = '' if scale == 1.0 else f'_x{scale:.2f}'
        write_manda_routes(os.path.join(folder, f'{name}{suffix}.rou.xml'), scale, gets_slip)
    print(f'built {out}\n  phases: {phases[::2]}')


def feeding_lane(j, n_up, n_bay):
    """The normal-road lane that feeds stop-line lane j where the road widens."""
    return round(j * (n_up - 1) / (n_bay - 1)) if n_bay > 1 else 0


def write_chain_detectors(net, folder, arms, reach):
    """One continuous detector per stop-line lane, from the stop line back `reach` metres.
    Where the road widens before the stop line, the detector carries on through the short
    connector onto the normal lane that feeds it (SUMO multi-lane detector), so there is no
    blind spot. Each normal lane joins only one detector, so no car is counted twice.
    Slip roads get none: their cars never wait at the signal."""
    lines, names = [], []
    for a, n_up, n_bay in arms:
        up, bay = net.getEdge(f'{a}_up'), net.getEdge(f'{a}2C')
        taken, arm = set(), []
        for j, lane in enumerate(bay.getLanes()):
            i = feeding_lane(j, n_up, n_bay)
            det = lane.getID()
            if i in taken or lane.getLength() >= reach:
                start = max(0.1, lane.getLength() - reach)
                lines.append(f'    <laneAreaDetector id="{det}" lane="{det}" pos="{start:.1f}" '
                             f'endPos="{lane.getLength() - 0.1:.1f}" friendlyPos="true" file="/dev/null" period="1000"/>')
            else:
                taken.add(i)
                up_lane = up.getLanes()[i]
                start = max(0.1, up_lane.getLength() - (reach - lane.getLength()))
                lines.append(f'    <laneAreaDetector id="{det}" lanes="{up_lane.getID()} {det}" pos="{start:.1f}" '
                             f'endPos="{lane.getLength() - 0.1:.1f}" friendlyPos="true" file="/dev/null" period="1000"/>')
            arm.append(det)
        names.append(arm)
    with open(os.path.join(folder, 'det.add.xml'), 'w') as f:
        f.write('<additional>\n    <!-- One continuous detector per stop-line lane, reaching the stop line -->\n'
                + '\n'.join(lines) + '\n</additional>\n')
    return names


def main_side_phases(net_file, tls_id, arm_of_edge):
    """Four greens, every turn protected so no car ever waits inside the junction:
    0 main road through and left, 1 main-road right turns, 2 Manchinchi Road,
    3 Addis Ababa Drive. Where two green links of one approach feed the same exit
    lane, only the first keeps priority and the other merges."""
    conns = [c for c in ET.parse(net_file).getroot().findall('connection') if c.get('tl') == tls_id]
    n_links = max(int(c.get('linkIndex')) for c in conns) + 1
    move = {int(c.get('linkIndex')): (arm_of_edge[c.get('from')], c.get('dir').lower()) for c in conns}
    target = {int(c.get('linkIndex')): (c.get('to'), c.get('toLane')) for c in conns}

    def state(rule):
        s, seen = [rule(*move[i]) for i in range(n_links)], set()
        for i, ch in enumerate(s):
            if ch == 'G':
                if target[i] in seen:
                    s[i] = 'g'
                seen.add(target[i])
        return ''.join(s)
    greens = [state(lambda arm, d: 'G' if arm in MAIN_ARMS and d != 'r' else 'r'),
              state(lambda arm, d: 'G' if arm in MAIN_ARMS and d == 'r' else 'r'),
              state(lambda arm, d: 'G' if arm == SIDE_ARMS[0] else 'r'),
              state(lambda arm, d: 'G' if arm == SIDE_ARMS[1] else 'r')]
    yellow = lambda s: ''.join('y' if ch in 'Gg' else 'r' for ch in s)
    return [p for g in greens for p in (g, yellow(g))]


def write_manda_routes(path, scale, gets_slip):
    lines = ['<routes>',
             '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" '
             'maxSpeed="22.22" guiShape="passenger"/>',
             f'    <!-- Morning peak, random arrivals, demand x{scale:.2f}. See networks/build_networks.py',
             '         for how the volumes were derived. Left turns take the slip roads where they exist. -->']
    for a, per_hour in MANDA_ARRIVALS.items():
        for turn, share in MANDA_TURNS[a].items():
            p = per_hour * scale * share / 3600
            b = MANDA_TURNS_TO[a][turn]
            dst = f'{b}_out' if b in gets_slip else f'C2{b}'
            lines.append(f'    <flow id="{a}_{turn}" type="car" begin="0" end="3600" probability="{p:.4f}" '
                         f'from="{a}_up" to="{dst}" departLane="best"/>')
    lines.append('</routes>')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    targets = sys.argv[1:] or ['four_way', 'manda_hill']
    if 'four_way' in targets:
        build_four_way()
    if 'manda_hill' in targets:
        build_manda_hill()
