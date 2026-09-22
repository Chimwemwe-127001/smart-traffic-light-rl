"""Time a fixed-time plan properly, with Webster's method, for every scenario.

The hand-set fixed-time plans (42/42 s, 30 s per arm, 40/10/20 s) were never
optimized, so beating them says little. This script builds the fixed-time plan
a traffic engineer would: Webster (1958), from measured saturation flows and
the scenario's average demand.

1. Measure, in SUMO, what each green can discharge. For every green phase, only
   that phase's movements get traffic, heavy enough to queue, and the signal
   cycles 60 s red, 20 s green, 3 s yellow. Stop-line crossing times of the
   queued vehicles give, per approach (HCM Chapter 31, field measurement of
   saturation flow):
     saturation headway h   mean headway from the 5th queued vehicle on; s = 3600 / h
     start-up lost time     extra time the first 4 vehicles take, compared with h
     clearance lost time    3 s yellow minus the vehicles still crossing in it, times h
2. Demand per movement from the route file, averaged over the measured window
   (after the warm-up). Each movement belongs to the phase that gives it
   priority green (G), otherwise to the one that lets it go after giving way (g).
3. Webster: y = q / (s x lanes) per approach, the largest per phase; Y = sum of y;
   L = sum of lost times; cycle C0 = (1.5 L + 5) / (1 - Y), at most 120 s;
   greens in proportion to y, kept within the same 10 to 60 s as every controller.

Each junction is timed on its own: the corridor plans have no green-wave offset.

Outputs:
    networks/<net>/webster.add.xml (corridor_heavy: webster_heavy.add.xml)
    results/webster.json            measured saturation flows, lost times, y, Y, L, cycles, greens

Usage:
    python experiments/webster.py
"""
import json
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic_rl.baselines import MAX_CYCLE_S, webster_cycle, webster_greens
from traffic_rl.env import MAX_GREEN_S, MIN_GREEN_S, WARMUP_S, YELLOW_S, sumolib, traci
from traffic_rl.scenarios import SCENARIOS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEASURE_S = 1200                    # the measured window after the warm-up
RED_S, GREEN_S, CYCLES = 60, 20, 30
QUEUE_RATE = 1800                   # veh/h per lane offered during the measurement: enough to queue
PLAN_FILE = {'corridor_heavy': 'webster_heavy.add.xml'}


def load(scenario):
    sc = SCENARIOS[scenario]
    cfg = ET.parse(sc['cfg']).getroot()
    net_file = os.path.join(os.path.dirname(sc['cfg']), cfg.find('.//net-file').get('value'))
    return sc, sumolib.net.readNet(net_file, withPrograms=True)


def green_phases(net, tls):
    """(index, state) of every green phase in the network's own program."""
    phases = net.getTLS(tls).getPrograms()['0'].getPhases()
    return [(i, p.state) for i, p in enumerate(phases) if 'G' in p.state and 'y' not in p.state]


def movement_links(net, tls):
    """(in edge, out edge) -> [(in lane, link index)] for every movement the signal controls."""
    links = defaultdict(list)
    for in_lane, out_lane, idx in net.getTLS(tls).getConnections():
        links[(in_lane.getEdge().getID(), out_lane.getEdge().getID())].append((in_lane.getID(), idx))
    return links


def phase_of(links, greens):
    """The green that shows this movement green on all its lanes with priority (G) on at least one,
    else the one where it may go after giving way (g) on all its lanes."""
    for k, (_, state) in enumerate(greens):
        marks = [state[idx] for _, idx in links]
        if all(m in 'Gg' for m in marks) and 'G' in marks:
            return k
    for k, (_, state) in enumerate(greens):
        if all(state[idx] == 'g' for _, idx in links):
            return k
    raise ValueError(f'movement never gets green: {links}')


def demand(scenario, net, tls, routes=None):
    """Average demand (veh/h) per movement at this junction over the measured window."""
    sc = SCENARIOS[scenario]
    moves = movement_links(net, tls)
    q = defaultdict(float)
    start, end = WARMUP_S, WARMUP_S + MEASURE_S
    for flow in ET.parse(routes or sc['routes']).getroot().findall('flow'):
        overlap = max(0.0, min(end, float(flow.get('end'))) - max(start, float(flow.get('begin'))))
        path, _ = net.getShortestPath(net.getEdge(flow.get('from')), net.getEdge(flow.get('to')))
        for a, b in zip(path, path[1:]):
            if (a.getID(), b.getID()) in moves:
                q[(a.getID(), b.getID())] += float(flow.get('probability')) * 3600 * overlap / MEASURE_S
    return dict(q)


# ------------------------------------------------------------ measurement

def measure_phase(scenario, net, tls, green_state, moves, shares):
    """Discharge of one green: saturation headway, start-up and clearance lost time per approach."""
    sc = SCENARIOS[scenario]
    vtype = ET.parse(sc['routes']).getroot().find('vType')
    lanes = defaultdict(set)
    for (a, b) in moves:
        for lane, _ in movement_links(net, tls)[(a, b)]:
            lanes[a].add(lane)
    tmp = tempfile.mkdtemp(prefix='webster_')
    route_file = os.path.join(tmp, 'saturation.rou.xml')
    lines = ['<routes>', '    ' + ET.tostring(vtype, encoding='unicode').strip()]
    for k, (a, b) in enumerate(moves):
        arm_total = sum(shares[m] for m in moves if m[0] == a)
        rate = QUEUE_RATE * len(lanes[a]) * shares[(a, b)] / arm_total
        lines.append(f'    <flow id="m{k}" type="{vtype.get("id")}" begin="0" end="{CYCLES * 100}" '
                     f'vehsPerHour="{rate:.1f}" from="{a}" to="{b}" departLane="best" departSpeed="max"/>')
    with open(route_file, 'w') as f:
        f.write('\n'.join(lines + ['</routes>']) + '\n')

    traci.start([sumolib.checkBinary('sumo'), '-c', sc['cfg'], '-r', route_file, '--seed', '0',
                 '--step-length', '1', '--time-to-teleport', '-1', '--no-warnings', '--no-step-log'])
    red = 'r' * len(green_state)
    yellow = ''.join('y' if ch in 'Gg' else 'r' for ch in green_state)
    edges = sorted(lanes)
    edge_of = {lane: a for a in edges for lane in lanes[a]}
    t, prev = 0, {}
    startup, later, in_yellow = defaultdict(list), defaultdict(list), defaultdict(list)

    def step():
        """Advance 1 s. Returns {vehicle: lane} for the vehicles that crossed the stop line."""
        nonlocal t, prev
        traci.simulationStep()
        t += 1
        now = {v: traci.vehicle.getLaneID(v) for e in edges for v in traci.edge.getLastStepVehicleIDs(e)}
        crossed = {v: prev[v] for v in prev if v not in now}
        prev = now
        return crossed

    traci.trafficlight.setRedYellowGreenState(tls, red)
    for _ in range(CYCLES):
        for _ in range(RED_S):
            step()
        # queued cars in the lanes this green serves (a car may still wait in another lane to change over)
        queued = {v for v, lane in prev.items() if lane in edge_of and traci.vehicle.getSpeed(v) < 0.1}
        traci.trafficlight.setRedYellowGreenState(tls, green_state)
        green_start, headways, last = t, defaultdict(list), {}
        for _ in range(GREEN_S):
            for v, lane in step().items():
                if v in queued and lane in edge_of:    # headway to the previous queued car, or to the green start
                    headways[lane].append(t - last.get(lane, green_start))
                    last[lane] = t
        for lane, h in headways.items():
            if len(h) > 4:
                startup[edge_of[lane]].append(sum(h[:4]))
                later[edge_of[lane]].extend(h[4:])
        still_queued = {lane for v, lane in prev.items() if v in queued and lane in edge_of}   # at the yellow
        traci.trafficlight.setRedYellowGreenState(tls, yellow)
        crossing = defaultdict(int)
        for _ in range(YELLOW_S):
            for v, lane in step().items():
                crossing[lane] += 1
        for lane in still_queued:
            in_yellow[edge_of[lane]].append(crossing[lane])
        traci.trafficlight.setRedYellowGreenState(tls, red)
    traci.close()
    shutil.rmtree(tmp, ignore_errors=True)

    out = {}
    for a in edges:
        if len(later[a]) < 20:
            raise RuntimeError(f'{scenario} {tls} {a}: only {len(later[a])} saturation headways measured')
        h_sat = float(np.mean(later[a]))
        startup_lost = float(np.mean(startup[a])) - 4 * h_sat
        effective_yellow = float(np.mean(in_yellow[a])) * h_sat if in_yellow[a] else 0.0
        out[a] = {'saturation_veh_h_lane': round(3600 / h_sat), 'saturation_headway_s': round(h_sat, 3),
                  'startup_lost_s': round(max(0.0, startup_lost), 2),
                  'clearance_lost_s': round(max(0.0, YELLOW_S - effective_yellow), 2),
                  'lanes': len(lanes[a]), 'headways': len(later[a])}
    return out


# ----------------------------------------------------------------- plan

def plan(scenario, net, tls, measured):
    """Webster timing for one junction from the measured discharge and the average demand."""
    greens = green_phases(net, tls)
    links = movement_links(net, tls)
    q = demand(scenario, net, tls)
    y, lost, phases = [], [], []
    for k, (idx, state) in enumerate(greens):
        moves = [m for m in links if phase_of(links[m], greens) == k and q.get(m, 0) > 0]
        arms = sorted({a for a, _ in moves})
        ratios = {a: sum(q.get(m, 0.0) for m in moves if m[0] == a)
                  / (measured[k][a]['saturation_veh_h_lane'] * measured[k][a]['lanes']) for a in arms}
        critical = max(ratios, key=ratios.get)
        lost_k = measured[k][critical]['startup_lost_s'] + measured[k][critical]['clearance_lost_s']
        y.append(ratios[critical])
        lost.append(lost_k)
        phases.append({'phase': idx, 'state': state, 'flow_ratio_y': {a: round(r, 4) for a, r in ratios.items()},
                       'critical': critical, 'lost_s': round(lost_k, 2)})
    Y, L = sum(y), sum(lost)
    c0 = webster_cycle(Y, L)
    cycle = min(c0, MAX_CYCLE_S)
    g = webster_greens(y, lost, cycle, YELLOW_S, MIN_GREEN_S, MAX_GREEN_S)
    for p, gi in zip(phases, g):
        p['green_s'] = gi
    return {'Y': round(Y, 4), 'L': round(L, 2), 'webster_cycle_s': round(c0, 1), 'cycle_used_s': round(cycle, 1),
            'cycle_run_s': sum(g) + YELLOW_S * len(g), 'phases': phases}


def write_plan(scenario, net, plans):
    sc = SCENARIOS[scenario]
    folder = os.path.dirname(sc['cfg'])
    lines = ['<additional>',
             "    <!-- Fixed-time plan timed with Webster's method from measured saturation flows",
             '         and the average demand. Generated by experiments/webster.py; see results/webster.json. -->']
    for tls, p in plans.items():
        greens = {ph['phase']: ph['green_s'] for ph in p['phases']}
        lines.append(f'    <tlLogic id="{tls}" type="static" programID="webster" offset="0">')
        for i, ph in enumerate(net.getTLS(tls).getPrograms()['0'].getPhases()):
            lines.append(f'        <phase duration="{greens.get(i, YELLOW_S)}" state="{ph.state}"/>')
        lines.append('    </tlLogic>')
    path = os.path.join(folder, PLAN_FILE.get(scenario, 'webster.add.xml'))
    with open(path, 'w', newline='\n') as f:
        f.write('\n'.join(lines + ['</additional>']) + '\n')
    return os.path.relpath(path, REPO).replace(os.sep, '/')


def main():
    out = {'method': "Webster (1958) cycle and splits from saturation flows measured in SUMO "
                     "(headways from the 5th queued vehicle on) and demand averaged over the measured window",
           'measurement': {'red_s': RED_S, 'green_s': GREEN_S, 'yellow_s': YELLOW_S, 'cycles': CYCLES,
                           'offered_veh_h_lane': QUEUE_RATE},
           'saturation': {}, 'plans': {}}
    measured_nets = {}
    for scenario in SCENARIOS:
        sc, net = load(scenario)
        key = sc['cfg']                                  # corridor and corridor_heavy share a network
        if key not in measured_nets:
            measured_nets[key] = {}
            for tls in sc['intersections']:
                greens = green_phases(net, tls)
                links = movement_links(net, tls)
                q = demand(scenario, net, tls)
                measured_nets[key][tls] = []
                for k, (idx, state) in enumerate(greens):
                    moves = [m for m in links if phase_of(links[m], greens) == k and q.get(m, 0) > 0]
                    shares = {m: q[m] for m in moves}
                    measured_nets[key][tls].append(measure_phase(scenario, net, tls, state, moves, shares))
                    print(f'measured {scenario:15s} {tls:6s} green {k}: '
                          + ', '.join(f'{a} {v["saturation_veh_h_lane"]} veh/h/lane' for a, v in
                                      measured_nets[key][tls][-1].items()), flush=True)
            out['saturation'][scenario] = {tls: [{a: v for a, v in ph.items()} for ph in phs]
                                           for tls, phs in measured_nets[key].items()}
        plans = {tls: plan(scenario, net, tls, measured_nets[key][tls]) for tls in sc['intersections']}
        path = write_plan(scenario, net, plans)
        out['plans'][scenario] = {'file': path, **plans}
        for tls, p in plans.items():
            print(f'plan     {scenario:15s} {tls:6s} Y {p["Y"]:.3f}  L {p["L"]:.1f} s  C0 {p["webster_cycle_s"]:.1f} s  '
                  f'greens {[ph["green_s"] for ph in p["phases"]]}', flush=True)
    with open(os.path.join(REPO, 'results', 'webster.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote results/webster.json')


if __name__ == '__main__':
    main()
