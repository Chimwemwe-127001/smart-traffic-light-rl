"""SUMO environment for one or more signalized intersections.

Time runs in 1 s simulation steps. Every intersection is an agent that makes a
keep/switch decision on its current green:

    keep   -> hold the green for another DECISION_S seconds
    switch -> 3 s yellow, then the other approach gets at least MIN_GREEN_S of green

The agent is only asked when a switch is actually allowed (minimum green has
passed), so every action it takes has a real effect. This is the usual
decision model in traffic signal RL (e.g. Wei et al. 2018, Alegre 2019
sumo-rl) and fixes the problem we hit with 0.1 s per-step decisions, where
keep and switch were indistinguishable to the learner.

A green is never held longer than MAX_GREEN_S: at that point a keep is turned
into a switch, the same way a real controller caps green time. Without this
cap a trained tabular agent could get stuck keeping one green forever in a
rarely visited state (see the README, "What did not work").

Agents decide asynchronously: after a switch an agent is busy for 13 s, after a
keep only 5 s. step() applies the actions of the agents that were asked, then
runs the simulation until the next agent is ready.

Programs:
    'agent'    - RL agents own the signal, greens are held until switched
    'fixed'    - the default 42 s / 3 s / 42 s / 3 s cycle in the .net.xml
    'actuated' - SUMO's gap-based actuated control (the common real-world upgrade)
"""
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

if 'SUMO_HOME' not in os.environ:
    try:
        import sumo
        os.environ['SUMO_HOME'] = sumo.SUMO_HOME
    except ImportError:
        sys.exit("SUMO not found: pip install eclipse-sumo, or set SUMO_HOME")

import sumolib

# libsumo runs SUMO inside this Python process: several times faster than the
# TraCI socket and safe to run in parallel (no ports to collide). It cannot open
# sumo-gui, so set the environment variable LIBSUMO=0 to use TraCI when you
# want to watch a run with gui=True.
if os.environ.get('LIBSUMO', '1') == '1':
    try:
        import libsumo as traci
    except ImportError:
        import traci
else:
    import traci

from traffic_rl.scenarios import SCENARIOS

MIN_GREEN_S = 10
MAX_GREEN_S = 60            # same cap as the actuated baseline (maxDur)
YELLOW_S = 3
DECISION_S = 5
KEEP, SWITCH = 0, 1
DIRECTIONS = ('EB', 'SB')


class TrafficEnv:
    def __init__(self, scenario, program='agent', episode_s=1200, gui=False, record=False):
        self.name = scenario
        self.sc = SCENARIOS[scenario]
        self.ids = list(self.sc['intersections'])
        self.program = program
        self.episode_s = episode_s
        self.gui = gui
        self.record = record          # keep a per-second log of observations (used by SEMMA)
        self.records = []

    # ------------------------------------------------------------------ setup

    def reset(self, seed):
        """Start a fresh, seeded episode. Returns (ready agents, observations)."""
        self._tmp = tempfile.mkdtemp(prefix='sumo_')
        self._tripinfo = os.path.join(self._tmp, 'tripinfo.xml')
        additional = [os.path.join(os.path.dirname(self.sc['cfg']),
                                   ET.parse(self.sc['cfg']).find('.//additional-files').get('value'))]
        if self.program == 'actuated':
            additional.append(self.sc['actuated'])
        cmd = [sumolib.checkBinary('sumo-gui' if self.gui else 'sumo'),
               '-c', self.sc['cfg'], '-r', self.sc['routes'], '-a', ','.join(additional),
               '--seed', str(seed), '--step-length', '1', '--time-to-teleport', '-1',
               '--no-warnings', '--no-step-log',
               '--tripinfo-output', self._tripinfo, '--tripinfo-output.write-unfinished']
        if self.gui:
            cmd += ['--start', '--quit-on-end']
        traci.start(cmd)

        self.t = 0
        self.queue_trace = []
        self.switches = 0
        self._green_dir = {}
        self._next_decision = {}
        self._green_since = {}
        self._acc = {i: {j: 0.0 for j in self.ids} for i in self.ids}
        self._acc_n = {i: 0 for i in self.ids}
        for tls in self.ids:
            if self.program == 'actuated':
                traci.trafficlight.setProgram(tls, 'actuated')
            self._green_dir[tls] = self._map_green_phases(tls)
            self._next_decision[tls] = MIN_GREEN_S
            self._green_since[tls] = 0

        if self.program != 'agent':
            return [], self.observe()
        return self._run_until_ready()

    def _map_green_phases(self, tls):
        """Find which green phase serves EB and which serves SB, from the signal program itself.

        Never hardcode phase numbers: netconvert decides the order."""
        logic = traci.trafficlight.getAllProgramLogics(tls)[0]
        links = traci.trafficlight.getControlledLinks(tls)
        edges = self.sc['intersections'][tls]['edges']
        mapping = {}
        for idx, phase in enumerate(logic.phases):
            if 'y' in phase.state or 'G' not in phase.state:
                continue
            first_green = phase.state.index('G')
            in_lane = links[first_green][0][0]
            edge = in_lane.rsplit('_', 1)[0]
            mapping[idx] = edges.index(edge)
        return mapping

    # ----------------------------------------------------------- observation

    def _detectors(self, tls):
        cfg = self.sc['intersections'][tls]
        return cfg['EB'] + cfg['SB']

    def _queue(self, tls):
        """Stopped vehicles seen by this intersection's detectors (the reward signal)."""
        return sum(traci.lanearea.getLastStepHaltingNumber(d) for d in self._detectors(tls))

    def observe(self):
        """What each intersection's cameras report right now."""
        obs = {}
        for tls in self.ids:
            cfg = self.sc['intersections'][tls]
            lanes = np.array([traci.lanearea.getLastStepVehicleNumber(d) for d in self._detectors(tls)],
                             dtype=float)
            phase = traci.trafficlight.getPhase(tls)
            obs[tls] = {
                'lanes': lanes,
                'EB': float(lanes[:len(cfg['EB'])].sum()),
                'SB': float(lanes[len(cfg['EB']):].sum()),
                'green': self._green_dir[tls].get(phase, -1),   # 0 EB, 1 SB, -1 yellow
                'green_time': float(self.t - self._green_since[tls]),
                'queue': float(self._queue(tls)),
            }
        return obs

    # ------------------------------------------------------------------ step

    def _tick(self):
        for tls in self.ids:
            if self.program == 'agent' and traci.trafficlight.getPhase(tls) in self._green_dir[tls]:
                traci.trafficlight.setPhaseDuration(tls, 10_000)   # hold green until the agent switches
        traci.simulationStep()
        self.t += 1
        queues = {tls: self._queue(tls) for tls in self.ids}
        for i in self.ids:
            for j in self.ids:
                self._acc[i][j] += queues[j]
            self._acc_n[i] += 1
        self.queue_trace.append(sum(traci.edge.getLastStepHaltingNumber(e)
                                    for tls in self.ids
                                    for e in self.sc['intersections'][tls]['edges']))
        if self.record:
            for tls, o in self.observe().items():
                edge_queue = sum(traci.edge.getLastStepHaltingNumber(e)
                                 for e in self.sc['intersections'][tls]['edges'])
                self.records.append({'t': self.t, 'tls': tls, 'EB': o['EB'], 'SB': o['SB'],
                                     'green': o['green'], 'queue': o['queue'], 'edge_queue': edge_queue,
                                     **{f'lane_{k}': v for k, v in enumerate(o['lanes'])}})

    def done(self):
        return self.t >= self.episode_s

    def _run_until_ready(self):
        while not self.done():
            ready = [tls for tls in self.ids if self.t >= self._next_decision[tls]]
            if ready:
                return ready, self.observe()
            self._tick()
        return [], self.observe()

    def interval_queues(self, tls):
        """Mean queue at every intersection since this agent's last decision."""
        n = max(1, self._acc_n[tls])
        return {j: self._acc[tls][j] / n for j in self.ids}

    def step(self, actions):
        """Apply {tls: action} for the ready agents, then run until the next decision.

        Returns (ready agents, observations, done). Call interval_queues(tls) for
        each ready agent to get the traffic it experienced since it last acted.
        self.applied holds the actions actually executed (after the max green rule),
        which is what the agents should learn from."""
        self.applied = {}
        for tls, action in actions.items():
            if self.t - self._green_since[tls] + DECISION_S > MAX_GREEN_S:
                action = SWITCH
            self.applied[tls] = action
            if action == SWITCH:
                traci.trafficlight.setPhase(tls, traci.trafficlight.getPhase(tls) + 1)   # yellow
                self._next_decision[tls] = self.t + YELLOW_S + MIN_GREEN_S
                self._green_since[tls] = self.t + YELLOW_S
                self.switches += 1
            else:
                self._next_decision[tls] = self.t + DECISION_S
            self._acc[tls] = {j: 0.0 for j in self.ids}
            self._acc_n[tls] = 0
        self._tick()
        ready, obs = self._run_until_ready()
        return ready, obs, self.done()

    def run_to_end(self):
        """For fixed/actuated programs: nothing to decide, just simulate."""
        while not self.done():
            self._tick()

    # --------------------------------------------------------------- metrics

    def close(self):
        """End the episode and return its performance metrics."""
        backlog = len(traci.simulation.getPendingVehicles())
        traci.close()
        trips = ET.parse(self._tripinfo).getroot().findall('tripinfo')
        shutil.rmtree(self._tmp, ignore_errors=True)
        wait = [float(t.get('waitingTime')) for t in trips]
        travel = [float(t.get('duration')) for t in trips]
        arrived = sum(1 for t in trips if float(t.get('arrival')) >= 0)
        return {
            'avg_queue': float(np.mean(self.queue_trace)),
            'avg_wait_s': float(np.mean(wait)) if wait else 0.0,
            'avg_travel_s': float(np.mean(travel)) if travel else 0.0,
            'throughput': arrived,
            'backlog': backlog,
            'switches': self.switches,
        }
