"""SUMO environment for one or more signalized intersections.

Time runs in 1 s simulation steps. Every intersection is an agent. Each of its
approaches ("arms") gets green in its own phase, and the agent decides which
arm is served:

    'switch' mode (2 arms): action 0 keeps the green, action 1 switches to the other arm
    'select' mode (any number of arms, split phasing): action k gives green to arm k,
                  choosing the arm that already has green means keep

A keep holds the green for another DECISION_S seconds. A change shows 3 s of
yellow on the current arm, then the new arm gets at least MIN_GREEN_S of green.

The agent is only asked when a change is actually allowed (minimum green has
passed), so every action it takes has a real effect. This is the usual
decision model in traffic signal RL (e.g. Wei et al. 2018, Alegre 2019
sumo-rl) and fixes the problem we hit with 0.1 s per-step decisions, where
keep and switch were indistinguishable to the learner.

A green is never held longer than MAX_GREEN_S: at that point a keep is turned
into a change to the next arm, the same way a real controller caps green time.
Without this cap a trained tabular agent could get stuck keeping one green
forever in a rarely visited state (see the README, "What did not work").

Agents decide asynchronously: after a change an agent is busy for 13 s, after a
keep only 5 s. step() applies the actions of the agents that were asked, then
runs the simulation until the next agent is ready.

Programs:
    'agent'    - RL agents own the signal, greens are held until changed
    'fixed'    - the fixed cycle stored in the .net.xml
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


class TrafficEnv:
    def __init__(self, scenario, program='agent', episode_s=1200, gui=False, record=False, routes=None):
        self.name = scenario
        self.sc = SCENARIOS[scenario]
        self.ids = list(self.sc['intersections'])
        self.mode = self.sc['action_mode']
        self.program = program
        self.episode_s = episode_s
        self.gui = gui
        self.routes = routes or self.sc['routes']     # override for demand sweeps
        self.record = record          # keep a per-second log of observations (used by SEMMA)
        self.records = []
        # static layout per intersection: arm names, detectors and edges in arm order
        self._arms, self._dets, self._edges, self._edge_arm = {}, {}, {}, {}
        for tls, cfg in self.sc['intersections'].items():
            self._arms[tls] = list(cfg['approaches'])
            self._dets[tls] = [d for a in cfg['approaches'].values() for d in a['detectors']]
            self._edges[tls] = [e for a in cfg['approaches'].values() for e in a['edges']]
            for k, a in enumerate(cfg['approaches'].values()):
                for e in a['edges']:
                    self._edge_arm[e] = (tls, k)

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
               '-c', self.sc['cfg'], '-r', self.routes, '-a', ','.join(additional),
               '--seed', str(seed), '--step-length', '1', '--time-to-teleport', '-1',
               '--no-warnings', '--no-step-log',
               '--tripinfo-output', self._tripinfo, '--tripinfo-output.write-unfinished',
               '--tripinfo-output.write-undeparted']
        if self.gui:
            cmd += ['--start', '--quit-on-end']
        traci.start(cmd)

        self.t = 0
        self.queue_trace = []
        self.switches = 0
        self._green_dir = {}          # tls -> {green phase index: arm index}
        self._arm_phase = {}          # tls -> {arm index: green phase index}
        self._pending = {}            # tls -> (time, green phase to show after the yellow)
        self._next_decision = {}
        self._green_since = {}
        self._acc = {i: {j: 0.0 for j in self.ids} for i in self.ids}
        self._acc_n = {i: 0 for i in self.ids}
        for tls in self.ids:
            if self.program == 'actuated':
                traci.trafficlight.setProgram(tls, 'actuated')
            self._green_dir[tls] = self._map_green_phases(tls)
            self._arm_phase[tls] = {a: p for p, a in self._green_dir[tls].items()}
            self._next_decision[tls] = MIN_GREEN_S
            self._green_since[tls] = 0

        if self.program != 'agent':
            return [], self.observe()
        return self._run_until_ready()

    def _map_green_phases(self, tls):
        """Give every green phase an index, read from the signal program itself.

        greens='arm' (default): the index is the arm the phase serves. Never
        hardcode phase numbers: netconvert decides the order.
        greens='order': the index is the phase's position among the greens, for
        programs where one phase serves several arms (e.g. both directions of a
        main road)."""
        logic = traci.trafficlight.getAllProgramLogics(tls)[0]
        links = traci.trafficlight.getControlledLinks(tls)
        mapping = {}
        for idx, phase in enumerate(logic.phases):
            if 'y' in phase.state or 'G' not in phase.state:
                continue
            if self.sc.get('greens') == 'order':
                mapping[idx] = len(mapping)
                continue
            first_green = phase.state.index('G')
            in_lane = links[first_green][0][0]
            edge = in_lane.rsplit('_', 1)[0]
            mapping[idx] = self._edge_arm[edge][1]
        return mapping

    # ----------------------------------------------------------- observation

    def _queue(self, tls):
        """Stopped vehicles seen by this intersection's detectors (the reward signal)."""
        return sum(traci.lanearea.getLastStepHaltingNumber(d) for d in self._dets[tls])

    def observe(self):
        """What each intersection's cameras report right now."""
        obs = {}
        for tls in self.ids:
            approaches = self.sc['intersections'][tls]['approaches']
            lanes = np.array([traci.lanearea.getLastStepVehicleNumber(d) for d in self._dets[tls]],
                             dtype=float)
            counts, start = [], 0
            for a in approaches.values():
                counts.append(float(lanes[start:start + len(a['detectors'])].sum()))
                start += len(a['detectors'])
            phase = traci.trafficlight.getPhase(tls)
            o = {
                'lanes': lanes,
                'counts': counts,                                # vehicles seen per arm, in arm order
                'green': self._green_dir[tls].get(phase, -1),   # index of the green shown, -1 during yellow
                'n_greens': len(self._green_dir[tls]),
                'green_time': float(self.t - self._green_since[tls]),
                'queue': float(self._queue(tls)),
            }
            o.update(zip(approaches, counts))                    # also by name, e.g. o['EB']
            obs[tls] = o
        return obs

    # ------------------------------------------------------------------ step

    def _tick(self):
        for tls in self.ids:
            if tls in self._pending and self.t >= self._pending[tls][0]:
                traci.trafficlight.setPhase(tls, self._pending.pop(tls)[1])   # yellow over: show chosen green
            if self.program == 'agent' and traci.trafficlight.getPhase(tls) in self._green_dir[tls]:
                traci.trafficlight.setPhaseDuration(tls, 10_000)   # hold green until the agent changes it
        traci.simulationStep()
        self.t += 1
        queues = {tls: self._queue(tls) for tls in self.ids}
        for i in self.ids:
            for j in self.ids:
                self._acc[i][j] += queues[j]
            self._acc_n[i] += 1
        self.queue_trace.append(sum(traci.edge.getLastStepHaltingNumber(e)
                                    for tls in self.ids for e in self._edges[tls]))
        if self.record:
            for tls, o in self.observe().items():
                edge_queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in self._edges[tls])
                arm_queue = {f'queue_{name}': sum(traci.edge.getLastStepHaltingNumber(e) for e in a['edges'])
                             for name, a in self.sc['intersections'][tls]['approaches'].items()}
                self.records.append({'t': self.t, 'tls': tls, 'green': o['green'], 'queue': o['queue'],
                                     'edge_queue': edge_queue, **arm_queue,
                                     **{name: c for name, c in zip(self._arms[tls], o['counts'])},
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

    def _change(self, tls, target_phase):
        """3 s yellow on the current green, then target_phase (set in _tick)."""
        traci.trafficlight.setPhase(tls, traci.trafficlight.getPhase(tls) + 1)   # its yellow
        if self.mode == 'select':
            self._pending[tls] = (self.t + YELLOW_S, target_phase)
        self._next_decision[tls] = self.t + YELLOW_S + MIN_GREEN_S
        self._green_since[tls] = self.t + YELLOW_S
        self.switches += 1

    def step(self, actions):
        """Apply {tls: action} for the ready agents, then run until the next decision.

        Returns (ready agents, observations, done). Call interval_queues(tls) for
        each ready agent to get the traffic it experienced since it last acted.
        self.applied holds the actions actually executed (after the max green rule),
        which is what the agents should learn from."""
        self.applied = {}
        for tls, action in actions.items():
            too_long = self.t - self._green_since[tls] + DECISION_S > MAX_GREEN_S
            if self.mode == 'switch':
                if too_long:
                    action = SWITCH
                if action == SWITCH:
                    self._change(tls, None)
                else:
                    self._next_decision[tls] = self.t + DECISION_S
            else:
                current = self._green_dir[tls][traci.trafficlight.getPhase(tls)]
                if too_long and action == current:
                    action = (current + 1) % len(self._green_dir[tls])
                if action == current:
                    self._next_decision[tls] = self.t + DECISION_S
                else:
                    self._change(tls, self._arm_phase[tls][action])
            self.applied[tls] = action
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
        all_trips = ET.parse(self._tripinfo).getroot().findall('tripinfo')
        shutil.rmtree(self._tmp, ignore_errors=True)
        trips = [t for t in all_trips if float(t.get('depart')) >= 0]   # vehicles that got into the network
        wait = [float(t.get('waitingTime')) for t in trips]
        travel = [float(t.get('duration')) for t in trips]
        arrived = sum(1 for t in trips if float(t.get('arrival')) >= 0)
        # Total delay counts every vehicle, including those still queued outside the network
        # (depart = -1): time stopped + time waiting to enter. Otherwise a controller that
        # starves an arm until its cars cannot even enter would look better, not worse.
        delay, arm_delay = [], {}
        for t in all_trips:
            d = float(t.get('waitingTime')) + float(t.get('departDelay', 0))
            delay.append(d)
            key = self._trip_arm(t)
            if key is not None:
                arm_delay.setdefault(key, []).append(d)
        return {
            'avg_queue': float(np.mean(self.queue_trace)),
            'avg_wait_s': float(np.mean(wait)) if wait else 0.0,
            'avg_travel_s': float(np.mean(travel)) if travel else 0.0,
            'throughput': arrived,
            'backlog': backlog,
            'switches': self.switches,
            'avg_delay_s': float(np.mean(delay)) if delay else 0.0,
            'max_delay_s': float(max(delay)) if delay else 0.0,
            'arm_delay': {k: float(np.mean(v)) for k, v in arm_delay.items()},
        }

    def _trip_arm(self, t):
        """'tls:arm' a vehicle entered on: from its lane, or for a vehicle that never
        entered, from its flow name (flows are named '<arm>_...')."""
        key = self._edge_arm.get(t.get('departLane', '').rsplit('_', 1)[0])
        if key is None and len(self.ids) == 1:
            arm = t.get('id').split('_')[0]
            if arm in self._arms[self.ids[0]]:
                key = (self.ids[0], self._arms[self.ids[0]].index(arm))
        return None if key is None else f'{key[0]}:{self._arms[key[0]][key[1]]}'
