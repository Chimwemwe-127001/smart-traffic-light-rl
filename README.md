# Smart Traffic Lights with Reinforcement Learning

Traffic signals that learn when to switch, trained and tested in the SUMO
traffic simulator. This repo compares **tabular Q-learning**, a **Deep
Q-Network (DQN)** and **multi-agent Q-learning** (independent vs coordinated
intersections) against fixed-time (hand-set, and properly timed with
Webster's method), actuated, longest-queue-first and random control, on traffic
the agents have never seen. Every learner is trained in 5 independent runs, and
the results cover all 5. Part 2 moves to realistic
roads: a four-way junction with two-way traffic, and a real junction in
Lusaka, Zambia, rebuilt from OpenStreetMap.

| Scenario | Webster fixed-time | Actuated | Best learner | Learner | vs Webster | vs Actuated |
|---|---|---|---|---|---|---|
| Single intersection, shifting demand | 3.8 s | 4.0 s | DQN | **3.2 s** | **-14%** | **-20%** |
| Corridor, 2 intersections, normal demand | **3.2 s** | 3.6 s | Independent Q-learning | 3.5 s | **+10%** (worse) | -1% (tie) |
| Corridor, 2 intersections, heavy demand | 3.8 s | 4.7 s | Independent Q-learning | 3.9 s | +3% (tie) | **-16%** |
| Four-way junction, one lane each way (Part 2) | 77.1 s | 21.6 s | DQN | **19.3 s** | **-75%** | **-11%** |
| Lusaka, Great East Rd / Manda Hill, morning peak (Part 2) | 24.2 s | 25.9 s | DQN | **16.6 s** | **-31%** | **-36%** |

*Mean over 5 independent training runs and 10 held-out traffic seeds, after a
300 s warm-up. v1 rows: average waiting time per vehicle. Part 2 rows: average
delay per vehicle, which also counts time queued before entering the road
(section 8). Differences in bold have a 95% confidence interval, over both
training runs and traffic seeds, that excludes zero. "Webster" is a fixed-time
plan timed properly from measured saturation flows; the hand-set plans are much
weaker (12.1 to 75.1 s). Full tables: [results/RESULTS.md](results/RESULTS.md).*

**In short.** Against a properly timed fixed plan, reinforcement learning earns
its place where traffic is uneven or where many movements compete: the DQN
beats every baseline on the single intersection, the four-way junction and the
Manda Hill junction on Great East Road in Lusaka. On the steady two-signal
corridor, Webster's plan is as good as or better than every learner.

---

## Contents

1. [Why it matters](#1-why-it-matters)
2. [Process flow](#2-process-flow)
3. [Repository structure](#3-repository-structure)
4. [Input: the simulated world](#4-input-the-simulated-world)
5. [Data study (SEMMA)](#5-data-study-semma)
6. [Methods](#6-methods)
7. [Output and performance](#7-output-and-performance)
8. [Part 2: a real junction, four arms and a Lusaka case study](#8-part-2-a-real-junction-four-arms-and-a-lusaka-case-study)
9. [Rubric](#9-rubric)
10. [What did not work, and what we learned](#10-what-did-not-work-and-what-we-learned)
11. [Limitations and next steps](#11-limitations-and-next-steps)
12. [Reproduce](#12-reproduce)
13. [References](#13-references)

---

## 1. Why it matters

**The problem.** Most signals still run fixed timing plans. A plan tuned for
the morning peak wastes green time at noon, and it cannot react when one road
suddenly gets busy. Drivers pay for this in waiting time, fuel and emissions,
and a slow junction pushes its queue onto the next one.

**The idea.** Give every intersection a small learning agent. It reads the
same numbers a traffic camera can produce (vehicles per lane, which road has
green) and learns from experience when to keep or switch the green. Each agent
runs locally at its own junction, and neighbors can share what they see
(distributed multi-agent Q-learning, as proposed in the project write-up).

**Value proposition.** Better use of the roads a city already has, using the
cameras or detectors it already has. No new lanes, no central supercomputer.

**Who it is for.** City and regional traffic departments, smart-city pilot
programs, and traffic engineering consultancies that need a low-cost upgrade
path from fixed-time control.

**How it differs.** Fixed-time plans do not react at all. Actuated control
reacts to gaps between cars but follows hand-set rules. A learning controller
optimizes the actual goal (short queues) from data, and adapts when demand
patterns change, without an engineer re-tuning timings.

**One-year vision.** Part 2 already moves to a real, signalized Lusaka junction
(Great East Road at Manda Hill) rebuilt from OpenStreetMap. Next: replace the estimated demand with measured turning
counts, extend to a stretch of Great East Road with several signals, add
pedestrians, and run a shadow-mode pilot where the agent's decisions are logged
next to the real controller's before it is ever given control.

---

## 2. Process flow

```mermaid
flowchart LR
    A[Build networks<br/>and demand] --> B[SEMMA data study<br/>Sample, Explore, Modify]
    B -->|bins and scales| C[Train agents<br/>5 runs x 900 episodes,<br/>seeds 0 to 899]
    W[Webster plan from<br/>measured saturation flow] --> D
    C -->|best checkpoint on<br/>validation seeds 2000 to 2009| D[Evaluate on held-out<br/>seeds 1000 to 1009]
    D --> E[Before vs after,<br/>head to head, probes]
    E --> F[Rubric<br/>pass or fail]
    B -.->|found detector bug| A
```

Inside every training episode the agent and the simulator run the standard
reinforcement learning loop:

```mermaid
flowchart LR
    S[SUMO simulation] -->|detector counts,<br/>green direction,<br/>green age| AG[Agent]
    AG -->|keep or switch| S
    S -->|reward = minus mean queue<br/>since last decision| AG
```

---

## 3. Repository structure

```
smart-traffic-light-rl/
├── traffic_rl/                 the library: one idea per file
│   ├── scenarios.py            the 5 scenarios: files, arms, detectors, action mode
│   ├── env.py                  SUMO environment: min/max green, yellow, async decisions, metrics
│   ├── runner.py               one episode loop shared by training and evaluation
│   ├── baselines.py            fixed-time, Webster, actuated, random, longest queue first + queue reward
│   ├── state.py                detector counts -> table state / DQN input (uses SEMMA config)
│   ├── q_learning.py           tabular Q-learning (independent learners on the corridor)
│   ├── dqn.py                  NumPy DQN: MLP + Adam, replay buffer, target net, Double DQN
│   ├── multi_agent.py          coordinated Q-learning: neighbor state + shared reward
│   └── metrics.py              bootstrap CIs, paired differences, two-level bootstrap over runs
├── experiments/
│   ├── semma.py                data study: sample, explore, modify
│   ├── webster.py              measures saturation flow in SUMO, times a Webster fixed-time plan
│   ├── train.py                train any agent on any scenario (one training run)
│   ├── train_all.py            every learner x 5 runs, in parallel
│   ├── evaluate.py             held-out evaluation, probes, rubric -> results/RESULTS.md
│   └── make_figures.py         every figure in this README
├── networks/
│   ├── build_networks.py       builds four_way/ and manda_hill/ (left-hand traffic, phases, detectors, demand)
│   ├── single/                 1 signal, 3-lane EB and SB approaches, 6 detectors
│   ├── corridor/               2 signals 285 m apart, normal and heavy demand
│   ├── four_way/               4 two-way arms, one lane each way, split phasing
│   ├── manda_hill/             Great East Rd / Manda Hill from OpenStreetMap, 3 demand levels
│   └── */webster.add.xml       the Webster fixed-time plan of each network
├── results/
│   ├── semma/                  sampled data, exploration figures, state_config.json
│   ├── logs/                   training and validation curves (CSV), one per run
│   ├── models/                 trained agents, 5 runs each (Q-tables as JSON, DQN weights as NPZ)
│   ├── figures/                result figures and sumo-gui screenshots
│   ├── webster.json            measured saturation flows, lost times and the Webster plans
│   ├── evaluation.csv          every controller x run x held-out seed
│   └── RESULTS.md              all tables and the rubric
├── docs/
│   ├── SEMMA.md                the data study write-up
│   └── CONCEPTS.md             a short course on every concept used here
└── tests/
    ├── test_agents.py          unit tests (Q update, gradient check, replay, LQF, Webster, statistics)
    └── test_env.py             SUMO tests (min/max green, yellow, warm-up, Webster plan, left-hand driving)
```

---

## 4. Input: the simulated world

<table>
<tr>
<td><img src="results/figures/gui_single_t250s.png" alt="Single intersection in sumo-gui"></td>
<td><img src="results/figures/gui_corridor_heavy_t400s.png" alt="Corridor in sumo-gui"></td>
</tr>
<tr>
<td align="center"><b>Single intersection</b> at t = 250 s. Cyan strips are the<br/>lane-area detectors, the agent's "cameras".</td>
<td align="center"><b>Corridor, heavy demand</b> at t = 400 s. Eastbound<br/>platoons leave Node2 and arrive at Node5.</td>
</tr>
</table>

| Scenario | Network | Demand (random arrivals) | Episode |
|---|---|---|---|
| `single` | 1 signal, 3 lanes per approach | EB 1500 / SB 500 veh/h, then 1000 / 1000, then 500 / 1500 | 300 s warm-up + 1200 s |
| `corridor` | 2 signals, EB 3 lanes, SB 2 lanes | EB 900, SB 400 at each junction (veh/h) | 300 s warm-up + 1200 s |
| `corridor_heavy` | same | EB 1400, SB 550 at each junction (veh/h) | 300 s warm-up + 1200 s |

Arrivals are random (a probability of a new car every second), so each seed
is different traffic. Every episode starts with empty roads, so the first
300 s are a **warm-up**: the controllers run as usual, but no metric counts
them. The seed ranges never overlap: 0 to 899 for training, 900 to 902 for the
SEMMA sample, 1000 to 1009 for the final test and 2000 to 2009 for validation.

**The decision problem (a Markov Decision Process):**

| | Definition |
|---|---|
| State (Q-learning) | EB count bin, SB count bin, which road has green, green age bin. Coordinated agents add the neighbor's green and count bin. |
| State (DQN) | 6 lane counts / 8, green one-hot, green age / 60 (9 numbers) |
| Action | 0 keep green for 5 more seconds, 1 switch (3 s yellow, then at least 10 s green) |
| Reward | minus the mean number of stopped vehicles at the detectors since the last decision. Coordinated agents also subtract 0.5 x the neighbor's queue. |
| Constraints | minimum green 10 s, maximum green 60 s, yellow 3 s. The same rules a real controller follows. |

---

## 5. Data study (SEMMA)

Before training we studied the detector data with the **SEMMA** process
(Sample, Explore, Modify, Model, Assess). Full write-up: [docs/SEMMA.md](docs/SEMMA.md).

- **Sample:** 67,500 per-second detector readings from fixed-time, actuated and random control on all three scenarios.
- **Explore:** counts are right-skewed, demand shifts as designed, and lanes within an approach are strongly correlated (0.79).
- **Found a data bug:** the detectors ended 10 to 16 m before the stop line, so on the corridor they saw only **37%** of the real queue. After moving them to the stop line, the same run showed **108%** (slightly over 100% because detectors count cars under 1.39 m/s as stopped). This was fixed before any training.
- **Modify:** the Q-learning bin edges (2, 4, 7, 11 vehicles) are the 25th, 50th, 75th and 90th percentiles of the data. The DQN input and reward scales are the 99th percentiles.

<table>
<tr>
<td><img src="results/semma/explore_demand_over_time.png" alt="Demand over time"></td>
<td><img src="results/semma/explore_lane_correlation.png" alt="Lane correlation" width="360"></td>
</tr>
</table>

---

## 6. Methods

**Tabular Q-learning** (Watkins and Dayan, 1992). A table of values Q(s, a),
updated after every decision:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]      alpha = 0.1, gamma = 0.9
```

**Deep Q-Network** (Mnih et al., 2015). A neural network (9 -> 32 -> 32 -> 2,
ReLU) replaces the table, so the agent can use all six lane counts and
generalize between similar states. It uses experience replay (buffer of 20,000,
batches of 64), a target network synced every 250 updates, the Double DQN target
(van Hasselt et al., 2016) and Huber loss. Written in plain NumPy, including
backpropagation and Adam, with a unit test that checks the gradients against
finite differences.

**Multi-agent Q-learning.** One Q-learning agent per intersection.
- *Independent* (Tan, 1993): each agent sees only its own detectors.
- *Coordinated*: neighbors share which road has green and how busy they are
  (one-hop message passing), and each agent's reward includes half of its
  neighbor's queue, so it "feels" the congestion it sends downstream. This is
  the cooperative reward shaping used in networked traffic control (Chu et
  al., 2019).

**Training.** Every agent, on every scenario, gets the same budget: 900
episodes of 25 simulated minutes each (5 of warm-up), so all learning curves
and comparisons are like for like. Epsilon-greedy exploration decays from 1.0
to 0.05 over the first 60% of episodes (540). Every 10 episodes the greedy
policy is scored on 10 validation seeds, and the best checkpoint is kept.

**Five training runs per learner.** One training run can be lucky or unlucky
(Henderson et al., 2018), so every learner is trained 5 times from scratch.
The run number seeds the initial weights, the exploration and the order in
which the 900 training traffic draws are met. All 5 trained models are
evaluated, and the tables report their mean.

**A fair fixed-time baseline.** The hand-set fixed-time plans (42 s per road,
30 s per arm, 40/10/20 s) were never optimized, so beating them says little.
`experiments/webster.py` therefore builds the plan a traffic engineer would:
it measures, in SUMO, how fast each green discharges a queue (saturation flow,
from the headways of the 5th queued car on, the Highway Capacity Manual's
field method) and how much of each phase is lost (start-up and unused yellow),
then times the cycle with Webster's (1958) formula from the average demand.
Both fixed-time plans are reported; the rubric uses Webster's.

**Evaluation protocol.** Every controller runs on the same 10 held-out seeds.
Because the traffic is identical for every controller, we compare them seed by
seed (paired differences) and report 95% percentile bootstrap confidence
intervals (Efron and Tibshirani, 1993). For a learner the bootstrap works at
two levels: each resample draws 5 training runs, then 10 traffic seeds, so the
interval covers both "which traffic" and "which training run". "Before
training" means the same agent with its initial parameters, so before vs after
isolates what learning added. The agents are trained on queue length, and we
also report waiting time and travel time, which they never optimized directly.
Every metric leaves out the 300 s warm-up.

---

## 7. Output and performance

### Before vs after training

![Before vs after training](results/figures/before_after.png)

| Scenario | Learner | Wait before | Wait after | Change |
|---|---|---|---|---|
| Single | Q-learning | 17.1 s | 3.3 s | **-81%** |
| Single | DQN | 27.0 s | 3.2 s | **-88%** |
| Corridor | Independent QL | 21.0 s | 3.5 s | **-83%** |
| Corridor | Coordinated QL | 21.0 s | 4.3 s | **-79%** |
| Corridor heavy | Independent QL | 22.9 s | 3.9 s | **-83%** |
| Corridor heavy | Coordinated QL | 22.9 s | 4.6 s | **-80%** |

*Mean over the 5 runs, each before and after its own training.*

An untrained Q-table has all values at zero, so it always keeps the green
until the 60 s maximum forces a switch. That is a slow fixed cycle, worse than
even the hand-set 42 s plan, and it is the same for every run. An untrained DQN
starts from random weights, so it depends on the run: three of the five
initial networks happen to switch often and already score 3.8 s, the other two
score 28 s and 95 s. Training brings all five to 3.2 s. Five runs show both
sides; a single run would have shown only one.

### Learning curves

Each curve is the greedy policy scored on the 10 validation seeds every 10
episodes, all on the same 0 to 900 episode axis: the line is the mean of the 5
runs and the band goes from the best to the worst run. The dashed lines are the
baselines' scores on the held-out test seeds.

![Learning curve, single intersection](results/figures/learning_curve_single.png)

On the single intersection the DQN gets below both actuated control and the
Webster plan within 10 episodes. Q-learning hovers around them for about 250
episodes, then settles below. All 5 runs of each learner stay close together.
The DQN is the smoother learner: the network generalizes, so one bad episode
does not flip whole regions of the policy the way it can in a table.

![Learning curve, corridor normal demand](results/figures/learning_curve_corridor.png)

![Learning curve, corridor heavy demand](results/figures/learning_curve_corridor_heavy.png)

On both corridors the independent learners settle near actuated control
within about 300 episodes, but never reach the Webster line. The coordinated
learners stay above them the whole way; they keep closing the gap until
episode 900, but do not catch up.

### Head to head (held-out traffic)

![Head to head](results/figures/head_to_head.png)

| Controller | Single | Corridor | Corridor heavy |
|---|---|---|---|
| Fixed-time, hand-set | 12.1 s | 16.4 s | 19.6 s |
| Fixed-time, Webster | 3.8 s | **3.2 s** | **3.8 s** |
| Random switching | 6.3 s | 8.1 s | 10.2 s |
| Actuated (SUMO) | 4.0 s | 3.6 s | 4.7 s |
| Q-learning / Independent QL | 3.3 s | 3.5 s | **3.9 s** |
| DQN | **3.2 s** | | |
| Coordinated QL | | 4.3 s | 4.6 s |

*Average waiting time per vehicle, mean over 5 runs and 10 seeds; bold = best,
or tied with the best. Paired confidence intervals for every learner against
Webster, actuated control and longest queue first are in
[results/RESULTS.md](results/RESULTS.md#trained-learners-vs-webster-fixed-time).
Every learner serves the full demand (100.2% to 100.8% of Webster's throughput).*

**A properly timed fixed plan is hard to beat on steady traffic.** Replacing
the hand-set plans with Webster's changes the v1 story. On the single
intersection, where demand shifts three times, both learners still beat
Webster (DQN -0.54 s, CI -0.70 to -0.39). On the corridor the demand is steady
for the whole hour, which is exactly what Webster's method assumes, and its
short cycle (26 to 28 s) serves it well: independent QL is 0.32 s worse (CI +0.17 to
+0.47) and under heavy demand the two tie (+0.11 s, CI -0.03 to +0.25), while
independent QL still beats actuated control there (-0.73 s, CI -0.99 to -0.42).
The short fixed cycle also caps how long any single car can wait: every learner
has a longer worst-case wait than Webster on the v1 networks (by 16 to 30 s).

**DQN vs Q-learning.** On the single intersection the DQN is slightly better
(3.2 vs 3.3 s) and much more robust. The Q-value probes below show why: every
one of the 5 DQNs answers all 4 hand-made traffic situations correctly. The
Q-tables get the two "switch" cases right, but none of them visited the two
"keep" states during training, so they have no answer for them (the one run that did visit one of them
chose the wrong action there). A table cannot generalize
to states it has not seen. This is the "huge knowledge space" argument for deep
RL made concrete.

| Probe state | Right answer | Q-learning (5 runs) | DQN (5 runs) |
|---|---|---|---|
| SB busy, EB empty, EB has green | switch | switch in 5 of 5 | switch in 5 of 5 |
| EB busy, SB empty, SB has green | switch | switch in 5 of 5 | switch in 5 of 5 |
| EB busy, SB empty, EB has green | keep | never visited in 5 of 5 | keep in 5 of 5 |
| SB busy, EB empty, SB has green | keep | never visited in 4, "switch" in 1 | keep in 5 of 5 |

**Independent vs coordinated.** This is the research question from the
write-up, and the honest answer here is **no, coordination did not help**.
Coordinated agents were 0.8 s (normal) and 0.7 s (heavy) slower than
independent ones (95% CI over runs and seeds: +0.6 to +1.0 s and +0.5 to
+0.9 s). Under heavy demand they did at least match actuated control
(-0.03 s, CI -0.32 to +0.27, a tie). All 5 coordinated runs agree (4.1 to 4.5 s
on the normal corridor), so this is not one unlucky run. The likely reasons:

- *State explosion.* The neighbor information multiplies the table size by
  about 10 (about 2,360 to 2,550 states vs 195 to 236, see the growth curve
  below). Each state is visited far less often, so its value estimate stays noisier.
- *The neighbor signal is weak here.* The two signals are 285 m apart and
  platoons disperse on the way, so the neighbor's current phase says little
  about arrivals in the next 5 seconds.
- *Shared reward adds noise.* Half of the neighbor's queue enters the reward,
  but the agent has only partial control over it.

This agrees with the literature: coordination pays off in dense grids and
near saturation, and it needs function approximation (e.g. a DQN per agent)
to cope with the larger state. That is the natural next step (section 11).

![Training diagnostics](results/figures/training_diagnostics.png)

---

## 8. Part 2: a real junction, four arms and a Lusaka case study

The v1 networks were simplified: one-way approaches and only two arms. Part 2
asks the same questions on roads that look like real ones.

### 8.1 Why Great East Road at Manda Hill

Great East Road (T4) is one of Lusaka's main arterial roads, and its junction
with **Manchinchi Road and Addis Ababa Drive**, next to Manda Hill Mall, is one
of the busiest signalized crossroads on it. It is a fair test: three lanes each
way on the main road, turn lanes, slip roads, and side roads that compete for
green. (Earlier versions modelled Great East Road / Lufubu Road, but its signals
were switched off in 2021 (Lusaka Times, 2021) and OpenStreetMap shows none
there today, so the case study moved to a junction that is still signalized.)

Zambia drives on the **left**, so every Part 2 network is built with
`netconvert --lefthand`. Cars keep left, and the right turn is the movement
that crosses oncoming traffic. A test checks every arm of both networks.

### 8.2 The two new networks

<table>
<tr>
<td><img src="results/figures/osm_manda_hill.png" alt="Great East Road and Manda Hill junction on OpenStreetMap"></td>
<td><img src="results/figures/gui_manda_hill_t600s.png" alt="Manda Hill junction in sumo-gui"></td>
</tr>
<tr>
<td align="center"><b>The real junction</b> (OpenStreetMap standard layer): Great East Road<br/>crosses Manchinchi Road (north-west) and Addis Ababa Drive (south-east).</td>
<td align="center"><b>The SUMO model</b> at t = 600 s: straight dual carriageways, turn lanes,<br/>slip roads, and the detectors (cyan) running up to every stop line.</td>
</tr>
</table>

<img src="results/figures/gui_four_way_t120s.png" alt="Four-way junction in sumo-gui" width="480">

*The four-way junction: every road two-way with one lane each direction. The north arm queues under fixed-time control.*

| | Four-way junction | Manda Hill junction |
|---|---|---|
| Geometry | 4 arms of 200 m, one lane in and one lane out each | From an OpenStreetMap extract: bearings, lane counts, turn lanes and speed limits (60 to 80 km/h) from the OSM tags. Straight dual carriageways with medians; Great East Road 3 lanes each way (4 at the stop line), Addis Ababa Drive 3 lanes, Manchinchi Road 1 lane widening to 3; the three free left turns use slip roads that give way where they merge |
| Signal program | Split phasing: one arm green at a time (common in Zambia) | 4 greens, every turn protected: Great East Road through and left, Great East Road right turns, Manchinchi Road, Addis Ababa Drive |
| Agent's action | Which arm to serve next (4 actions) | Which of the 4 greens to show next |
| Fixed-time plans | hand-set 30 s per arm; Webster 15 / 15 / 13 / 13 s (N, E, S, W) | hand-set 40 / 12 / 18 / 20 s; Webster 20 / 14 / 10 / 13 s (3 s yellow after each green) |
| Detectors | 105 m per approach lane | One continuous detector per stop-line lane, 105 m back through the widening onto the normal road; slip roads have none |
| Demand | 60% straight, 20% left, 20% right; N-S heavy, then balanced, then E-W heavy | Morning peak, see the sources below |
| Built by | `networks/build_networks.py four_way` | `networks/build_networks.py manda_hill` |

**Where the Manda Hill demand comes from.** No public turning counts exist for this junction. The demand is built from one old published daily volume and assumed planning factors, and then tested at ±25%. Each row says whether it is measured, reported or assumed:

| Quantity | Value | Status | Source |
|---|---|---|---|
| Geometry, lanes, turn lanes, speed limits | as mapped | Measured | OpenStreetMap extract, committed in `networks/manda_hill/` |
| Great East Road daily traffic | about 31,000 veh/day, **in 2009** | Reported | Choongo (2020), UNZA; counting place and method not stated |
| Peak-hour share K | 0.09 | Assumed | A common planning value, not measured in Lusaka |
| Direction split D | 0.6 towards the city (westbound) | Assumed | The value is assumed; the direction follows the westbound morning peak quoted by Ng'andu (2024) |
| Great East Road peak volumes | 1,674 veh/h westbound, 1,116 eastbound | Derived | 31,000 × K × D |
| Manchinchi Road and Addis Ababa Drive | 600 and 700 veh/h | Assumed | No source |
| Turning shares | Great East Road 70 to 75% through; Addis Ababa Drive 35% right; Manchinchi Road 30% left, no right turn | Assumed | Guided by how many lanes OSM gives each movement |
| Signal timings | 40 / 12 / 18 / 20 s | Assumed | The real plan is not public |
| Sensitivity | every controller re-tested at ×0.75 and ×1.25 | | Section 8.5 |

**A check against a recent count.** Ng'andu (2024) counted 1,545 veh/h on Great East Road from video on a weekday between 07:00 and 08:00 in 2022, near Bwinjimfumu bus stop, not at this junction. The study does not say whether the count covers one direction or both. If it is one direction, our westbound 1,674 veh/h is 8% higher. If it is both, our two-way 2,790 veh/h is 80% higher, and even the ×0.75 sweep (2,093 veh/h) stays above it. The same study quotes Lusaka City Council peak periods of 07:30 to 08:30 westbound and 15:30 to 18:30 eastbound. The Manda Hill results therefore compare controllers on the real geometry under a plausible but possibly high peak. They are not predictions of real delays at this junction.

### 8.3 Data study (SEMMA) on the new junctions

The same SEMMA steps were applied to 27,000 new per-second samples. The full
write-up, with all figures, is in [docs/SEMMA.md](docs/SEMMA.md#part-2-the-four-way-and-manda-hill-junctions).

- **The detectors see the queue.** Where a road widens before the stop line,
  SUMO joins the two road pieces with a short connector, and a detector per piece
  left a blind spot of about 10 m there (4% of the stopped cars on the city-bound
  arm). Each stop-line lane now has one continuous detector over the connected
  lanes. With 105 m of reach they see 99% of the real queue (250 m: 111%), so
  105 m was kept.
- **The city-bound arm queues most.** Great East Road towards the city holds
  17.7 stopped cars on average under the hand-set plan and 10.1 under actuated
  control, against 5 to 7 on the other arms.

<table>
<tr>
<td><img src="results/semma/manda_hill_explore_arm_queue.png" alt="Manda Hill stopped cars per arm"></td>
<td><img src="results/semma/manda_hill_explore_arm_counts.png" alt="Manda Hill vehicles seen per arm"></td>
</tr>
</table>

- **Bins from the data.** The Q-learning bin edges (6, 11, 18, 32 vehicles per
  arm) are quantiles of what the arms see. Each junction gets its own bins and
  scales in `state_config.json`.
- **Turn lanes behave differently.** The lanes of Manchinchi Road and Addis
  Ababa Drive move together (correlation 0.76), but those of Great East Road
  much less (0.44 and 0.50), because its median-side lane is a right-turn lane
  with its own green. The Q-learning state keeps one count per arm, as in v1;
  the DQN gets every lane, so it can see the right-turn queues on their own.

### 8.4 A new metric: delay, not just wait

At a busy junction the queue can grow past the end of the modelled road, and
those cars never "enter". A controller that starves an arm could therefore
*lower* the average wait, simply by keeping cars out. So the Part 2 junctions
are judged on **average delay**: time stopped plus time queued before
entering, for every car, including those still outside at the end. Checkpoint
selection during training uses the same metric. When it was added (v1.1), a
regression check reproduced every v1 result exactly before any agent was
retrained.

### 8.5 Results

![Head to head on the new junctions](results/figures/head_to_head_v11.png)

| Controller | Four-way delay | Four-way worst delay | Manda Hill delay | Manda Hill worst delay | Manda Hill cars through |
|---|---|---|---|---|---|
| Fixed-time, hand-set | 75.1 s | 262 s | 35.6 s | 335 s | 1,353 |
| Fixed-time, Webster | 77.1 s | 249 s | 24.2 s | 175 s | **1,373** |
| Random | 126.2 s | 461 s | 104.5 s | 379 s | 1,222 |
| Actuated | 21.6 s | **78 s** | 25.9 s | **150 s** | 1,371 |
| Longest queue first | 22.7 s | 157 s | 23.9 s | 196 s | 1,355 |
| Q-learning | 74.7 s | 304 s | 42.8 s | 345 s | 1,333 |
| **DQN** | **19.3 s** | 141 s | **16.6 s** | 154 s | 1,366 |

*Mean over 10 held-out seeds (learners: and over 5 training runs). "Cars through" = vehicles that completed their trip in the 20 measured minutes.*

**Before vs after training** (average delay, same seeds): Q-learning 315.4 → 74.7 s (four-way) and 202.7 → 42.8 s (Manda Hill); DQN 395.7 → 19.3 s and 376.5 → 16.6 s. Every change has a 95% CI below zero.

**What the numbers say:**

- **The DQN is the best controller on both junctions.** It beats Webster,
  actuated control and longest-queue-first at both, with CIs over runs and
  seeds that exclude zero. At Manda Hill it cuts average delay by 36% against
  actuated control (-9.2 s, CI -10.3 to -8.3) and by 31% against the Webster
  plan (-7.6 s, CI -8.9 to -6.0). On the four-way the margin over actuated
  control is smaller but consistent (-2.3 s, CI -3.2 to -1.4, or 11%).
- **Every one of the five DQN runs wins.** The runs score 19.0 to 19.9 s on the
  four-way and 16.4 to 16.9 s at Manda Hill, all below every baseline. All 5
  runs also answer every logic probe on the single intersection, the four-way
  and Manda Hill: for example, when Manchinchi Road is busy and Addis Ababa
  Drive has green, all five switch to Manchinchi Road.
- **A well-timed fixed plan is strong at Manda Hill.** Demand there is steady
  through the peak hour, which is what Webster's method assumes. Its plan
  (24.2 s) is 32% better than the hand-set one (35.6 s) and matches actuated
  control (25.9 s) and longest-queue-first (23.9 s). On the four-way, where
  demand moves from one road to another, the same method does not help
  (77.1 s).
- **Tabular Q-learning breaks down on four arms.** With a bin per arm, the
  table grows to about 3,270 states on the four-way and 2,560 at Manda Hill,
  and 900 episodes cannot fill it. At Manda Hill it is worse than every
  baseline except random (42.8 s; runs 38.5 to 48.2 s), and no run answers all
  four probes; two of the four probe states never came up in its training.
  This is the "huge knowledge space" argument for deep Q-networks made
  concrete.
- **Fairness: the average hides the unlucky driver.**

![Fairness](results/figures/fairness_v11.png)

  - **At Manda Hill, the DQN is fair as well as fast.** Its worst single delay
    (154 s) is about the same as actuated control's (150 s), and its average
    delay per arm is balanced (11.5 to 25.0 s). The Webster plan keeps the main
    road moving but leaves side-road cars waiting 41 to 42 s on average.
  - **On the four-way, the DQN fails the fairness check.** Its per-arm averages
    are balanced (18 to 21 s), but its worst single delay (141 s) is 1.8 times
    actuated control's (78 s). Actuated control cycles regularly, which keeps
    the tail short. We report this as a trade-off, not a win.

**Demand sweep.** Because the Manda Hill volumes are estimates, every controller was re-run at 75% and 125% of them:

![Manda Hill demand sweep](results/figures/manda_hill_sweep.png)

The DQN is the best at 75% and 100% of the estimated demand (11.7 s and
16.6 s, against 15.9 s and 25.9 s for actuated control). At 125% the junction
is near capacity: the Webster plan (44.6 s) and the DQN (46.2 s) come out
close, both far ahead of actuated control (93.9 s), but the DQN's worst single
delay grows to 574 s, against 305 s for Webster. Trained only at 100%, it has
not learned what to do in overload. Training on a range of demand levels is
the obvious next step.

**Learning curves on the new junctions** (average delay on the validation seeds):

![Learning curve, four-way junction](results/figures/learning_curve_four_way.png)

On the four-way the DQN drops below actuated control within about 170
episodes and keeps creeping down to about 18.5 s by episode 900, with all 5
runs close together. Tabular Q-learning improves quickly for about 200
episodes, then hovers around the Webster line for the rest of training: its
table is too big to fill.

![Learning curve, Manda Hill junction](results/figures/learning_curve_manda_hill.png)

At Manda Hill the DQN passes every baseline within about 40 episodes and then
stays at about 17 s for the rest of training, with the 5 runs close together.
The selected checkpoints scored 15.9 to 16.4 s on the validation seeds and
16.6 s on the test seeds, so the gap from picking the best checkpoint is small.
Tabular Q-learning levels off at 45 to 50 s.

---

## 9. Rubric

Scored automatically by `experiments/evaluate.py`. **55 of 76 checks pass.** v1
scenarios are judged on average wait, Part 2 junctions on average delay. Every
CI covers both the 5 training runs and the 10 traffic seeds. Most failures are
tabular Q-learning on the four-arm junctions and the corridor learners against
Webster's plan.

| Criterion | Measure | Threshold | Result |
|---|---|---|---|
| Every learner learned (10 checks) | trained minus untrained, paired 95% CI | CI below 0 | **10 of 10 PASS** |
| Beats Webster fixed-time (10) | paired 95% CI | CI below 0 | 4 of 10 (fails: all four corridor checks, and Q-learning on both four-arm junctions) |
| Beats random switching (10) | paired 95% CI | CI below 0 | **10 of 10 PASS** |
| Competitive with actuated control (10) | vs actuated, paired 95% CI | within +10% | 7 of 10 (fails: coordinated corridor +21%, Q-learning four-way +247%, Q-learning Manda Hill +65%) |
| Beats longest queue first (4, Part 2) | paired 95% CI | CI below 0 | 2 of 4 (the DQN on both junctions; Q-learning on neither) |
| Starves no one (4, Part 2) | worst single delay vs actuated | at most 1.25 x | 1 of 4 (only the DQN at Manda Hill: 154 s vs 150 s) |
| Learned traffic logic (6) | runs answering every hand-made probe | at least 3 of 5 runs | 3 of 6 (every DQN passes; no Q-table does) |
| Stable learning (10) | TD loss, last 20 vs first 20 episodes | in at least 3 of 5 runs | **10 of 10 PASS** (5 of 5 runs each) |
| Coordination helps (2) | coordinated minus independent wait, 95% CI | CI below 0 | 0 of 2 (+0.8 s, +0.7 s) |
| Serves all demand (10) | vehicles completed vs Webster fixed-time | at least 98% | 8 of 10 (fails: Q-learning on the four-way 95.6%, at Manda Hill 97.1%) |

Every row with the per-check numbers is in [results/RESULTS.md](results/RESULTS.md#rubric).

---

## 10. What did not work, and what we learned

These dead ends are part of the result:

1. **Deciding every 0.1 s.** The first DQN (adapted from the tutorial) chose
   keep or switch at every 0.1 s step with a per-sample online update. At that
   time scale one action barely changes anything, so Q(keep) and Q(switch) were
   equal up to noise. The trained policy was *worse than an untrained network*
   on held-out traffic. Adding replay and a target network alone did not fix it.
   The fix was the decision model: decide only when a switch is allowed, then
   every 5 s. That is the standard in traffic signal RL.
2. **Detectors that missed the queue.** Found by the SEMMA data study (section 5).
3. **Greedy lock-in in the Q-table.** Early trained Q-tables sometimes kept one
   green for the whole 20 minutes on a test seed (197 s average wait). In a
   rarely visited state, both Q-values were still near their zero start, "keep"
   led back into the same state, and a greedy agent never escaped. Exploration
   had hidden this during training. The fix was a 60 s maximum green, a rule
   every real controller has and the actuated baseline already used.
4. **Port clashes in parallel runs.** Several runs connecting to SUMO over
   TraCI sockets occasionally picked the same port and crashed. Switching to
   `libsumo` (SUMO inside the Python process) removed the ports and made
   episodes 7.6x faster, with identical results.
5. **A metric that could reward starvation.** Average wait only counts cars
   that entered the modelled road. On a busy Great East Road junction a
   controller could push its queue outside and look better. Found while checking a result that seemed too good.
   The fix was total delay, for evaluation and for checkpoint selection
   (section 8.4). The good result survived the stricter metric.
6. **Detectors too short for a dual carriageway.** On an earlier Great East
   Road model with two lanes each way, 105 m detectors filled up and saw only
   part of the queue. The SEMMA coverage check (section 8.3) now runs for every
   new junction; at Manda Hill, with three lanes each way, 105 m sees 99%.
7. **Split phasing does not suit a busy dual carriageway.** Serving one arm at a
   time would oversaturate Great East Road at about 1,700 veh/h. The Manda Hill
   junction uses a main road / right turns / Manchinchi Road / Addis Ababa Drive
   program instead.
8. **A weak baseline.** Up to v1.2.1 the fixed-time baseline was a hand-set plan
   that nobody had optimized, and every learner looked 70 to 80% better than
   fixed-time on the v1 networks. A plan timed with Webster's method from
   measured saturation flows is as good as actuated control there, and on the
   steady corridor it beats every learner. The comparison that matters is
   against the best simple method, not the easiest one.
9. **One training run is not a result.** Up to v1.2.1 each learner was trained
   once. The DQN on an earlier Great East Road junction scored 23.9 s or
   32.3 s depending on the budget, and nothing showed whether that was training
   or luck. With 5 runs, Manda Hill Q-learning spans 38.5 to 48.2 s while the
   DQN spans 16.4 to 16.9 s. Every learner
   is now trained 5 times and judged over all of them.
10. **Three validation seeds overfit.** Picking the best of 90 checkpoints on 3
   validation seeds chose models that were lucky on those seeds (17.9 s on
   validation, 32.3 s on test, on the earlier Great East Road junction). With 10
   validation seeds the gap is small: at Manda Hill 15.9 to 16.4 s on validation
   against 16.6 s on test.
11. **Permitted right turns locked the junction.** The first Manda Hill program
   let right turns go on the same green as the oncoming traffic, giving way
   inside the junction. The waiting cars were still there when the lights
   changed, blocked the next movement, and the junction jammed for good. Every
   turn now has its own protected green (section 8.2 and
   [CONCEPTS](docs/CONCEPTS.md)).
12. **Blind spots between detectors.** Where Great East Road widens before the
   stop line, SUMO joins the two road pieces with a short connector. One
   detector per piece left about 10 m unseen, so a car could disappear from the
   count for a moment. One detector now runs over the whole chain of connected
   lanes, and the approaches are covered without a gap.
13. **A baseline that counted the wrong cars.** Longest queue first first
   compared whole arms. At Manda Hill two greens serve the same arms (through
   traffic, then right turns), so their totals tied and it kept picking the
   wrong one (160.7 s on three check seeds). Counting only the lanes each green
   serves brought it to 23.5 s on the same seeds, a fair and strong baseline.

---

## 11. Limitations and next steps

- **Manda Hill demand is estimated**, not counted: a daily volume from 2009,
  assumed peak factors, and assumed side-road volumes and turning shares. It
  may be high compared with a 2022 count (section 8.2). The ×0.75 / ×1.25
  sweep shows the ranking holds, but real turning counts from the Council or a
  video survey are the most valuable next step.
- **Only the junction itself is modelled**: 240 to 300 m arms, with no
  neighbouring junctions, minibus stops, informal parking or pedestrians. The
  east arm ends before the Manda Hill Mall entrance signal, which is not
  modelled. Each affects real capacity on Great East Road.
- **The real signal timings are not public.** The hand-set plan and the phase
  order at Manda Hill are assumptions, so the hand-set row is not the real
  controller. Webster's plan is the fairer stand-in.
- **Detector reach on the four-way.** The 105 m single-lane detectors sometimes
  fill up (14 cars) on the busy north arm, so the agent then sees only part of
  the queue. Coverage is still 89%, but longer detectors may help there too.
- **Simulated detectors.** Real cameras add noise, occlusion and delay; the
  agent should be trained with noisy counts before any field test.
- **The DQN's worst-case delay** on the four-way is 1.8 times actuated
  control's. A fairness term in the reward (e.g. penalising the longest wait)
  is the obvious next experiment.
- **The DQN in overload.** At 125% of the Manda Hill demand the DQN (46.2 s)
  no longer beats Webster's plan (44.6 s), and its worst single delay is 574 s
  against 305 s. It was trained at 100% only; training over a range of demand
  levels is the next step.
- **One fixed plan per scenario.** Webster's plan uses the average demand. A
  traffic engineer would add time-of-day plans where demand shifts (the single
  intersection and the four-way) and green-wave offsets on the corridor. Both
  would make the fixed-time baseline stronger still.
- **Untuned hyperparameters.** The learning rate, discount, network size and the
  0.5 neighbour weight were chosen once and never tuned or varied; a
  sensitivity study is future work.
- **Many comparisons.** The rubric makes 76 checks without a
  multiple-comparison correction, so a few borderline passes or fails may be
  chance.
- **Coordination used tabular agents.** Next: one DQN per intersection with the
  neighbor features as extra inputs, on a stretch of Great East Road with
  several signals.

---

## 12. Reproduce

Tested with Python 3.14.7 on Windows 11. SUMO installs through pip; no separate
install is needed, and `requirements.txt` pins the versions the results were
produced with.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

```bash
python -m pytest tests                   # 27 tests; the SUMO ones skip if SUMO is missing
python experiments/semma.py              # data study: bins and scales (under a minute)
python experiments/webster.py            # saturation flows and Webster plans (about 1 min)
python experiments/train_all.py          # 10 learners x 5 runs x 900 episodes, in parallel
python experiments/evaluate.py           # results/RESULTS.md (about 5 min in parallel)
python experiments/make_figures.py       # results/figures/
```

The Part 2 networks are already in `networks/`. To rebuild them from the
node/edge definitions and the committed OpenStreetMap extract:

```bash
python networks/build_networks.py
```

One training run is a single call, e.g.
`python experiments/train.py --agent dqn --scenario manda_hill --run 0`.
`train_all.py` runs all 50 in parallel (one process each; about 1 h 45 min on 14
of 16 CPU threads). Everything is seeded, so the numbers should match
`results/` exactly with the pinned versions.

---

## 13. References

- Watkins, C. and Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279-292.
- Tan, M. (1993). Multi-agent reinforcement learning: independent vs. cooperative agents. *ICML*.
- Efron, B. and Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
- van Hasselt, H., Guez, A. and Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
- Lopez, P. A. et al. (2018). Microscopic traffic simulation using SUMO. *IEEE ITSC*.
- Wei, H., Zheng, G., Yao, H. and Li, Z. (2018). IntelliLight: a reinforcement learning approach for intelligent traffic light control. *KDD*.
- Chu, T., Wang, J., Codeca, L. and Li, Z. (2019). Multi-agent deep reinforcement learning for large-scale traffic signal control. *IEEE Transactions on Intelligent Transportation Systems*, 21(3).
- Wei, H., Zheng, G., Gayah, V. and Li, Z. (2019). A survey on traffic signal control methods. arXiv:1904.08117.
- Sutton, R. and Barto, A. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
- SAS Institute. SEMMA data mining methodology.
- Varaiya, P. (2013). Max pressure control of a network of signalized intersections. *Transportation Research Part C*, 36, 177-195.
- Webster, F. V. (1958). *Traffic Signal Settings*. Road Research Technical Paper 39, HMSO, London.
- Transportation Research Board (2022). *Highway Capacity Manual*, 7th ed., Chapter 31, Signalized Intersections: Supplemental (field measurement of saturation flow).
- Henderson, P. et al. (2018). Deep reinforcement learning that matters. *AAAI*.
- Lusaka Times (23 April 2021). LCC "switches off" robots at Great East and Lufubu roads.
- Choongo, B. (2020). *Quality of public transport service in the city of Lusaka: a case study of the minibus service on the Great East Road*. University of Zambia (quotes 31,000 veh/day on Great East Road in 2009).
- Ng'andu, L. (2024). *An assessment of the Lusaka Decongestion Project (LDP): a case study of Great East Road*. MSc dissertation, University of Zambia (2022 video count and LCC peak periods).
- OpenStreetMap contributors. Map data for the Great East Road / Manda Hill junction, ODbL. https://www.openstreetmap.org/copyright

Built on the RoadwayVR SUMO tutorial (MIT); see [NOTICE.md](NOTICE.md). Map data © OpenStreetMap contributors (ODbL). Licensed under MIT.
