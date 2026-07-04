# Data study: SEMMA

Before training any agent we studied the data the agents would see. We used
**SEMMA** (Sample, Explore, Modify, Model, Assess), the data mining process
from SAS Institute. We chose SEMMA over CRISP-DM because the business
question is already fixed (reduce waiting at signals, see the README), and
SEMMA focuses on the technical data-to-model steps. For readers who know
CRISP-DM: Sample and Explore are "Data Understanding", Modify is
"Data Preparation", Model and Assess are "Modeling" and "Evaluation".

Code: [`experiments/semma.py`](../experiments/semma.py). Output: [`results/semma/`](../results/semma/).

## 1. Sample

The data is what the lane-area detectors (E2 detectors, the "cameras") report
every second: vehicles per lane, stopped vehicles, which approach has green.

- 3 scenarios (single intersection, corridor normal, corridor heavy)
- 3 controllers that need no training: fixed-time, actuated, random switching
- 3 seeds each (900-902, never used for training or evaluation)
- 1,200 s per run, one row per intersection per second

That gives **54,000 rows**, saved as `detector_samples.csv.gz`. Using three
different controllers matters: an agent will visit states that fixed-time
never produces (for example very long or very short greens), so the sample
has to cover them.

## 2. Explore

| Finding | Evidence | Figure |
|---|---|---|
| Counts are right-skewed. Most seconds see a few cars, a long tail sees many. | Single: median 4, 90th pct 12, max 34 vehicles per approach | `explore_approach_counts.png` |
| Demand really shifts over the episode, so a fixed plan cannot fit all of it. | EB peaks around 15 in the first third, SB peaks around 15 in the last third | `explore_demand_over_time.png` |
| Lanes of one approach move together, and the two approaches move opposite. | Mean correlation 0.78 within an approach, -0.37 across | `explore_lane_correlation.png` |
| The red approach builds up and the green one drains. | Median red-side count about 2x the green side | `explore_queue_by_phase.png` |
| **The detectors were missing part of the queue.** | See below | |

**Data quality bug found in Explore.** We compared the queue seen by the
detectors with the true number of stopped vehicles on each approach. On the
corridor the detectors saw only **37%** of the queue (fixed-time run, seed 900).
The cause: every detector ended 10 to 16 m before the stop line, so the first
two cars in every queue were invisible, and those are exactly the cars a signal
decision depends on. We moved every detector to end at the stop line. On the
same run coverage became **108%** on the corridor and 110% on the single
intersection (up from 94%). Over the whole 54,000-row sample it is 113%
(corridor) and 116% (single). It sits slightly above 100% because a detector
counts a car as stopped below 1.39 m/s and an edge only below 0.1 m/s.
This fix was made before any training.

## 3. Modify

The exploration decides how raw counts become agent inputs. The results are
written to `results/semma/state_config.json` and read by the agents at run time.

| Decision | Choice | Why (from Explore) |
|---|---|---|
| Tabular state uses the approach total, not each lane | EB total, SB total | Lanes within an approach are strongly correlated (0.78), so six lane bins would multiply the table size without adding much information |
| Q-learning bin edges | 2, 4, 7, 11 vehicles (5 bins) | The 25th, 50th, 75th and 90th percentiles of non-empty counts. Every bin is visited often, and the long tail gets its own bin instead of being spread across many rare ones |
| Green age bins | under 20 s, 20-40 s, over 40 s | Between the 10 s minimum green and the 42 s fixed-time green |
| DQN input scale | lane count / 8 | 99th percentile of a lane count, so inputs sit roughly in [0, 1] |
| DQN reward scale | queue / 19 | 99th percentile of the queue, so TD targets stay near [-1, 0] |

The DQN keeps all six lane counts: a neural network can use the lane detail
that the table cannot afford.

## 4. Model

Three learners, all described in the README:

- tabular **Q-learning** (Watkins and Dayan, 1992)
- **DQN** with experience replay, target network (Mnih et al., 2015) and the Double DQN target (van Hasselt et al., 2016)
- **multi-agent Q-learning**, independent (Tan, 1993) and coordinated with neighbor state and shared reward (in the spirit of Chu et al., 2019)

## 5. Assess

Every controller is scored on the same 10 held-out traffic seeds, with
paired bootstrap confidence intervals and a pass/fail rubric. See
[`results/RESULTS.md`](../results/RESULTS.md) and the README.
