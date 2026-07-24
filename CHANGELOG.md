# Changelog

## 1.0.1 (2026-07-24)

- Fix: validation seeds (now 700-702) no longer overlap the corridor training seeds (0-599); all agents retrained
- Fix: validation runs on a copy of the agent, so it no longer adds rows to the Q-table being trained
- Evaluation reports each learner against actuated control with paired 95% confidence intervals
- Fixed-time and actuated controllers show "n/a" switches instead of 0
- New SUMO integration tests: greens last 10 to 60 s, yellows 3 s, fixed-time runs its 42 s plan, runs are reproducible
- Docs: corrected seed ranges and setup commands

## 1.0.0 (2026-07-20)

- Single intersection and two-intersection corridor networks with random, shifting demand
- Shared SUMO environment: 1 s steps, minimum and maximum green, asynchronous agent decisions
- Baselines: fixed-time, actuated, random
- SEMMA data study that sets the state bins and input scales (and found the detector coverage bug)
- Tabular Q-learning, NumPy DQN (replay, target network, Double DQN target), coordinated multi-agent Q-learning
- Held-out evaluation with paired bootstrap confidence intervals, Q-value probes and a pass/fail rubric
