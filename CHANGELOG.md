# Changelog

## 1.0.0 (2026-07-20)

- Single intersection and two-intersection corridor networks with random, shifting demand
- Shared SUMO environment: 1 s steps, minimum and maximum green, asynchronous agent decisions
- Baselines: fixed-time, actuated, random
- SEMMA data study that sets the state bins and input scales (and found the detector coverage bug)
- Tabular Q-learning, NumPy DQN (replay, target network, Double DQN target), coordinated multi-agent Q-learning
- Held-out evaluation with paired bootstrap confidence intervals, Q-value probes and a pass/fail rubric
