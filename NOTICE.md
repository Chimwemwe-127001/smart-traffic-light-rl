# Notice and attribution

This project started from the companion code of the YouTube playlist
"SUMO Traffic Simulator Tutorial" by Dr. Ahmad Mohammadi (RoadwayVR):

- Source: https://github.com/RoadwayVR/SUMO-Traffic-Simulator-Tutorial
- License: MIT

What came from the tutorial: the idea of a single-intersection keep/switch
agent with a queue-based reward, the lane-area detector naming
(`Node1_2_EB_0` ... `Node2_7_SB_2`), and the original tabular Q-learning and
Keras DQN scripts (`traci6.QL.py`, `traci7.DQL.py`), which this code no longer
contains but was first adapted from.

Everything else was built for this project: the networks and demand files, the
shared environment with asynchronous decisions and minimum green, the NumPy DQN
with replay, target network and Double DQN target, the coordinated multi-agent
learner, the SEMMA data study, the evaluation protocol and all results.
