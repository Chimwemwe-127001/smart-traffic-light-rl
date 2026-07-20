"""Deep Q-Network in plain NumPy (Mnih et al. 2015, with the Double DQN target of van Hasselt et al. 2016).

A small neural network replaces the Q-table, so the agent can use the raw
per-lane counts instead of coarse bins and generalize between similar states.
Three standard ingredients make it stable:

    experience replay  - learn from random mini-batches of past transitions,
                         not from the latest (highly correlated) one
    target network     - a frozen copy of the network computes the TD target,
                         synced every SYNC_EVERY updates
    Double DQN target  - the online network picks the next action, the target
                         network scores it: y = r + gamma * Q_target(s', argmax_a Q(s', a))

Written in NumPy on purpose: every line of the forward pass, backprop and Adam
update is visible, and there is no framework to install.
"""
import numpy as np

from traffic_rl.baselines import queue_reward
from traffic_rl.state import feature_vector, load_config

N_FEATURES = 9          # 6 lane counts + 2 green one-hot + green age (single intersection)
HIDDEN = 32


class MLP:
    """Two hidden ReLU layers, linear output, Huber loss, Adam optimizer."""

    def __init__(self, n_in=N_FEATURES, n_hidden=HIDDEN, n_out=2, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        he = lambda fan_in, shape: rng.normal(0, np.sqrt(2 / fan_in), shape)
        self.params = [he(n_in, (n_in, n_hidden)), np.zeros(n_hidden),
                       he(n_hidden, (n_hidden, n_hidden)), np.zeros(n_hidden),
                       he(n_hidden, (n_hidden, n_out)), np.zeros(n_out)]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.lr = lr

    def forward(self, x):
        W1, b1, W2, b2, W3, b3 = self.params
        z1 = x @ W1 + b1; a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2; a2 = np.maximum(0, z2)
        return a2 @ W3 + b3, (x, z1, a1, z2, a2)

    def predict(self, x):
        return self.forward(np.atleast_2d(x))[0]

    def train_step(self, x, actions, targets):
        """One Adam step on Huber loss, only for the actions actually taken. Returns the loss."""
        out, (x, z1, a1, z2, a2) = self.forward(x)
        idx = np.arange(len(actions))
        err = out[idx, actions] - targets
        loss = float(np.mean(np.where(np.abs(err) < 1, 0.5 * err ** 2, np.abs(err) - 0.5)))
        d3 = np.zeros_like(out)
        d3[idx, actions] = np.clip(err, -1, 1) / len(actions)       # Huber gradient
        W1, b1, W2, b2, W3, b3 = self.params
        d2 = (d3 @ W3.T) * (z2 > 0)
        d1 = (d2 @ W2.T) * (z1 > 0)
        grads = [x.T @ d1, d1.sum(0), a1.T @ d2, d2.sum(0), a2.T @ d3, d3.sum(0)]

        self.t += 1
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[:] = b1_ * m + (1 - b1_) * g
            v[:] = b2_ * v + (1 - b2_) * g * g
            p -= self.lr * (m / (1 - b1_ ** self.t)) / (np.sqrt(v / (1 - b2_ ** self.t)) + eps)
        return loss

    def copy_from(self, other):
        for p, q in zip(self.params, other.params):
            p[:] = q


class ReplayBuffer:
    """Fixed-size circular buffer of (s, a, r, s') transitions."""

    def __init__(self, capacity=20_000, n_features=N_FEATURES, seed=0):
        self.s = np.zeros((capacity, n_features))
        self.a = np.zeros(capacity, dtype=int)
        self.r = np.zeros(capacity)
        self.s2 = np.zeros((capacity, n_features))
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2):
        self.s[self.pos], self.a[self.pos], self.r[self.pos], self.s2[self.pos] = s, a, r, s2
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        i = self.rng.integers(0, self.size, n)
        return self.s[i], self.a[i], self.r[i], self.s2[i]


class DQN:
    program = 'agent'
    BATCH = 64
    WARMUP = 500
    SYNC_EVERY = 250

    def __init__(self, gamma=0.9, epsilon=1.0, lr=1e-3, seed=0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.cfg = load_config()
        self.net = MLP(lr=lr, seed=seed)
        self.target = MLP(seed=seed)
        self.target.copy_from(self.net)
        self.buffer = ReplayBuffer(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def features(self, tls, obs):
        return feature_vector(obs[tls], self.cfg)

    def reward(self, tls, queues):
        return queue_reward(tls, queues)

    def act(self, tls, obs, explore=False):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        return int(np.argmax(self.net.predict(self.features(tls, obs))[0]))

    def learn(self, tls, obs, action, queues, next_obs):
        r = self.reward(tls, queues) / self.cfg['queue_scale']      # keep targets near [-1, 0]
        self.buffer.add(self.features(tls, obs), action, r, self.features(tls, next_obs))
        if self.buffer.size < self.WARMUP:
            return None
        s, a, r, s2 = self.buffer.sample(self.BATCH)
        best_next = np.argmax(self.net.predict(s2), axis=1)                          # online net chooses
        y = r + self.gamma * self.target.predict(s2)[np.arange(self.BATCH), best_next]  # target net scores
        loss = self.net.train_step(s, a, y)
        self.updates += 1
        if self.updates % self.SYNC_EVERY == 0:
            self.target.copy_from(self.net)
        return loss

    def save(self, path):
        np.savez(path, *self.net.params)

    def load(self, path):
        data = np.load(path)
        for i, p in enumerate(self.net.params):
            p[:] = data[f'arr_{i}']
        self.target.copy_from(self.net)
        return self
