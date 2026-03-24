from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


class QLearningAgent:
    """Sparse Q-learning agent for continuous observation vectors."""

    def __init__(
        self,
        n_actions: int = 3,
        n_bins: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_table: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )

    def discretize(self, obs: np.ndarray) -> tuple[int, ...]:
        clipped = np.clip(obs, 0.0, 1.0)
        bins = np.minimum((clipped * self.n_bins).astype(int), self.n_bins - 1)
        return tuple(int(v) for v in bins.tolist())

    def choose_action(self, obs: np.ndarray) -> int:
        state = self.discretize(obs)
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        state = self.discretize(obs)
        next_state = self.discretize(next_obs)

        current_q = self.q_table[state][action]
        next_max = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.gamma * next_max

        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str = "models/q_table.pkl") -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        serializable_q = {k: v.tolist() for k, v in self.q_table.items()}
        with out.open("wb") as f:
            pickle.dump({"q_table": serializable_q, "epsilon": self.epsilon}, f)
        return out

    def load(self, path: str = "models/q_table.pkl") -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        loaded = data.get("q_table", {})
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
        for k, v in loaded.items():
            self.q_table[tuple(k)] = np.array(v, dtype=np.float32)
        self.epsilon = float(data.get("epsilon", self.epsilon))
