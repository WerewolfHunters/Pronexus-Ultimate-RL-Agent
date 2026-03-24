from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ExpertVerificationEnv(gym.Env):
    """Custom environment for PASS / FLAG / FOLLOW_UP decisions."""

    metadata = {"render_modes": []}

    ACTION_PASS = 0
    ACTION_FLAG = 1
    ACTION_FOLLOW_UP = 2

    def __init__(self, candidate_data: list[dict], max_questions: int = 5, max_follow_ups: int = 2):
        super().__init__()
        if not candidate_data:
            raise ValueError("candidate_data cannot be empty")

        self.candidate_data = candidate_data
        self.max_questions = max_questions
        self.max_follow_ups = max_follow_ups

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32)

        self.current_candidate: dict[str, Any] | None = None
        self.question_index = 0
        self.follow_ups_used = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        idx = int(self.np_random.integers(0, len(self.candidate_data)))
        self.current_candidate = self.candidate_data[idx]
        self.question_index = 0
        self.follow_ups_used = 0
        return self._get_observation(), {}

    def _current_question(self) -> dict:
        if self.current_candidate is None:
            raise RuntimeError("Environment not reset.")
        qs = self.current_candidate["questions"]
        safe_idx = min(self.question_index, len(qs) - 1)
        return qs[safe_idx]

    def _get_observation(self) -> np.ndarray:
        q = self._current_question()
        s = q["signals"]

        q_idx_norm = min(self.question_index / max(self.max_questions - 1, 1), 1.0)
        follow_norm = min(self.follow_ups_used / max(self.max_follow_ups, 1), 1.0)

        obs = np.array(
            [
                float(q["t1_fired"]),
                float(q["t2_fired"]),
                float(s["S1_hedging"]),
                float(s["S2_failure_absence"]),
                float(s["S3_temporal_vagueness"]),
                float(s["S4_tribal_vocab"]),
                float(s["S5_tool_polarity"]),
                float(s["S6_length_uniformity"]),
                float(s["S7_structural_symmetry"]),
                float(s["S8_depth_breadth"]),
                float(q["fps"]),
                float(q_idx_norm),
                float(follow_norm),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int):
        if self.current_candidate is None:
            raise RuntimeError("Environment not reset.")

        label = self.current_candidate["label"]
        done = False
        reward = 0.0

        if action == self.ACTION_FOLLOW_UP:
            if self.follow_ups_used < self.max_follow_ups:
                self.follow_ups_used += 1
                self.question_index = min(self.question_index + 1, self.max_questions - 1)
                reward = -0.1
            else:
                reward = -0.3

            if self.question_index >= self.max_questions - 1:
                done = True

        elif action == self.ACTION_FLAG:
            done = True
            reward = 1.0 if label == "FRAUD" else -2.0

        else:  # ACTION_PASS
            done = True
            reward = 0.5 if label == "GENUINE" else -1.0

        obs = self._get_observation()
        info = {
            "action_taken": {0: "PASS", 1: "FLAG", 2: "FOLLOW_UP"}[int(action)],
            "true_label": label,
            "question_index": self.question_index,
            "follow_ups_used": self.follow_ups_used,
        }

        return obs, float(reward), done, False, info
