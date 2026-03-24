from __future__ import annotations

import numpy as np


def train(agent, env, n_episodes: int = 5000, verbose_every: int = 500):
    metrics = {
        "episode_rewards": [],
        "fraud_caught": [],
        "false_positives": [],
    }

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        info = {"action_taken": "PASS", "true_label": "GENUINE"}

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

        agent.decay_epsilon()
        metrics["episode_rewards"].append(total_reward)

        metrics["fraud_caught"].append(
            1 if (info["action_taken"] == "FLAG" and info["true_label"] == "FRAUD") else 0
        )
        metrics["false_positives"].append(
            1 if (info["action_taken"] == "FLAG" and info["true_label"] == "GENUINE") else 0
        )

        if verbose_every and (episode + 1) % verbose_every == 0:
            recent = slice(-verbose_every, None)
            print(
                f"Episode {episode + 1:5d} | "
                f"AvgReward: {np.mean(metrics['episode_rewards'][recent]):+.3f} | "
                f"FraudCaught%: {np.mean(metrics['fraud_caught'][recent]) * 100:.1f} | "
                f"FalsePos%: {np.mean(metrics['false_positives'][recent]) * 100:.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    agent.save()
    return metrics
