from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.generator import generate_dataset
from rl.agent import QLearningAgent
from rl.environment import ExpertVerificationEnv
from rl.trainer import train

if __name__ == "__main__":
    dataset = generate_dataset(n_fraud=500, n_genuine=500)
    env = ExpertVerificationEnv(dataset)
    agent = QLearningAgent()
    metrics = train(agent, env, n_episodes=5000, verbose_every=500)
    print("Training complete. Q-table saved to models/q_table.pkl")
    print(f"Final average reward (last 100): {sum(metrics['episode_rewards'][-100:]) / 100:.3f}")
