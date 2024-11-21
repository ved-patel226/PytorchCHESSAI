import gym
import numpy as np
from model import Agent
from tqdm import tqdm


def main() -> None:
    env = gym.make("CarRacing-v2", render_mode="human")

    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=5,
        eps_end=0.01,
        input_dims=(96, 96, 3),
        lr=0.003,
        max_steps=100_000,
    )

    agent.load_model("checkpoints/models/racer_100.pth")

    n_episodes = 10
    scores = []

    for episode in tqdm(range(n_episodes), desc="Test Episodes"):
        done = False
        score = 0
        observation, info = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, extra_info = env.step(action)
            score += reward
            observation = observation_

        scores.append(score)
        print(f"Episode {episode+1}/{n_episodes}, Score: {score}")

    avg_score = np.mean(scores)
    print(f"Average Score over {n_episodes} episodes: {avg_score:.2f}")

    env.close()


if __name__ == "__main__":
    main()
