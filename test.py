import gym
import numpy as np


env = gym.make("CarRacing-v2", render_mode="human")


observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    if done or truncated:
        observation, info = env.reset()

env.close()
