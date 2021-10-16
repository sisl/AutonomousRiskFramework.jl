import numpy as np
import gym
import adv_carla

env = gym.make('adv-carla-v0') # AdversarialCARLAEnv()
obs = env.reset()
t = r = 0
done = False
while not done:
    t += 1
    action = np.array([2.868206, -4.096098])
    obs, reward, done, info = env.step(action)
    r += reward
env.close()
print("Reward: ", r)
