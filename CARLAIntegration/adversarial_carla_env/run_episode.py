import numpy as np
from adversarial_carla_env import AdversarialCARLAEnv

env = AdversarialCARLAEnv()
obs = env.reset()
t = r = 0
done = False
while not done:
    t += 1
    action = np.array([2.868206, -4.096098])
    obs, reward, done, info = env.step(action)
    r += reward

print("Reward: ", r)
