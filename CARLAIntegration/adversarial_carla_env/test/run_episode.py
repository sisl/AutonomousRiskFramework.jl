import numpy as np
import gym
import adv_carla

env = gym.make('adv-carla-v0') # AdversarialCARLAEnv()
obs = env.reset()
t = r = 0
done = False
σ = 1 # noise variance
while not done:
    t += 1
    action = np.array([0, 0]) # σ*np.random.rand(2) # xy in meters
    obs, reward, done, info = env.step(action)
    r += reward
    env.render()
env.close()
print("Total reward: ", r)
