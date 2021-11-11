import numpy as np
import gym
import adv_carla

# maps to the AutonomousAgent.sensors() return vector.
# TODO: default these adversarial sensor parameters!
sensors = [
    {
        'id': 'GPS',
        'lat': {'mean': 0, 'std': 0.0001, 'upper': 10, 'lower': -10},
        'lon': {'mean': 0, 'std': 0.0001, 'upper': 10, 'lower': -10},
        'alt': {'mean': 0, 'std': 0.00000001, 'upper': 0.0000001, 'lower': 0},
    },
]
env = gym.make('adv-carla-v0', sensors=sensors)
obs = env.reset()
t = r = 0
done = False
σ = 0.0001 # noise standard deviation
while not done:
    t += 1
    action = np.array([0, 0, 0]) # σ*np.random.rand(3) # lat/lon/alt
    # action = σ*np.random.rand(3) # lat/lon/alt
    obs, reward, done, info = env.step(action)
    r += reward
    env.render()
env.close()
print("Total reward: ", r)
