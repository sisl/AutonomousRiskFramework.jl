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
    {
        'id': 'OBSTACLE',
        'distance': {'mean': 0, 'std': 1, 'upper': 10, 'lower': -10},
    },
    # {
    #     'id': 'COLLISION',
    #     'normal_impulse': {'mean': 0, 'std': 1, 'upper': 10, 'lower': -10},
    # },
]
weather = {
    'cloudiness': 0,
    'precipitation': 0,
    'precipitation_deposits': 0,
    'wind_intensity': 0,
    'sun_azimuth_angle': 0,
    'sun_altitude_angle': 70,
    'fog_density': 0,
    'fog_distance': 0,
    'wetness': 0,
}
for seed in [0, 92]:
    for scenario_type in ["Scenario2", "Scenario4"]:
        env = gym.make('adv-carla-v0', sensors=sensors, seed=seed, scenario_type=scenario_type, weather=weather)
        obs = env.reset()
        t = r = 0
        done = False
        σ = 0.00001 # noise standard deviation (1.11 meters)
        while not done:
            t += 1
            action = np.array([2*σ, 2*σ, 0, 0]) # lat/lon/alt/distance
            obs, reward, done, info = env.step(action)
            r += reward
            env.render()
        env.close()
        print("Total reward: ", r)
