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
    {
        'id': 'CAMERA',
        'dynamic_noise_std': {'mean': 0, 'std': 0.001, 'upper': 1, 'lower': 0},
        'exposure_compensation': {'mean': 0, 'std': 0.5, 'upper': 1, 'lower': -1},
    },
    # {
    #     'id': 'COLLISION',
    #     'normal_impulse': {'mean': 0, 'std': 1, 'upper': 10, 'lower': -10},
    # },
]
# weather = {
#     'cloudiness': 0,
#     'precipitation': 0,
#     'precipitation_deposits': 0,
#     'wind_intensity': 0,
#     'sun_azimuth_angle': 0,
#     'sun_altitude_angle': 70,
#     'fog_density': 0,
#     'fog_distance': 0,
#     'wetness': 0,
# }
weather = "Random"
seeds = [3] # 228, 92, 103 (Random), 1 (Random Scenario4), 3 (Random Scenario2)
scenarios = ["Scenario2"] # ["scenario2", Scenario4"]:
for seed in seeds:
    for scenario_type in scenarios:
        env = gym.make('adv-carla-v0', sensors=sensors, seed=seed, scenario_type=scenario_type, weather=weather)
        obs = env.reset()
        t = r = 0
        done = False
        σ = 0.00001 # noise standard deviation (1.11 meters)
        while not done:
            t += 1
            if len(sensors) == 2:
                action = np.array([2*σ, 2*σ, 0, 0]) # lat/lon/alt/distance
            elif len(sensors) == 3:
                action = np.array([2*σ, 2*σ, 0, 0, 0.05, -0.5]) # lat/lon/alt/distance/dynamic_noise_std/exposure_compensation
            obs, reward, done, info = env.step(action)
            r += reward
            env.render()
        env.close()
        print("Total reward: ", r)
