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
    # {
    #     'id': 'OBSTACLE',
    #     'distance': {'mean': 0, 'std': 1, 'upper': 10, 'lower': -10},
    # },
    # {
    #     'id': 'rgb',
    #     'dynamic_noise_std': {'mean': 0, 'std': 0.001, 'upper': 1, 'lower': 0},
    #     'exposure_compensation': {'mean': 0, 'std': 0.5, 'upper': 1, 'lower': -1},
    # },
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
def main():
    # weather = "Random"
    weather = {
        'cloudiness': 0.0,
        'precipitation_deposits': 0.0,
        'wetness': 66.6667,
        'fog_density': 100.0,
        'wind_intensity': 66.6667,
        'precipitation': 33.3333,
        'sun_altitude_angle': -30.0,
        'sun_azimuth_angle': 120.0,
        'fog_distance': 100.0
    }
    seeds = [3] # 228, 92, 103 (Random), 1 (Random Scenario4), 3 (Random Scenario2)
    scenarios = ["Scenario2"] # ["scenario2", Scenario4"]:
    # agent = "C:/Users/mossr/Code/sisl/ast/Allstate/AutonomousRiskFramework/CARLAIntegration/neat/leaderboard/team_code/neat_agent.py"
    agent = None
    for seed in seeds:
        for scenario_type in scenarios:
            env = gym.make('adv-carla-v0', sensors=sensors, seed=seed, scenario_type=scenario_type, weather=weather, port=3000, agent=agent)
            obs = env.reset()
            t = r = 0
            done = False
            σ = 0.00001 # noise standard deviation (1.11 meters)
            while not done:
                t += 1
                if len(sensors) == 1:
                    action = np.array([2*σ, 2*σ, 0]) # lat/lon/alt
                elif len(sensors) == 2:
                    exposure = 2*np.random.rand() - 1 # 0.5
                    action = np.array([2*σ, 2*σ, 0, 0.05, exposure]) # lat/lon/alt/dynamic_noise_std/exposure_compensation
                elif len(sensors) == 3:
                    action = np.array([2*σ, 2*σ, 0, 0, 0.05, -0.5]) # lat/lon/alt/distance/dynamic_noise_std/exposure_compensation
                elif len(sensors) == 1:
                    action = np.array([2*σ, 2*σ, 0]) # lat/lon/alt
                obs, reward, done, info = env.step(action)
                r += reward
                env.render()
            env.close()
            print("Total reward: ", r)

main()