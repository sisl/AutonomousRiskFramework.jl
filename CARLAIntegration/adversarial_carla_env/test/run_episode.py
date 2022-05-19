import numpy as np
import gym
import adv_carla
import os

AGENT = "WorldOnRails" # Choose between: ["NEAT", "WorldOnRails", "GNSS"]
USE_GNSS_SENSOR = False or (AGENT == "GNSS")
USE_RANDOM_WEATHER = False

# maps to the AutonomousAgent.sensors() return vector.
sensors = []

if USE_GNSS_SENSOR:
    gnss_sensor_config = {
        'id': 'GPS',
        'lat': {'mean': 0, 'std': 0.0001, 'upper': 10, 'lower': -10},
        'lon': {'mean': 0, 'std': 0.0001, 'upper': 10, 'lower': -10},
        'alt': {'mean': 0, 'std': 0.00000001, 'upper': 0.0000001, 'lower': 0},
    }
    sensors.append(gnss_sensor_config)

if AGENT != "GNSS":
    camera_sensor_config = {
        'id': 'rgb',
        'dynamic_noise_std': {'mean': 0, 'std': 0.001, 'upper': 1, 'lower': 0},
        'exposure_compensation': {'mean': 0, 'std': 0.5, 'upper': 1, 'lower': -1},
    }
    sensors.append(camera_sensor_config)


def main():
    if USE_RANDOM_WEATHER:
        weather = "Random"
    else:
        weather = {
            'cloudiness': 0.0,
            'precipitation_deposits': 0.0,
            'wetness': 66.6667,
            'fog_density': 100.0,
            'wind_intensity': 66.6667,
            'precipitation': 33.3333,
            'sun_altitude_angle': 30.0,
            'sun_azimuth_angle': 120.0,
            'fog_distance': 100.0
        }
    seeds = [3] # 228, 92, 103 (Random), 1 (Random Scenario4), 3 (Random Scenario2)
    scenarios = ["Scenario2"] # ["scenario2", Scenario4"]:
    if AGENT == "NEAT":
        dirname = os.path.dirname(__file__)
        agent = os.path.abspath(os.path.join(dirname, "../../neat/leaderboard/team_code/neat_agent.py"))
        agent_config = os.path.abspath(os.path.join(dirname, "../../neat/model_ckpt/neat"))
    elif AGENT == "WorldOnRails":
        dirname = os.path.dirname(__file__)
        agent = os.path.abspath(os.path.join(dirname, "../../WorldOnRails/autoagents/image_agent.py"))
        agent_config = os.path.abspath(os.path.join(dirname, "../../WorldOnRails/config.yaml"))
    elif AGENT == "GNSS":
        agent = None
        agent_config = None
    for seed in seeds:
        for scenario_type in scenarios:
            env = gym.make('adv-carla-v0', sensors=sensors, seed=seed, scenario_type=scenario_type, weather=weather, port=3000, agent=agent, agent_config=agent_config)
            obs = env.reset()
            t = r = 0
            done = False
            σ = 0.00001 # noise standard deviation (1.11 meters)
            while not done:
                t += 1
                action = np.array([])
                if USE_GNSS_SENSOR:
                    lat = 2*σ
                    lon = 2*σ
                    alt = 0
                    action = np.append(action, [lat, lon, alt])
                if AGENT != "GNSS":
                    exposure = 2*np.random.rand() - 1 # random number between -1 and 1
                    dynamic_noise_std = np.random.rand()/10 # random number between 0 and 0.1
                    action = np.append(action, [dynamic_noise_std, exposure])
                obs, reward, done, info = env.step(action)
                r += reward
                env.render()
            env.close()
            print("Total reward: ", r)

main()