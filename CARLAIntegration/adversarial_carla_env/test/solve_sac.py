import os
import pickle
import numpy as np
import gym
import adv_carla
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback

max_step = 200
sensors = [
    {
        'id': 'GPS',
        'lat': {'mean': 0, 'std': 0.0001, 'upper': 10, 'lower': -10},
        'lon': {'mean': 0, 'std': 0.0001, 'upper': 10, 'lower': -10},
        'alt': {'mean': 0, 'std': 0.00000001, 'upper': 0.0000001, 'lower': 0},
    },
]
env = gym.make('adv-carla-v0', sensors=sensors)
model = sb3.SAC("MlpPolicy", env, verbose=1)
# model = sb3.SAC.load(os.path.join(os.getcwd(), "checkpoints", "td3_carla_10000_steps"))
model.set_env(env)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./outputs/checkpoints/',
                                         name_prefix='sac_carla_test')
# Turn off if only evaluating
n_episodes = 1
episode_length = 200
total_timesteps = n_episodes*episode_length # 10000
model.learn(total_timesteps=total_timesteps, log_interval=1, callback=checkpoint_callback)

# model.save(os.path.join(os.getcwd(), "variables", "td3_carla"))
dataset_save_path = os.path.join(os.getcwd(), "outputs", "variables", "dataset_test")
if not os.path.exists(dataset_save_path):
    os.makedirs(dataset_save_path)

# _samples =[]
# _dists = []
# _rates = []
# _y = []
# for data in env.dataset:
#     _y.append(data[1])
#     _samples.append(data[0][0])
#     _dists.append(data[0][1])
#     _rates.append(data[0][2])
# pickle.dump( _y, open( os.path.join(dataset_save_path, "y.pkl"), "wb" ) )
# pickle.dump( _samples, open( os.path.join(dataset_save_path, "samples.pkl"), "wb" ) )
# pickle.dump( _rates, open( os.path.join(dataset_save_path, "rates.pkl"), "wb" ) )
# pickle.dump( _dists, open( os.path.join(dataset_save_path, "dists.pkl"), "wb" ) )
env = model.get_env()


# ls_obs = []
ls_action = []
observation = env.reset()
# print(observation, env.observation_space)
# raise
for t in range(max_step):
    # ls_obs.append(observation)
    action, _states = model.predict(observation, deterministic=True)
    ls_action.append(action)
    observation, reward, done, info = env.step(action)
    # observation, reward, done, info = env.step(np.zeros_like(action))   # Zero noise for debugging
    env.render()
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        np.save(os.path.join(os.getcwd(), "variables", "actions_sac_10000_steps"), np.stack(ls_action))
        # ls_obs = np.stack(ls_obs)
        # print(np.mean(ls_obs, axis=0), np.std(ls_obs, axis=0))
        break
env.close()
