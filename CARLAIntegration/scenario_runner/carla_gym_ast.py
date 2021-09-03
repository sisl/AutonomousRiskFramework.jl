import gym
import numpy as np
import copy
import random
from gym import spaces
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

default_params = {
  'endtime': 200,
  'reward_bonus': 100,
  'discount': 1.0,
  'max_past_step': 3,
  'lower_disturbance': [-10, -10],
  'upper_disturbance': [10, 10],
  'var_disturbance': [1, 1],
  'mean_disturbance': [0, 0],
  'lower_actor_state': [-100, -100, -100, -100, 0],   # x_topright, y_topright, x_bottomleft, y_bottomleft, status
  'upper_actor_state': [100, 100, 100, 100, 1],
}

class CARLAEnv(gym.Env):
  """Custom Environment for CARLA with ScenarioRunner"""

  def __init__(self, step_fn, reset_fn, params=default_params):
    super(CARLAEnv, self).__init__()
    # parameters
    self.endtime = params['endtime']
    self.reward_bonus = params['reward_bonus']
    self.discount = params['discount']
    self.max_past_step = params['max_past_step']
    self.var_disturbance = np.array(params['var_disturbance'])
    self.mean_disturbance = np.array(params['mean_disturbance'])

    # Functions
    self._step = step_fn
    self._reset = reset_fn

    obs = self.reset(retdict=True)

    self.actor_keys = list(obs.keys())  # Ensure actor is matched to the right key for observations

    # action/observation spaces
    assert len(params['lower_disturbance']) == len(params['upper_disturbance'])
    self.action_space = spaces.Box(np.array(params['lower_disturbance']), np.array(params['upper_disturbance']), dtype=np.float32)

    assert len(params['lower_actor_state']) == len(params['upper_actor_state'])

    # observation_space_dict = {}
    # for key in self.actor_keys:
    #   observation_space_dict[key] = spaces.Box(
    #     np.array(params['lower_actor_state']), 
    #     np.array(params['upper_actor_state']), dtype=np.float32)

    # self.observation_space = spaces.Dict(observation_space_dict)

    self.observation_space = spaces.Box(
        np.array(params['lower_actor_state']*len(self.actor_keys)), 
        np.array(params['upper_actor_state']*len(self.actor_keys)), dtype=np.float32)

  def step(self, action):
    self._action = action
    self._actions.append(action)

    # Execute one time step within the environment. 
    running, collision, distance = self._step(action)
    
    if running == 0 or collision:
      self._done = True

    if collision:
      self._failed_scenario = True

    self._collision = collision

    self._distances.append(distance)

    # Calculate the reward for this step
    self._reward = self._get_reward(self._action, self._done, self._failed_scenario, collision, self._distances)

    # Append actors polygon list
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)
    while len(self.walker_polygons) > self.max_past_step:
      self.walker_polygons.pop(0)

    # state information
    info = {
      'vehicles': self.vehicle_polygons,
      'walkers': self.walker_polygons
    }

    # Update timesteps
    self._timestep += 1

    return (self._get_obs(), self._reward, self._done, copy.deepcopy(info))

  def _get_actor_polygons(self, filt):
    """Get the bounding box polygon of actors.
    Args:
      filt: the filter indicating what type of actors we'll look at.
    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
    actor_poly_dict={}
    for actor in self.world.get_actors().filter(filt):
      # Get x, y and yaw of the actor
      trans=actor.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=actor.bounding_box
      l=bb.extent.x
      w=bb.extent.y
      # Get bounding box polygon in the actor's local coordinate
      poly_local=np.array([[l,w],[-l,-w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],2,axis=0)
      actor_poly_dict[actor.id]=poly
    return actor_poly_dict

  def reset(self, retdict=False):
    # Reset the state of the environment to an initial state
    self._reset()
    self.world = CarlaDataProvider.get_world()

    self._done = False
    self._reward = 0.0
    self._timestep = 0
    self._action = None
    self._actions = []
    self._first_step = True
    self._distances = []
    self._collision = False
    self._failed_scenario = False

    # Get actors polygon list
    self.vehicle_polygons = []
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    self.walker_polygons = []
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)

    return self._get_obs(retdict=retdict)
  
  def _get_obs(self, retdict=False):
    obs = {}
    
    if hasattr(self, 'actor_keys'):
      # Default values
      for key in self.actor_keys:
        obs[key] = np.zeros(5, dtype=np.float32)


    # for _time, vehicle_poly_dict in enumerate(self.vehicle_polygons):
    vehicle_poly_dict = self.vehicle_polygons[-1]
    for i, (key, value) in enumerate(vehicle_poly_dict.items()):
      obs['veh_'+str(i)] = np.ones(5, dtype=np.float32)
      obs['veh_'+str(i)][:4] = value.flatten().astype(np.float32)
    
    # for _time, walker_poly_dict in enumerate(self.walker_polygons):
    walker_poly_dict = self.walker_polygons[-1]
    for i, (key, value) in enumerate(walker_poly_dict.items()):
      obs['walker_'+str(i)] = np.ones(5, dtype=np.float32)
      obs['walker_'+str(i)][:4] = value.flatten().astype(np.float32)

    if retdict:
      return obs

    return normalize_observations(np.concatenate([v for v in obs.values()], axis=0))

  def _get_reward(self, action, done, failed_scenario, collision, distances):
    # TODO: scale to be reasonable sized (Within [-1, 1])
    # reward = -(self.mahalanobis_d(action)/self.mahalanobis_d([5, 5]) + np.clip(distances[-1], 0, 10)/10)
    reward = -self.mahalanobis_d(action)/self.mahalanobis_d([10, 10])
    if collision:
      reward += 100

    return reward

  def mahalanobis_d(self, action):
    # Mean action is 0
    action = np.array(action)
    mean = self.mean_disturbance
    # Assemble the diagonal covariance matrix
    cov = self.var_disturbance
    big_cov = np.diagflat(cov)

    # subtract the mean from our actions
    dif = np.copy(action)
    dif -= mean

    # calculate the Mahalanobis distance
    dist = np.dot(np.dot(dif.T, np.linalg.inv(big_cov)), dif)

    return np.sqrt(dist)

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    print("Timestep: ", self._timestep)
    print("Action: ", self._action)
    print("Reward: ", self._reward)
    print("Collision: ", self._collision)
    print("Failed: ", self._failed_scenario)

def normalize_observations(obs):
  mean = np.array([83.78031, 12.687019, 84.24346, 16.653917, 0.5, 64.62255, -2.442448, 64.96491, -3.9550319, 0.5])
  std = np.array([13.913469, 14.930986, 10.969871, 15.964834, 0.5, 18.97383, 4.713275, 19.074358, 4.8009253, 0.5])
  return np.divide(obs - mean, std)