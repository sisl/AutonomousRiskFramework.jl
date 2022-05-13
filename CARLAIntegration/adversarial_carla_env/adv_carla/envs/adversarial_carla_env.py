#!/usr/bin/env python

# Copyright (c) 2021 Stanford University
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This is the adversarial CARLA gym environment, designed to
reformulate the problem as an adversarial MDP and to use
adaptive stress testing (AST) to find likely failures.
"""

import gym
import numpy as np
import random
import copy
from gym import spaces

import traceback
import argparse
import os
import signal
import sys
import time
import warnings
import math

import carla
import py_trees

if 'SCENARIO_RUNNER_ROOT' not in os.environ:
    raise Exception("Please add 'SCENARIO_RUNNER_ROOT' to your environment variables.")
else:
    sys.path.append(os.getenv('SCENARIO_RUNNER_ROOT')) # Add scenario_runner package to import path

from ..ast_scenario_runner import ASTScenarioRunner
from ..generate_random_scenario import *
from .adversarial_sensors import *
from .camera_exposure import *

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from pdb import set_trace as breakpoint # DEBUG. TODO!

from . import utils

# Version of adversarial_carla_env (tied to CARLA and scenario_runner versions)
VERSION = '0.9.11'

# TODO: fan out into kwargs of the init() function
DEFAULT_PARAMS = {
    'endtime': 100,
    'reward_bonus': 100,
    'discount': 1.0,
    'max_past_step': 3,
    'lower_actor_state': [-100, -100, -100, -100, 0],   # x_topright, y_topright, x_bottomleft, y_bottomleft, status
    'upper_actor_state': [100, 100, 100, 100, 1],
}

# Mapping from sensor type => adversarial callback class
ADVERSARIAL_SENSOR_MAPPING = {
    'sensor.other.gnss': {
        'callback': AdvGNSSCallBack,
        'params': ['lat', 'lon', 'alt']
    },
    'sensor.other.obstacle': {
        'callback': AdvObstacleCallBack,
        'params': ['distance']
    },
    'sensor.other.collision': {
        'callback': AdvCollisionCallBack,
        'params': ['normal_impulse']
    },
    'sensor.camera.rgb': {
        'callback': AdvCameraCallBack,
        'params': ['dynamic_noise_std', 'exposure_compensation']
    },
}


class AdversarialCARLAEnv(gym.Env):

    """
    This is the core scenario runner module. It is responsible for
    running (and repeating) a single scenario or a list of scenarios.

    Usage:
    env = gym.make('adv-carla-v0', sensors=sensors)
    obs = env.reset()
    t = r = 0
    done = False
    while not done:
        t += 1
        action = np.array([0])
        obs, reward, done, info = env.step(action)
        r += reward
    env.close()
    """

    # CARLA scenario handler
    scenario_runner = None
    carla_map = None
    spectator_loc = None
    world = None

    # CARLA configuration parameters
    carla_running = False
    follow_ego = True
    save_video = False

    # Gym environment parameters
    block_size = 10

    # Dictionary of adversarial sensors, keyed by the `id` from AutonomousAgent.sensors() return vector.
    adv_sensor_callbacks = {}

    # Parameters associated with each disturbance type.
    disturbance_params = []


    def __init__(self, *, seed=0, scenario_type="Scenario2", weather="Random", agent=None, port=3000, record="recordings", params=DEFAULT_PARAMS, sensors=disturbance_params, no_rendering=False):
        self.hardreset(seed=seed, scenario_type=scenario_type, weather=weather, agent=agent, port=port, record=record, params=params, sensors=sensors, no_rendering=no_rendering)


    def hardreset(self, *, seed=0, scenario_type="Scenario2", weather="Random", agent=None, port=3000, record="recordings", params=DEFAULT_PARAMS, sensors=disturbance_params, no_rendering=False):
        print("Hard resetting...")
        # Scenario/route selections
        dirname = os.path.dirname(__file__)
        example_scenario = False
        if example_scenario:
            route_file = os.path.join(dirname, "../data/test.xml")
            scenario_file = os.path.join(dirname, "../data/test.json")
            self.carla_map = "Town04"
            # route_file = os.path.join(dirname, "../data/routes_ast.xml")
            # scenario_file = os.path.join(dirname, "../data/ast_scenarios.json")
            # self.carla_map = "Town01"
            self.spectator_loc = [80.37, 25.30, 0.0]
        else:
            route_file, scenario_file, self.carla_map, self.spectator_loc = create_random_scenario(seed=seed, scenario_type=scenario_type, weather=weather)

        route_id = 0 # TODO: Can we use this to control the background activity?
        route = [route_file, scenario_file, route_id]
        scenario = None

        # Agent selections
        if agent is None:
            # agent = "E:/CARLA_0.9.13/PythonAPI/scenario_runner/srunner/autoagents/npc_agent.py"
            agent = os.path.join(dirname, "../agents/gnss_agent.py")


        ## Setup ScenarioRunner
        # Setup arguments passed to the ScenarioRunner constructor
        # TODO: How to piggy-back on scenario_runner.py "main()" defaults?
        args = argparse.Namespace(host="127.0.0.1",
                                  port=port,
                                  timeout=600.0, # for both ScenarioRunner and Watchdog (within ScenarioManager)
                                  trafficManagerPort=8000,
                                  trafficManagerSeed=0,
                                  sync=True,
                                  list=False,

                                  scenario=scenario, # TODO: NOTE, --agent used to execute the scenario is currently only compatible with route-based scenarios.
                                  openscenario=None,
                                  openscenarioparams=None,
                                  route=route,

                                  agent=agent,
                                  agentConfig=os.path.join(dirname, "../../../neat/model_ckpt/neat/"),

                                  output=False,
                                  file=False,
                                  junit=False,
                                  json=False,
                                  outputDir="",

                                  configFile="",
                                  additionalScenario="",

                                  debug=False,
                                  reloadWorld=False,
                                  record=record,
                                  randomize=False,
                                  repetitions=1,
                                  waitForEgo=False)

        print("Disturbance profile:", sensors)
        self.disturbance_params = sensors

        # Create signal handler for SIGINT
        if sys.platform != 'win32':
            signal.signal(signal.SIGHUP, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Save scenario_runner arguments
        self._args = args

        # Open CARLA client
        client = self.open_carla()
        if client is None:
            raise Exception("CARLA client failed to open!")

        print("Getting CARLA world from client...")
        world = client.get_world()
        assert(len(self.spectator_loc)==3)
        print("Setting spectator location...")
        spectator = world.get_spectator()
        new_location = carla.Location(x=float(self.spectator_loc[0]), y=float(self.spectator_loc[1]), z=50+float(self.spectator_loc[2]))
        spectator.set_transform(carla.Transform(new_location, carla.Rotation(pitch=-90)))

        # if self.save_video:
            # video_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            # camera_tf = carla.Transform(new_location, carla.Rotation(pitch=90))
            # self.video_camera = world.spawn_actor(video_camera_bp, camera_tf)
            # image_count = 0
            # def save_video_image(image):
            #     nonlocal image_count # closure
            #     image.save_to_disk('images/spectator/%.6d.jpg' % image_count)
            #     image_count += 1
            # self.video_camera.listen(save_video_image)
        # else:
        #     self.video_camera = None

        print("Getting CARLA world settings...")
        settings = world.get_settings()
        print("Setting CARLA no render mode:", no_rendering)
        settings.no_rendering_mode = no_rendering
        print("Applying CARLA world settings...")
        world.apply_settings(settings)

        # Create ScenarioRunner object to handle the core route/scenario parsing
        print("Creating AST scenario runner...")
        self.scenario_runner = ASTScenarioRunner(args, remove_other_actors=True)

        # Warm up the scenario_runner
        print("Warm-starting AST scenario runner...")
        self.scenario_runner.parse_scenario()

        # Environment parameters
        self.endtime = params['endtime']
        self.reward_bonus = params['reward_bonus']
        self.discount = params['discount']
        self.max_past_step = params['max_past_step']

        # self.dataset = []
        print("Resetting adversarial gym environment...")
        obs = self.reset(retdict=True)

        self.center = np.concatenate([v for v in obs.values()], axis=0)
        self.center[4::5] = 0.0

        self.actor_keys = list(obs.keys())  # Ensure actor is matched to the right key for observations
        print("No. of Actors: ", len(self.actor_keys))
        if len(self.actor_keys) <= 1:
            warnings.warn("Only one actor in the scenario.")
        # assert(len(self.actor_keys) > 1) # Ensure other actors are around (after removing background actors)

        # reward parameters
        self.sensor_params_list = self._associate_adv_sensor_params() # TODO: How to handle multiple adv. sensor types in the action space?
        self.mean_disturbance, self.var_disturbance = self._get_disturbance_params(self.sensor_params_list)

        # action space
        lower, upper = self._get_disturbance_action_bounds(self.sensor_params_list)
        self.action_space = spaces.Box(lower, upper, dtype=np.float32)

        # observation_space_dict = {}
        # for key in self.actor_keys:
        #   observation_space_dict[key] = spaces.Box(
        #     np.array(params['lower_actor_state']),
        #     np.array(params['upper_actor_state']), dtype=np.float32)

        # self.observation_space = spaces.Dict(observation_space_dict)

        # observation space
        assert len(params['lower_actor_state']) == len(params['upper_actor_state'])
        self.observation_space = spaces.Box(
                np.array(params['lower_actor_state']*len(self.actor_keys)),
                np.array(params['upper_actor_state']*len(self.actor_keys)), dtype=np.float32)

        print("Finished initializing AdversarialCARLAEnv.")


    def open_carla(self):
        # Set CARLA configuration (see CARLA\PythonAPI\util\config.py)
        def setup_carla(timeout=self._args.timeout):
            print("Getting CARLA client...")
            client = carla.Client(self._args.host, self._args.port, worker_threads=1)
            print("Setting CARLA timeout...")
            client.set_timeout(timeout)
            print("Loading CARLA world:", self.carla_map)
            client.load_world(self.carla_map)
            print("Finished CARLA setup.")
            return client

        client = None
        try:
            print("Checking if CARLA is open, then setting up.")
            client = setup_carla()
            print("CARLA executable is already open.")
        except Exception as exception:
            traceback.print_exc()
            print(exception)
            self.close()
            # try:
            #     print("CARLA not open, now opening executable.")
            #     CARLA_ROOT_NAME = "CARLA_ROOT"
            #     if CARLA_ROOT_NAME not in os.environ:
            #         raise Exception("Please set your " + CARLA_ROOT_NAME + " environment variable to the base directory where CarlaUE4.{exe|sh} lives.")
            #     else:
            #         CARLA_ROOT = os.environ[CARLA_ROOT_NAME]
            #     if os.name == 'nt': # Windows
            #         cmd_str = "start " + CARLA_ROOT + "\\CarlaUE4.exe -carla-rpc-port=2000 -windowed -ResX=320 -ResY=240 -benchmark -fps=10 -quality-level=Low"
            #     else:
            #         cmd_str = CARLA_ROOT + "/CarlaUE4.sh -carla-rpc-port=2000 -windowed -ResX=320 -ResY=240 -benchmark -fps=10 -quality-level=Low &"
            #     os.system(cmd_str)
            #     self.carla_running = True
            #     time.sleep(self._args.timeout) # Delay while CARLA spins up
            #     print("Configuring CARLA.")
            #     world = setup_carla()
            # except Exception as next_exception:
            #     print("CARLA cannot be opened.")
            #     traceback.print_exc()
            #     print(next_exception)
            #     self.close()
            #     return None
        return client


    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self.scenario_runner is not None:
            self.scenario_runner._signal_handler(signum, frame)
            self.close()


    def destroy(self):
        self.close()


    def close(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        print("(AdversarialCARLAEnv) Destroyed.")
        if self.scenario_runner is not None:
            print("\t- Deleting CARLA scenario_runner...")
            self.scenario_runner.finished = False
            self.scenario_runner.destroy()
            del self.scenario_runner
        # if self.video_camera is not None:
        #     del self.video_camera


    def reset(self, retdict=False):
        try:
            # Reload scenario
            self.scenario_runner.load_scenario()

            # Handle replacing sensors with their adversarial counterpart
            self._adversarial_replace_sensors()

            self._prev_distance = 10000 # TODO...
            self._prev_speed = 0
            self._timestep = 0
            self._info = {
                'timestep': 0,
                'collision': None,
                'failed_scenario': None}
            self.turn_on_headlights()
            return self._observation(retdict)
        except Exception as e:
            traceback.print_exc()
            print("Could not reset env due to: {}".format(e))
            self.scenario_runner._cleanup()
            exit(-1)


    def turn_on_headlights(self):
        light_mask = carla.VehicleLightState.NONE | carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position
        world = CarlaDataProvider.get_world()
        all_vehicles = world.get_actors()
        for ve in all_vehicles:
            if "vehicle." in ve.type_id:
                ve.set_light_state(carla.VehicleLightState(light_mask))


    def step(self, action):
        try:
            done = False
            # self._actions.append(action)
            # if isinstance(action, np.ndarray):
            #     disturbance = {'x': action[::2].astype(np.float64), 'y': action[1::2].astype(np.float64)}
            # else:
            #     disturbance = {'x': action[::2], 'y': action[1::2]}
            for _ in range(self.block_size):
                self.scenario_runner.running, distance = self._tick_scenario_ast(action)
                if not self.scenario_runner.running:
                    break

            collision, collision_intensity = self.scenario_runner._check_failures()

            observation = self._observation() # done before cleanup
            # self._observations.append(observation)

            if not self.scenario_runner.running:
                result = self.scenario_runner._stop_scenario(self.scenario_runner.start_time, self.scenario_runner.recorder_name, self.scenario_runner.scenario)
                self.scenario_runner._cleanup()

            running = self.scenario_runner.running

            rate = self._prev_distance - distance
            self._prev_distance = distance
            # rate = self._distances[-1] - distance
            # self._distances.append(distance)

            velocity = self.scenario_runner.manager.ego_vehicles[0].get_velocity()
            speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            ms_to_mph = 2.23694
            speed_mph = speed_ms * ms_to_mph
            delta_v = self._prev_speed - speed_mph
            
            # TODO: get force of collision.
            if collision:
                self._info['cost'] = collision_intensity # self._prev_speed # NOTE: speed at time _right_ before collision (not after collision happened).
            else:
                self._info['cost'] = 0

            # Update previous speed after we record it as a cost.
            self._info['speed_before_collision'] = self._prev_speed
            self._prev_speed = speed_mph
            self._failed_scenario = collision

            timestep = self._info['timestep']
            if not running or collision or timestep > self.endtime:
                done = True
            #     _y = self._failed_scenario
            #     _x = (self._actions, min(self._distances), rate)
            #     self.dataset.append((_x, _y))

            # Update info
            agent = self.scenario_runner.agent_instance
            self._info['timestep'] += 1
            self._info['action'] = action
            self._info['observation'] = observation
            self._info['collision'] = collision
            self._info['failed_scenario'] = self._failed_scenario
            self._info['distance'] = distance
            self._info['rate'] = rate
            self._info['speed'] = speed_mph
            self._info['delta_v'] = delta_v
            self._info['done'] = done

            # Include all necessary state information in `info` to pass to Julia (positions, velocities, etc)
            # if agent is None:
            #     self._info['ego_sensor_location'] = None
            #     self._info['ego_truth_location'] = None
            # else:
            #     if hasattr(agent, 'ego_truth_location') and agent.ego_truth_location is not None:
            #         self._info['ego_truth_location'] = [agent.ego_truth_location.x, agent.ego_truth_location.y]
            #     else:
            #         self._info['ego_sensor_location'] = 0
            #     if hasattr(agent, 'ego_sensor_location') and agent.ego_sensor_location is not None:
            #         self._info['ego_sensor_location'] = [agent.ego_sensor_location.x, agent.ego_sensor_location.y]
            #     else:
            #         self._info['ego_truth_location'] = 0

            # Calculate the reward for this step
            reward = self._reward(self._info)
            self._info['reward'] = reward

            return (observation, reward, done, copy.deepcopy(self._info))
        except Exception as e:
            traceback.print_exc()
            print("Could not step env due to: {}".format(e))
            self.scenario_runner._cleanup()
            exit(-1)


    def _reward(self, info):
        action = info['action']
        failed_scenario = info['failed_scenario']
        collision = info['collision']
        distance = info['distance']

        exposure = action[-1] # TODO.
        exposure_mean, exposure_std = exposure_mean_std(exposure)
        camera_idx = 1 # TODO.
        sensor_params_list = copy.deepcopy(self.sensor_params_list)
        self.mean_disturbance[-1] = exposure_mean
        self.var_disturbance[-1] = exposure_std**2
        # sensor_params_list[camera_idx]['exposure_compensation']['mean'] = exposure_mean
        # sensor_params_list[camera_idx]['exposure_compensation']['std'] = exposure_std
        # self.mean_disturbance, self.var_disturbance = self._get_disturbance_params(sensor_params_list) # TODO. Cleverly only recompute the camera exposure values (not all of them)

        # TODO: scale to be reasonable sized (Within [-1, 1])
        reward = -utils.mahalanobis_d(action, self.mean_disturbance, self.var_disturbance)/utils.mahalanobis_d(10*self.var_disturbance, self.mean_disturbance, self.var_disturbance)

        if collision:
            reward += 100

        return np.float32(reward)


    def _observation(self, retdict=False):
        obs = {}

        # Get actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        walker_poly_dict = self._get_actor_polygons('walker.*')

        if hasattr(self, 'actor_keys'):
            # Default values
            for key in self.actor_keys:
                obs[key] = np.zeros(5, dtype=np.float32)

        for i, (key, value) in enumerate(vehicle_poly_dict.items()):
            obs['veh_'+str(i)] = np.ones(5, dtype=np.float32)
            obs['veh_'+str(i)][:4] = value.flatten().astype(np.float32)

        for i, (key, value) in enumerate(walker_poly_dict.items()):
            obs['walker_'+str(i)] = np.ones(5, dtype=np.float32)
            obs['walker_'+str(i)][:4] = value.flatten().astype(np.float32)

        if retdict:
            return obs

        o = np.concatenate([v for v in obs.values()], axis=0)

        # TODO: turn into parameterized values in the gym env.
        mean = self.center
        n_act = int(len(o)/5)
        std = np.array([20, 20, 20, 20, 1.0]*n_act)

        return utils.normalize_observations(o, mean, std)


    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.
        Args:
            filt: the filter indicating what type of actors we'll look at.
        Returns:
            actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        world = CarlaDataProvider.get_world()
        actors = world.get_actors()
        for actor in actors.filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw/180*np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l,w],[-l,-w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],2,axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict


    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print("Timestep:", self._info['timestep'])
        print("Action:", self._info['action'])
        print("Reward:", self._info['reward'])
        print("Collision:", self._info['collision'])
        print("Failed:", self._info['failed_scenario'])
        print("Cost:", self._info['cost'])
        # print("Ego sensor (x,y):", self._info['ego_sensor_location'])
        # print("Ego truth (x,y):", self._info['ego_truth_location'])
        print("="*50)


    def _tick_scenario_ast(self, action):
        """
        Progresses the scenario tick-by-tick for AST interface
        """
        distance = 1000 # TODO. Parameterize? Larger?
        if self.scenario_runner.manager._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if self.scenario_runner.manager._timestamp_last_run < timestamp.elapsed_seconds:
                self.scenario_runner.manager._timestamp_last_run = timestamp.elapsed_seconds

                self._set_disturbances(action)

                self.scenario_runner.manager._watchdog.update()

                if self.scenario_runner.manager._debug_mode:
                    print("\n--------- Tick ---------\n")

                # Update game time and actor information
                GameTime.on_carla_tick(timestamp)
                CarlaDataProvider.on_carla_tick()

                # AST: Apply disturbance
                if self.scenario_runner.manager._agent is not None:
                    ego_action = self.scenario_runner.manager._agent()  # pylint: disable=not-callable
                    # TODO: Calculate minimium distance to other agents
                    # distance = self.scenario_runner.manager._agent._agent.min_distance

                if self.scenario_runner.manager._agent is not None:
                    self.scenario_runner.manager.ego_vehicles[0].apply_control(ego_action)

                # Tick scenario
                self.scenario_runner.manager.scenario_tree.tick_once()

                if self.follow_ego:
                    hero_actor = None
                    for actor in CarlaDataProvider.get_world().get_actors():
                        if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                            hero_actor = actor
                            break
                    if hero_actor:
                        spectator_loc = hero_actor.get_location()
                        # spectator_loc = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')[1].get_location()
                        new_location = carla.Location(x=spectator_loc.x, y=spectator_loc.y, z=50+spectator_loc.z)
                        spectator_tf = carla.Transform(new_location, carla.Rotation(pitch=-90))
                        spectator = world.get_spectator()
                        spectator.set_transform(spectator_tf)
                        # if self.save_video:
                        #     self.video_camera.set_transform(spectator_tf)


                if self.scenario_runner.manager._debug_mode:
                    print("\n")
                    py_trees.display.print_ascii_tree(self.scenario_runner.manager.scenario_tree, show_status=True)
                    sys.stdout.flush()

                if self.scenario_runner.manager.scenario_tree.status != py_trees.common.Status.RUNNING:
                    self.scenario_runner.manager._running = False

            if self.scenario_runner.manager._sync_mode and self.scenario_runner.manager._running and self.scenario_runner.manager._watchdog.get_status():
                CarlaDataProvider.get_world().tick()

        return self.scenario_runner.manager._running, distance


    def _set_disturbances(self, action):
        ids = self._associate_adv_sensor_id()
        for i,id in enumerate(ids):
            offset = self.calc_action_space_offset(ids, i) # handle flattened actions across different sensors
            self.adv_sensor_callbacks[id].set_disturbance(action, offset)


    def calc_action_space_offset(self, ids, i):
        return sum([self.adv_sensor_callbacks[id].dims for id in ids[:i]])


    def _get_disturbance_params(self, params_list):
        means = np.array([])
        variances = np.array([])
        for params in params_list:
            sensor_type = self._id_to_sensor_type(params['id'])
            if sensor_type in ADVERSARIAL_SENSOR_MAPPING:
                for key in ADVERSARIAL_SENSOR_MAPPING[sensor_type]['params']:
                    means = np.append(means, params[key]['mean'])
                    variances = np.append(variances, params[key]['std']**2)
            else:
                raise Exception("Please add the following sensor type to ADVERSARIAL_SENSOR_MAPPING: " + sensor_type)
        return means, variances


    def _get_disturbance_action_bounds(self, params_list):
        lower = np.array([])
        upper = np.array([])
        for params in params_list:
            sensor_type = self._id_to_sensor_type(params['id'])
            if sensor_type in ADVERSARIAL_SENSOR_MAPPING:
                for key in ADVERSARIAL_SENSOR_MAPPING[sensor_type]['params']:
                    lower = np.append(lower, params[key]['lower'])
                    upper = np.append(upper, params[key]['upper'])
            else:
                raise Exception("Please add the following sensor type to ADVERSARIAL_SENSOR_MAPPING: " + sensor_type)
        return lower, upper


    def _id_to_sensor_type(self, id):
        sensors = self.scenario_runner.manager._agent._agent.sensors()
        for sensor in sensors:
            if sensor['id'] == id:
                return sensor['type']
        raise Exception("No associated sensor with id " + id + " in the agent's sensors() list.")


    def _associate_adv_sensor_id(self):
        ids = []
        for params in self.disturbance_params:
            id = params['id']
            if id in self.adv_sensor_callbacks:
                ids.append(id)
        if len(ids) == 0:
            raise Exception("Could not find matching adversarial sensor IDs.")
        else:
            return ids


     # TODO: How to handle multiple adv. sensor types in the action space?
    def _associate_adv_sensor_params(self):
        params_list = []
        for params in self.disturbance_params:
            id = params['id']
            if id in self.adv_sensor_callbacks:
                params_list.append(params)
            else:
                raise Exception("No matching sensor with id: " + id)
        if len(params_list) == 0:
            raise Exception("No matching sensor found. Please set the `sensors` input to the gym environment.")
        else:
            return params_list


    def _associate_sensor_id(self, sensor_interface, sensor):
        sensors_objects = sensor_interface._sensors_objects
        for id in sensors_objects.keys():
            if sensors_objects[id] == sensor:
                return id # guarenteed to find something


    def _adversarial_replace_sensors(self):
        for sensor in self.scenario_runner.manager._agent._sensors_list:
            if sensor is not None:
                sensor_interface = self.scenario_runner.manager._agent._agent.sensor_interface

                # Replace adversarial version of the sensor(s)
                if sensor.type_id in ADVERSARIAL_SENSOR_MAPPING:
                    # Get the `id` field that is associated to the output of AutonomousAgent.sensors()
                    id = self._associate_sensor_id(sensor_interface, sensor)
                    if id is not None and (id is "rgb" or id is "GPS"):
                        # Get the adversarial sensor callback class from the mapping
                        if id == "VIDEO_CAMERA":
                            AdvCallBack = VideoCameraCallBack
                        else:
                            AdvCallBack = ADVERSARIAL_SENSOR_MAPPING[sensor.type_id]['callback']

                        # Replace the sensor by first deleting it, stopping it, creating a new callback, then starting the listener.
                        del sensor_interface._sensors_objects[id]
                        sensor.stop()
                        self.adv_sensor_callbacks[id] = AdvCallBack(id, sensor, sensor.type_id, sensor_interface)
                        sensor.listen(self.adv_sensor_callbacks[id])
                else:
                    warnings.warn("No adversarial version of the sensor type: " + sensor.type_id)




class VideoCameraCallBack(CallBack):
    """
    Class the sensors listen to in order to receive their data each frame
    """

    def __init__(self, tag, sensor_type, sensor, data_provider):
        """
        Initializes the call back
        """
        super().__init__(tag, sensor_type, sensor, data_provider)
        self.counter = 0


    def __call__(self, data):
        """
        call function
        """
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (data.height, data.width, 4))

        save_camera_image(array, "./images/spectator/" + str(self.counter) + "_spectator.png")

        self._data_provider.update_sensor(self._tag, array, data.frame)
        self.counter += 1

    def set_disturbance(self, disturbance, offset=0):
        # Dummy pass-through.
        pass