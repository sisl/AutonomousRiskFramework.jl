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

import carla
import py_trees

sys.path.append("../scenario_runner") # Add scenario_runner package to import path

from ast_scenario_runner import ASTScenarioRunner
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from pdb import set_trace as breakpoint # DEBUG. TODO!

import utils

# Version of adversarial_carla_env (tied to CARLA and scenario_runner versions)
VERSION = '0.9.11'

DEFAULT_PARAMS = {
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


class AdversarialCARLAEnv(gym.Env):

    """
    This is the core scenario runner module. It is responsible for
    running (and repeating) a single scenario or a list of scenarios.

    Usage:
    env = AdversarialCARLAEnv(scenario, agent, config)
    obs = env.reset()
    t = r = 0
    done = False
    while not done:
        t += 1
        action = 0.0
        obs, reward, done, info = env.step(action)
        r += reward
    del env
    """

    # CARLA scenario handler
    scenario_runner = None
    world = None

    # Scenario/route selections
    route_file = "data/routes_ast.xml" # TODO: None then configure as input to __init__
    scenario_file = "data/ast_scenarios.json" # TODO?
    route_id = 0 # TODO: Can we use this to control the background activity?
    route = [route_file, scenario_file, route_id]
    scenario = None # TODO?
    scenario_config = None

    # Agent selections
    agent = "agents/ast_agent.py"

    # CARLA configuration parameters
    port = 2222

    # Recoding parameters
    record = "recordings" # TODO: what are we recording here and is it needed?

    # CARLA configuration parameters
    carla_map = "Town01"
    spectator_loc = [80.37, 25.30, 0.0]
    no_rendering = False # TODO: make this configurable
    carla_running = False

    # Gym environment parameters
    block_size = 10


    def __init__(self, *, route=route, scenario=scenario, agent=agent, port=port, record=record, params=DEFAULT_PARAMS):
        """
        Setup ScenarioRunner
        """

        # Setup arguments passed to the ScenarioRunner constructor
        # TODO: How to piggy-back on scenario_runner.py "main()" defaults?
        args = argparse.Namespace(host="127.0.0.1",
                                  port=port,
                                  timeout=20.0,
                                  trafficManagerPort=8000,
                                  trafficManagerSeed=0,
                                  sync=True,
                                  list=False,

                                  scenario=scenario, # TODO: NOTE, --agent used to execute the scenario is currently only compatible with route-based scenarios.
                                  openscenario=None,
                                  openscenarioparams=None,
                                  route=route,

                                  agent=agent,
                                  agentConfig="",

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

        self.world = client.get_world()
        assert(len(self.spectator_loc)==3)
        spectator = self.world.get_spectator()
        new_location = carla.Location(x=float(self.spectator_loc[0]), y=float(self.spectator_loc[1]), z=50+float(self.spectator_loc[2]))
        spectator.set_transform(carla.Transform(new_location, carla.Rotation(pitch=-90)))

        settings = self.world.get_settings()
        settings.no_rendering_mode = self.no_rendering

        # Create ScenarioRunner object to handle the core route/scenario parsing
        self.scenario_runner = ASTScenarioRunner(args)

        # Warm up the scenario_runner
        self.scenario_runner.parse_scenario()

        # Environment parameters
        self.endtime = params['endtime']
        self.reward_bonus = params['reward_bonus']
        self.discount = params['discount']
        self.max_past_step = params['max_past_step']

        # self.dataset = []

        obs = self.reset(retdict=True)

        self.center = np.concatenate([v for v in obs.values()], axis=0)
        self.center[4::5] = 0.0

        self.actor_keys = list(obs.keys())  # Ensure actor is matched to the right key for observations
        print("No. of Actors: ", len(self.actor_keys))

        self.var_disturbance = np.array(params['var_disturbance']*(len(self.actor_keys)-1))
        self.mean_disturbance = np.array(params['mean_disturbance']*(len(self.actor_keys)-1))

        # action/observation spaces
        assert len(params['lower_disturbance']) == len(params['upper_disturbance'])
        self.action_space = spaces.Box(
                np.array(params['lower_disturbance']*(len(self.actor_keys)-1)),
                np.array(params['upper_disturbance']*(len(self.actor_keys)-1)), dtype=np.float32)

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


    def open_carla(self):
        # Set CARLA configuration (see CARLA\PythonAPI\util\config.py)
        def setup_carla(timeout=self._args.timeout):
            client = carla.Client(self._args.host, self._args.port, worker_threads=1)
            client.set_timeout(timeout)
            client.load_world(self.carla_map)
            return client

        client = None
        try:
            print("Checking if CARLA is open, then setting up.")
            client = setup_carla()
            print("CARLA executable is already open.")
        except Exception as exception:
            print("CARLA cannot be opened.")
            traceback.print_exc()
            print(exception)
            self.close()
            return None
            # print("CARLA not open, now opening executable.")
            # CARLA_ROOT_NAME = "CARLA_ROOT"
            # if CARLA_ROOT_NAME not in os.environ:
            #     raise Exception("Please set your " + CARLA_ROOT_NAME + " environment variable to the base directory where CarlaUE4.{exe|sh} lives.")
            # else:
            #     CARLA_ROOT = os.environ[CARLA_ROOT_NAME]

            # if os.name == 'nt': # Windows
            #     cmd_str = "start " + CARLA_ROOT + "\\CarlaUE4.exe -carla-rpc-port=2222 -windowed -ResX=320 -ResY=240 -benchmark -fps=10 -quality-level=Low"
            # else:
            #     cmd_str = CARLA_ROOT + "/CarlaUE4.sh -carla-rpc-port=2222 -windowed -ResX=320 -ResY=240 -benchmark -fps=10 -quality-level=Low &"
            # os.system(cmd_str)
            # self.carla_running = True
            # time.sleep(10) # Delay while CARLA spins up
            # print("Configuring CARLA.")
            # client = setup_carla()
        return client


    def __del__(self):
        self.close()


    def close(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        print("(AdversarialCARLAEnv) Destroyed.")
        if self.world is not None:
            del self.world
        if self.scenario_runner is not None:
            self.scenario_runner.destroy()
            del self.scenario_runner


    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self.scenario_runner._signal_handler(signum, frame)
        self.close()


    def reset(self, retdict=False):
        self.scenario_runner.load_scenario()
        self.world = CarlaDataProvider.get_world()
        self._prev_distance = 10000 # TODO...
        self._timestep = 0
        self._info = {
            'timestep': 0,
            'collision': None,
            'failed_scenario': None}

        return self._observation(retdict)


    def step(self, action):
        done = False
        # self._actions.append(action)

        disturbance = {'x': action[::2].astype(np.float64), 'y': action[1::2].astype(np.float64)}
        for _ in range(self.block_size):
            self.scenario_runner.running, distance = self._tick_scenario_ast(disturbance)
            if not self.scenario_runner.running:
                break

        collision = self.scenario_runner._check_failures()

        if not self.scenario_runner.running:
            result = self.scenario_runner._stop_scenario(self.scenario_runner.start_time, self.scenario_runner.recorder_name, self.scenario_runner.scenario)
            self.scenario_runner._cleanup()

        running = self.scenario_runner.running

        rate = self._prev_distance - distance
        self._prev_distance = distance
        # rate = self._distances[-1] - distance
        # self._distances.append(distance)

        self._failed_scenario = collision

        if not running or collision:
            done = True
        #     _y = self._failed_scenario
        #     _x = (self._actions, min(self._distances), rate)
        #     self.dataset.append((_x, _y))

        observation = self._observation()
        # self._observations.append(observation)

        # Update info
        self._info['timestep'] += 1
        self._info['action'] = action
        self._info['observation'] = observation
        self._info['collision'] = collision
        self._info['failed_scenario'] = self._failed_scenario
        self._info['distance'] = distance
        self._info['rate'] = rate

        # Calculate the reward for this step
        reward = self._reward(self._info)
        self._info['reward'] = reward

        return (observation, reward, done, copy.deepcopy(self._info))


    def _reward(self, info):
        action = info['action']
        failed_scenario = info['failed_scenario']
        collision = info['collision']
        distance = info['distance']

        # TODO: scale to be reasonable sized (Within [-1, 1])
        reward = -utils.mahalanobis_d(action, self.mean_disturbance, self.var_disturbance)/utils.mahalanobis_d(10*self.var_disturbance, self.mean_disturbance, self.var_disturbance)
        if collision:
            reward += 100

        return reward


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
        for actor in self.world.get_actors().filter(filt):
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
        print("Timestep: ", self._info['timestep'])
        print("Collision: ", self._info['collision'])
        print("Failed: ", self._info['failed_scenario'])



    def _tick_scenario_ast(self, disturbance):
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

                self.scenario_runner.manager._watchdog.update()

                if self.scenario_runner.manager._debug_mode:
                    print("\n--------- Tick ---------\n")

                # Update game time and actor information
                GameTime.on_carla_tick(timestamp)
                CarlaDataProvider.on_carla_tick()

                # AST: Apply disturbance
                if self.scenario_runner.manager._agent is not None:
                    self.scenario_runner.manager._agent._agent.set_disturbance(disturbance)
                    ego_action = self.scenario_runner.manager._agent()  # pylint: disable=not-callable
                    distance = self.scenario_runner.manager._agent._agent.min_distance

                if self.scenario_runner.manager._agent is not None:
                    self.scenario_runner.manager.ego_vehicles[0].apply_control(ego_action)

                # Tick scenario
                self.scenario_runner.manager.scenario_tree.tick_once()

                if self.scenario_runner.manager._debug_mode:
                    print("\n")
                    py_trees.display.print_ascii_tree(self.scenario_runner.manager.scenario_tree, show_status=True)
                    sys.stdout.flush()

                if self.scenario_runner.manager.scenario_tree.status != py_trees.common.Status.RUNNING:
                    self.scenario_runner.manager._running = False

            if self.scenario_runner.manager._sync_mode and self.scenario_runner.manager._running and self.scenario_runner.manager._watchdog.get_status():
                CarlaDataProvider.get_world().tick()

        return self.scenario_runner.manager._running, distance

