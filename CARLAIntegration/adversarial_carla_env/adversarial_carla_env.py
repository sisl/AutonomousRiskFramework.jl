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

from __future__ import print_function # TODO: Needed?

import gym
import numpy as np
import random
import copy
from gym import spaces

import traceback
import argparse
import os
import sys
import time

import carla_gym_ast as cg_ast

import pickle
import py_trees

sys.path.append("../scenario_runner") # Add scenario_runner package to import path

from scenario_runner import ScenarioRunner
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarios.route_scenario import RouteScenario
from srunner.scenariomanager.timer import GameTime

import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback

from pdb import set_trace as breakpoint # DEBUG. TODO!

# Version of adversarial_carla_env (tied to CARLA and scenario_runner versions)
VERSION = '0.9.11'

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

    # Scenario/route selections
    route_file = "data/routes_ast.xml" # TODO: None then configure as input to __init__
    scenario_file = "data/ast_scenarios.json" # TODO?
    route_id = 0 # TODO: Can we use this to control the background activity?
    route = [route_file, scenario_file, route_id]
    scenario = None # TODO?

    # Agent selections
    agent = "agents/ast_agent.py"

    # CARLA configuration parameters
    port = 2222

    # Recoding parameters
    record = "recordings" # TODO: what are we recording here and is it needed?


    def __init__(self, *, route=route, scenario=scenario, agent=agent, port=port, record=record):
        """
        Setup ScenarioRunner
        """

        # Setup arguments passed to the ScenarioRunner constructor
        # TODO: How to piggy-back on scenario_runner.py "main()" defaults?
        args = argparse.Namespace(host="127.0.0.1",
                                  port=port,
                                  timeout=10.0,
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
        self._args = args

        # Create ScenarioRunner object to handle the core route/scenario parsing
        self.scenario_runner = ScenarioRunner(args)
        self.scenario_runner._args.reloadWorld = False # Force no-reload.

        # Monkey patching the core "_load_and_run_scenario" function of scenario_runner.py
        self.scenario_runner._load_and_run_scenario = self._ast_run_scenario


    def run(self):
        return self.scenario_runner.run()


    def destroy(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        print("(AdversarialCARLAEnv) Destroyed.")
        del self.scenario_runner


    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self.scenario_runner._signal_handler(signum, frame)


    def create_env(self, config):
        self.scenario_runner._args.record = False
        self.scenario_runner.temp_scenario = {'count': 0, 'start_time': None,
                                              'recorder_name': None, 'scenario': None}

        block_size = 10
        self.scenario_runner.running = False

        def reset_fn():
            if self.scenario_runner.running:
                result = self._stop_scenario(self.scenario_runner.temp_scenario['start_time'], self.scenario_runner.temp_scenario['recorder_name'], self.scenario_runner.temp_scenario['scenario'])
                self.scenario_runner._cleanup()
            config.name = "RouteScenario_" + str(self.scenario_runner.temp_scenario['count'])
            start_time, recorder_name, scenario = self._load_scenario(config)
            self.scenario_runner.temp_scenario['start_time'] = start_time
            self.scenario_runner.temp_scenario['recorder_name'] = recorder_name
            self.scenario_runner.temp_scenario['scenario'] = scenario
            self.scenario_runner.temp_scenario['count'] += 1
            self.scenario_runner.running = True
            self.scenario_runner.test_status = "INIT"

        def step_fn(values):
            disturbance = {'x': values[::2].astype(np.float64), 'y': values[1::2].astype(np.float64)}
            # print("Disturbance: ", disturbance)
            for _ in range(block_size):
                self.scenario_runner.running, distance = self._tick_scenario_ast(disturbance)
                if not self.scenario_runner.running:
                    break
            failures = self._check_failures()
            if not self.scenario_runner.running:
                result = self._stop_scenario(self.scenario_runner.temp_scenario['start_time'], self.scenario_runner.temp_scenario['recorder_name'], self.scenario_runner.temp_scenario['scenario'])
                self.scenario_runner._cleanup()
            return self.scenario_runner.running, failures, distance

        # TODO: Abstract this env outside this part of the code!
        env = cg_ast.CARLAEnv(step_fn, reset_fn)
        return env


    def _ast_run_scenario(self, config, *, max_step=200):
        print("(AdversarialCARLAEnv) Monkey Patched: _ast_run_scenario")

        env = self.create_env(config)
        model = sb3.SAC("MlpPolicy", env, verbose=1)
        # model = sb3.SAC.load(os.path.join(os.getcwd(), "checkpoints", "td3_carla_10000_steps"))
        model.set_env(env)

        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/',
                                                 name_prefix='sac_carla_test')
        # Turn off if only evaluating
        n_episodes = 2
        episode_length = 200 # TODO: ?
        total_timesteps = n_episodes*episode_length # 10000
        model.learn(total_timesteps=total_timesteps, log_interval=2, callback=checkpoint_callback)

        # model.save(os.path.join(os.getcwd(), "variables", "td3_carla"))
        dataset_save_path = os.path.join(os.getcwd(), "variables", "dataset_test")
        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path)

        _samples =[]
        _dists = []
        _rates = []
        _y = []
        for data in env.dataset:
            _y.append(data[1])
            _samples.append(data[0][0])
            _dists.append(data[0][1])
            _rates.append(data[0][2])
        pickle.dump( _y, open( os.path.join(dataset_save_path, "y.pkl"), "wb" ) )
        pickle.dump( _samples, open( os.path.join(dataset_save_path, "samples.pkl"), "wb" ) )
        pickle.dump( _rates, open( os.path.join(dataset_save_path, "rates.pkl"), "wb" ) )
        pickle.dump( _dists, open( os.path.join(dataset_save_path, "dists.pkl"), "wb" ) )
        env = model.get_env()


        # TODO: Move this OUTSIDE somewhere else!!!!
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

        result = True
        return result


    def _stop_scenario(self, start_time, recorder_name, scenario):
        try:
            self._clean_scenario_ast(start_time)

            # Identify which criteria were met/not met
            # self.scenario_runner._analyze_scenario(config)

            # Remove all actors, stop the recorder and save all criterias (if needed)
            scenario.remove_all_actors()
            if self.scenario_runner._args.record:
                self.scenario_runner.client.stop_recorder()
                self.scenario_runner._record_criteria(self.scenario_runner.manager.scenario.get_criteria(), recorder_name)
            result = True

        except Exception as e:              # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        self.scenario_runner._cleanup()
        return result


    def _clean_scenario_ast(self, start_game_time):
        self.scenario_runner.manager._watchdog.stop()

        self.scenario_runner.manager.cleanup()

        self.scenario_runner.manager.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_runner.manager.scenario_duration_system = self.scenario_runner.manager.end_system_time - \
            self.scenario_runner.manager.start_system_time
        self.scenario_runner.manager.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_runner.manager.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("(AdversarialCARLAEnv) ScenarioManager: Terminated due to failure")


    def _load_scenario(self, config):
        """
        Load and run the scenario given by config
        NOTE: Copied and modified from scenario_runner.py _load_and_run_scenario()
        """
        start_time = -1000
        recorder_name = None
        scenario = None
        if not self.scenario_runner._load_and_wait_for_world(config.town, config.ego_vehicles):
            self.scenario_runner._cleanup()
            return start_time, recorder_name, scenario

        if self.scenario_runner._args.agent:
            agent_class_name = self.scenario_runner.module_agent.__name__.title().replace('_', '')
            try:
                self.scenario_runner.agent_instance = getattr(self.scenario_runner.module_agent, agent_class_name)(self.scenario_runner._args.agentConfig)
                config.agent = self.scenario_runner.agent_instance
            except Exception as e:          # pylint: disable=broad-except
                traceback.print_exc()
                print("Could not setup required agent due to {}".format(e))
                self.scenario_runner._cleanup()
                return start_time, recorder_name, scenario

        # Prepare scenario
        print("Preparing scenario: " + config.name)

        RouteScenario._initialize_actors = _initialize_actors # Monkey patching to avoid background actors

        try:
            self.scenario_runner._prepare_ego_vehicles(config.ego_vehicles)
            if self.scenario_runner._args.openscenario:
                scenario = OpenScenario(world=self.scenario_runner.world,
                                        ego_vehicles=self.scenario_runner.ego_vehicles,
                                        config=config,
                                        config_file=self.scenario_runner._args.openscenario,
                                        timeout=100000)
            elif self.scenario_runner._args.route:
                scenario = RouteScenario(world=self.scenario_runner.world,
                                         config=config,
                                         debug_mode=self.scenario_runner._args.debug)
            else:
                scenario_class = self.scenario_runner._get_scenario_class_or_fail(config.type)
                scenario = scenario_class(self.scenario_runner.world,
                                          self.scenario_runner.ego_vehicles,
                                          config,
                                          self.scenario_runner._args.randomize,
                                          self.scenario_runner._args.debug)
        except Exception as exception:                  # pylint: disable=broad-except
            print("The scenario cannot be loaded")
            traceback.print_exc()
            print(exception)
            self.scenario_runner._cleanup()
            return start_time, recorder_name, scenario

        # Change max trigger distance between route and scenario # TODO: Needed?
        # scenario.scenario.behavior.children[0].children[0].children[0]._distance = 5.5 # was 1.5 in route_scenario.py: _create_behavior()

        try:
            if self.scenario_runner._args.record:
                recorder_name = "{}/{}/{}.log".format(
                    os.getenv('SCENARIO_RUNNER_ROOT', "./"), self.scenario_runner._args.record, config.name)
                print(recorder_name)
                self.scenario_runner.client.start_recorder(recorder_name, True)

            # Load scenario and run it
            self.scenario_runner.manager.load_scenario(scenario, self.scenario_runner.agent_instance)
            start_time = self._prep_scenario_ast()

        except Exception as e:              # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        return start_time, recorder_name, scenario


    def _prep_scenario_ast(self):
        print("(AdversarialCARLAEnv) ScenarioManager: Running scenario {}".format(self.scenario_runner.manager.scenario_tree.name))
        self.scenario_runner.manager.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self.scenario_runner.manager._watchdog.start()
        self.scenario_runner.manager._running = True

        return start_game_time


    def _tick_scenario_ast(self, disturbance):
        """
        Progresses the scenario tick-by-tick for AST interface
        """
        distance = 1000
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


    def _cleanup(self):
        """
        Remove and destroy everything
        """
        print("AdversarialCARLAEnv: C L E A N  U P?")


    def _check_failures(self):
        failure = False
        for i, criterion in enumerate(self.scenario_runner.manager.scenario.get_criteria()):
            if i!=1:
                continue
            if (not criterion.optional and
                    criterion.test_status == "FAILURE" and
                        self.scenario_runner.test_status != "FAILURE"):
                failure = True
            elif criterion.test_status == "ACCEPTABLE":
                failure = False
        if failure:
            self.scenario_runner.test_status = "FAILURE"

        return failure


def _initialize_actors(self, config):
    """
    Set other_actors to the superset of all scenario actors
    NOTE: monkey patching _initialize_actors from route_scenario.py
    """

    # Add all the actors of the specific scenarios to self.other_actors
    for scenario in self.list_scenarios:
        self.other_actors.extend(scenario.other_actors)
