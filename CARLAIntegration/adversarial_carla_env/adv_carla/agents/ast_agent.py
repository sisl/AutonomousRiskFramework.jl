#!/usr/bin/env python

"""
This module provides an AST agent to control the ego vehicle
"""

from __future__ import print_function

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.agent import AgentState
from agents.tools.misc import is_within_distance_ahead, is_within_distance, compute_distance
import random
import numpy as np
import weakref
import sys
import os

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from ast_sensor_wrapper import ASTSensorWrapper

repetitions = 0

class AstAgent(AutonomousAgent):

    """
    Autonomous agent to control the ego vehicle for Adaptive Stress Testing
    """

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        global repetitions

        self._route_assigned = False
        self._agent = None
        self.time = 0
        self.min_distance = 100000
        repetitions += 1
        self.disturbance = None # dict of 'x' and 'y' disturbances per target
        self.ego_sensor = None
        self.target_sensors = []
        self.target_sensor_obs = []
        self.target_truths = []
        self.use_ego_truth = True


    def set_disturbance(self, disturbance):
        self.disturbance = disturbance


    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = BasicAgent(hero_actor, target_speed=25)
                self.ego_sensor = ASTSensorWrapper(hero_actor, 'gnss') # TODO: pass in `sensor_type` and `sensor_params`

            return control

        if not self._route_assigned:
            if self._global_plan:
                plan = []

                for transform, road_option in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    plan.append((wp, road_option))

                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True
        else:
            if self.detect_hazard():
                control = self._agent.emergency_stop()
            else:
                self.update_agent_state(AgentState.NAVIGATING)
                # standard local planner behavior
                control = self._agent._local_planner.run_step()

        self.time += 1
        return control


    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data()

        timestamp = GameTime.get_time()
        wallclock = GameTime.get_wallclocktime()
        # print('======[Agent] Wallclock_time = {} / Sim_time = {}'.format(wallclock, timestamp)) # OVERLOAD just to comment this out.

        control = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        return control


    def update_agent_state(self, state):
        self._agent._state = state


    def detect_hazard(self, debug=False):
        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._agent._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self.update_agent_state(AgentState.BLOCKED_BY_VEHICLE)
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = self._agent._is_light_red(lights_list)
        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self.update_agent_state(AgentState.BLOCKED_RED_LIGHT)
            hazard_detected = True

        return hazard_detected


    def _is_vehicle_hazard(self, vehicle_list):
        """
        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """
        global repetitions

        if self.use_ego_truth:
            ego_vehicle_location = self._agent._vehicle.get_location()
        else:
            # use sensor observation for ego location
            ego_vehicle_location = self.ego_sensor.get_location()

        ego_vehicle_waypoint = self._agent._map.get_waypoint(ego_vehicle_location)

        veh_idx = -1

        self.min_distance = 10000

        if len(vehicle_list) == 1: # i.e., only the ego
            # empty out stale info
            self.target_truths = []
            self.target_sensor_obs = []

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._agent._vehicle.id:
                continue

            veh_idx += 1

            # add sensor to every non-ego vehicle (if not done so already)
            if len(self.target_sensors) != veh_idx + 1:
                self.target_sensors.append(ASTSensorWrapper(target_vehicle, 'gnss')) # TODO: pass in target_sensor_type of 'gnss'
                self.target_sensor_obs.append(None)
                self.target_truths.append(None)

            target_sensor = self.target_sensors[veh_idx]

            target_tf = target_vehicle.get_transform()
            truth_target_location = target_tf.location

            # Save minimum distance for AST rewards
            self.min_distance = min(self.min_distance, compute_distance(truth_target_location, self._agent._vehicle.get_location()))

            # record truth value (for rendering)
            self.target_truths[veh_idx] = (truth_target_location.x, truth_target_location.y)

            # Compute the noisy target vehicle position
            if self.disturbance is None:
                x_noise = 0
                y_noise = 0
            else:
                x_noise = self.disturbance['x'][veh_idx]
                y_noise = self.disturbance['y'][veh_idx]

            # apply sensor noise disturbance
            target_tf.location = target_sensor.apply_disturbance({'x': x_noise, 'y': y_noise})

            # record sensor observation (for rendering)
            self.target_sensor_obs[veh_idx] = (target_tf.location.x, target_tf.location.y)

            target_vehicle_waypoint = self._agent._map.get_waypoint(target_tf.location) # Noise added
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(target_tf,
                                        self._agent._vehicle.get_transform(),
                                        self._agent._proximity_vehicle_threshold):
                return (True, target_vehicle)

        return (False, None)


    def destroy(self):
        sensors = [
            self.ego_sensor
        ]
        sensors += self.target_sensors
        for sensor in sensors:
            if sensor is not None:
                sensor.destroy()
