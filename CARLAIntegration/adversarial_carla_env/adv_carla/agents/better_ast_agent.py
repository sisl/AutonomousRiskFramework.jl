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
from ast_sensor_wrapper import GnssSensor

from pdb import set_trace as breakpoint # DEBUG. TODO!

repetitions = 0

class BetterAstAgent(AutonomousAgent):

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
        self.use_ego_truth = True

        self.ego_truth_location = None
        self.ego_sensor_location = None


    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = [
            {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'id': 'GPS'},
            {'type': 'sensor.other.obstacle', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'id': 'Obstacle'}
        ]

        return sensors


    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        gnss_data = input_data['GPS'][1]
        # breakpoint()

        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = BasicAgent(hero_actor, target_speed=25)

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
            if self.detect_hazard(gnss_data):
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


    def detect_hazard(self, gnss_data, debug=False):
        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._agent._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list, gnss_data)
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


    def _is_vehicle_hazard(self, vehicle_list, gnss_data):
        """
        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """
        global repetitions


        # record for `render`
        self.ego_truth_location = self._agent._vehicle.get_location() # Truth

        (x, y, z) = GnssSensor.gps_to_location(gnss_data[0], gnss_data[1], gnss_data[2])
        ego_vehicle_location = carla.Location(x=x, y=y, z=z)

        # record for `render`
        self.ego_sensor_location = ego_vehicle_location

        ego_vehicle_waypoint = self._agent._map.get_waypoint(ego_vehicle_location)

        veh_idx = -1

        self.min_distance = 10000

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._agent._vehicle.id:
                continue

            veh_idx += 1

            target_tf = target_vehicle.get_transform()
            truth_target_location = target_tf.location

            # Save minimum distance for AST rewards
            self.min_distance = min(self.min_distance, compute_distance(truth_target_location, self._agent._vehicle.get_location()))

            target_vehicle_waypoint = self._agent._map.get_waypoint(target_tf.location)
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(target_tf,
                                        self._agent._vehicle.get_transform(),
                                        self._agent._proximity_vehicle_threshold):
                return (True, target_vehicle)

        return (False, None)
