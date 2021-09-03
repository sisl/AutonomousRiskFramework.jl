#!/usr/bin/env python

"""
This module provides an AST agent to control the ego vehicle
"""

from __future__ import print_function

import carla
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.agent import AgentState
from agents.tools.misc import is_within_distance_ahead, is_within_distance, compute_distance
import random 
import numpy as np

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

repetitions = 0

class AstAgent2(AutonomousAgent):

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

    def set_disturbance(self, disturbance):
        self.disturbance = disturbance

    def sensors(self):
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


        """

        # sensors = [
        #     {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        #      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},
        # ]
        sensors = []

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

        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = BehaviorAgent(hero_actor, behavior='aggressive')    

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
            
            # if self.detect_hazard():
            #     control = self._agent.emergency_stop()
            # else:
            self.update_agent_state(AgentState.NAVIGATING)
            control = self._agent.run_step()
            self.disturbance = None

        self.time += 1
        return control

    def update_agent_state(self, state):
        # self._agent._state = state
        # world = CarlaDataProvider.get_world()
        self._agent.disturbance = self.disturbance
        self._agent.update_information(None)