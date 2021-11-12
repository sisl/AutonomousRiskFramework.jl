#!/usr/bin/env python

"""
This module provides an agent to control the ego vehicle using GNSS observations.
"""

from __future__ import print_function

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.agent import AgentState
from agents.tools.misc import is_within_distance_ahead, is_within_distance, compute_distance
import math
import random
import numpy as np
import weakref
import sys
import os

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


class GnssAgent(AutonomousAgent):
    """
    Autonomous agent to control the ego vehicle for that uses GNSS sensor observations.
    """

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self._route_assigned = False
        self._agent = None
        self.time = 0
        self.min_distance = 100000
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
            {'type': 'sensor.other.obstacle', 'id': 'OBSTACLE', 'distance': 100, 'debug_linetrace': True, 'hit_radius': 0.01},
            # {'type': 'sensor.other.collision', 'id': 'COLLISION'},
            # {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #           'width': 300, 'height': 200, 'fov': 100, 'id': 'CAMERA'},
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
        # gnss_data = [55, -1.9, 0] # input_data['GPS'][1]

        if input_data.get('GPS') is not None:
            gnss_data = input_data['GPS'][1]

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
        # record for `render`
        self.ego_truth_location = self._agent._vehicle.get_location() # Truth (not used for planning)

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


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """
    Class for GNSS sensors.

    Sensor parameters:
        https://carla.readthedocs.io/en/0.9.11/ref_sensors/#gnss-sensor
        ---------------------------------------------------------------
        noise_alt_bias      float   0.0    Mean parameter in the noise model for altitude.
        noise_alt_stddev    float   0.0    Standard deviation parameter in the noise model for altitude.
        noise_lat_bias      float   0.0    Mean parameter in the noise model for latitude.
        noise_lat_stddev    float   0.0    Standard deviation parameter in the noise model for latitude.
        noise_lon_bias      float   0.0    Mean parameter in the noise model for longitude.
        noise_lon_stddev    float   0.0    Standard deviation parameter in the noise model for longitude.
        noise_seed          int     0      Initializer for a pseudorandom number generator.
        sensor_tick         float   0.0    Simulation seconds between sensor captures (ticks).
    """


    def __init__(self, parent_actor, sensor_params=None):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')

        # set GNSS sensor noise parameters here (e.g.)
        if sensor_params is None:
            sensor_params = defaultdict(float) # defaults to all zeros
        else:
            # combine input sensor_params with defaultdict of zeros (prioritizing input values)
            sensor_params = defaultdict(float).update(sensor_params)

        # set sensor-level parameters
        blueprint.set_attribute('noise_alt_bias', str(sensor_params['noise_alt_bias']))
        blueprint.set_attribute('noise_alt_stddev', str(sensor_params['noise_alt_stddev']))
        blueprint.set_attribute('noise_lat_bias', str(sensor_params['noise_lat_bias']))
        blueprint.set_attribute('noise_lat_stddev', str(sensor_params['noise_lat_stddev']))
        blueprint.set_attribute('noise_lon_bias', str(sensor_params['noise_lon_bias']))
        blueprint.set_attribute('noise_lon_stddev', str(sensor_params['noise_lon_stddev']))
        blueprint.set_attribute('noise_seed', str(int(sensor_params['noise_seed'])))
        blueprint.set_attribute('sensor_tick', str(sensor_params['sensor_tick']))

        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)

        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))


    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()


    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS listener method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        self.alt = event.altitude


    def get_observation(self):
        """ Get tuple of current sensor observation"""
        return (self.lat, self.lon, self.alt)


    def get_location(self):
        """ Get the CARLA x/y Location based on the current sensor observation"""
        (x, y, z) = GnssSensor.gps_to_location(self.lat, self.lon, self.alt)
        return carla.Location(x=x, y=y, z=z)


    @staticmethod
    def gps_to_location(latitude, longitude, altitude):
        """Creates Location from GPS (latitude, longitude, altitude).
        This is the inverse of the _location_to_gps method found in:
            https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py
        Taken from the `pylot` project function `from_gps`:
            https://github.com/erdos-project/pylot/blob/master/pylot/utils.py
        """
        EARTH_RADIUS_EQUA = 6378137.0
        # The following reference values are applicable for towns 1 through 7,
        # and are taken from the corresponding OpenDrive map files.
        # LAT_REF = 49.0
        # LON_REF = 8.0
        LAT_REF = 0.0
        LON_REF = 0.0

        scale = math.cos(LAT_REF * math.pi / 180.0)
        basex = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * LON_REF
        basey = scale * EARTH_RADIUS_EQUA * math.log(
            math.tan((90.0 + LAT_REF) * math.pi / 360.0))

        x = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * longitude - basex
        y = scale * EARTH_RADIUS_EQUA * math.log(
            math.tan((90.0 + latitude) * math.pi / 360.0)) - basey

        # This wasn't in the original method, but seems to be necessary.
        y *= -1

        return (x, y, altitude)
