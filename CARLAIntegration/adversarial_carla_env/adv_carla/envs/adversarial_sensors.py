"""
This module contains all of the adversarial sensor callback classes.
"""

import carla
import numpy as np
from srunner.autoagents.sensor_interface import CallBack
from pdb import set_trace as breakpoint # DEBUG. TODO!


class AdvGNSSCallBack(CallBack):

    """
    Class the sensors listen to in order to receive their data each frame
    """

    def __init__(self, tag, sensor_type, sensor, data_provider):
        """
        Initializes the call back
        """
        super().__init__(tag, sensor_type, sensor, data_provider)
        self.noise_lat = 0
        self.noise_lon = 0
        self.noise_alt = 0
        self.dims = 3


    def __call__(self, data):
        """
        call function
        """
        assert isinstance(data, carla.GnssMeasurement)
        array = np.array([data.latitude,
                          data.longitude,
                          data.altitude], dtype=np.float64)

        noise = np.array([self.noise_lat, self.noise_lon, self.noise_alt])
        array += noise

        self._data_provider.update_sensor(self._tag, array, data.frame)


    def set_disturbance(self, disturbance, offset=0):
        self.noise_lat = disturbance[offset]
        self.noise_lon = disturbance[offset+1]
        self.noise_alt = disturbance[offset+2]



class AdvObstacleCallBack(CallBack):

    """
    Class the sensors listen to in order to receive their data each frame
    """

    def __init__(self, tag, sensor_type, sensor, data_provider):
        """
        Initializes the call back
        """
        super().__init__(tag, sensor_type, sensor, data_provider)
        self._tag = tag
        self._data_provider = data_provider
        self.noise_distance = 0
        self.dims = 1


    def __call__(self, data):
        """
        call function
        """
        # assert isinstance(data, carla.GnssMeasurement)
        # array = np.array([data.latitude,
        #                   data.longitude,
        #                   data.altitude], dtype=np.float64)

        # noise = np.array([self.noise_lat, self.noise_lon, self.noise_alt])
        # array += noise

        if data.distance != 0:
            # print("=--"*20)
            # print("=--"*20)
            # print("=--"*20)
            print("Obstacle data distance:", data.distance)
            # print("=--"*20)
            # print("=--"*20)
            # print("=--"*20)

        array = np.array([data.distance], dtype=np.float64)
        self._data_provider.update_sensor(self._tag, array, data.frame)


    def set_disturbance(self, disturbance, offset=0):
        self.noise_distance = disturbance[offset]



class AdvCollisionCallBack(CallBack):

    """
    Class the sensors listen to in order to receive their data each frame
    """

    def __init__(self, tag, sensor_type, sensor, data_provider):
        """
        Initializes the call back
        """
        super().__init__(tag, sensor_type, sensor, data_provider)
        self.noise_normal_impulse = 0
        self.dims = 1


    def __call__(self, data):
        """
        call function
        """
        # assert isinstance(data, carla.GnssMeasurement)
        print("TYPE:", type(data))
        print("Collision data:", data.normal_impulse)
        # array = np.array([data.normal_impulse], dtype=np.float64)
        array = np.array([1], dtype=np.float64) # True == collision == 1

        noise = np.array([self.noise_normal_impulse])
        array += noise

        self._data_provider.update_sensor(self._tag, array, data.frame)


    def set_disturbance(self, disturbance, offset=0):
        self.noise_normal_impulse = disturbance[offset]
