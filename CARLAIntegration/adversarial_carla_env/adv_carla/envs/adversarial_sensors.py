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

    def __init__(self, tag, sensor, data_provider):
        """
        Initializes the call back
        """
        super().__init__(tag, sensor, data_provider)
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


    def set_disturbance(self, disturbance):
        self.noise_lat = disturbance[0]
        self.noise_lon = disturbance[1]
        self.noise_alt = disturbance[2]



class AdvObstacleCallBack(CallBack):

    """
    Class the sensors listen to in order to receive their data each frame
    """

    def __init__(self, tag, sensor, data_provider):
        """
        Initializes the call back
        """
        super().__init__(tag, sensor, data_provider)
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

        # breakpoint()
        print("Obstacle data:", data)
        print("Obstacle data.distance:", data.distance)
        # print("Obstacle properties:", dir(data))

        array = np.array([0])
        self._data_provider.update_sensor(self._tag, array, data.frame)
        print(self._data_provider._new_data_buffers)


    def set_disturbance(self, disturbance):
        self.noise_distance = disturbance[0]
