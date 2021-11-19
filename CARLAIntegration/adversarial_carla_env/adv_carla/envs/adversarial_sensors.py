"""
This module contains all of the adversarial sensor callback classes.
"""

import copy
import carla
import numpy as np
from srunner.autoagents.sensor_interface import CallBack
from pdb import set_trace as breakpoint # DEBUG. TODO!

from .camera_artifacts import *


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

        array = np.array([data.distance], dtype=np.float64)
        noise = np.array([self.noise_distance])
        array += noise

        if data.distance != 0:
            print("Obstacle data distance (truth | noisy):", data.distance, "|", array[0])

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


class AdvCameraCallBack(CallBack):

    """
    Class the sensors listen to in order to receive their data each frame
    """

    def __init__(self, tag, sensor_type, sensor, data_provider):
        """
        Initializes the call back
        """
        super().__init__(tag, sensor_type, sensor, data_provider)
        self.simulated_camera = None
        self.dynamic_noise_std = 0
        self.exposure_compensation = 0
        self.save_image = True
        self.counter = 0
        self.dims = 2


    def __call__(self, data):
        """
        call function
        """
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (data.height, data.width, 4))

        if self.simulated_camera is None:
            self.simulated_camera = SimulatedCamera(array, exposure_compensation=self.exposure_compensation)

        if self.save_image:
            save_camera_image(array, "./images/ego/" + str(self.counter) + "_before_image.png")

        # Apply disturbance(s)
        self.simulated_camera.dynamic_noise_std = (self.dynamic_noise_std, self.dynamic_noise_std, self.dynamic_noise_std)
        # self.simulated_camera.exposure_compensation = self.exposure_compensation # TODO: This is part of the create_static_noise
        array = self.simulated_camera.simulate(array)

        if self.save_image:
            save_camera_image(array, "./images/ego/" + str(self.counter) + "_after_image.png")

        self._data_provider.update_sensor(self._tag, array, data.frame)
        self.counter += 1


    def set_disturbance(self, disturbance, offset=0):
        self.dynamic_noise_std = disturbance[offset]
        self.exposure_compensation = disturbance[offset+1]
