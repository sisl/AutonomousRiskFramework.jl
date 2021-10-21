import math
import carla
import weakref
from collections import defaultdict


class ASTSensorWrapper(object):
    """
    Class for wrapping CARLA sensors and applying AST noise disturbances.

    Inputs:
        - sensor type (e.g., GNSS, Radar, etc.)
        - sensor parameters (e.g., noise_lat_stddev)
    Handles:
        - creating sensors
        - converting from sensor output to CARLA x/y location
    Outputs:
        - noisy x/y location
    """


    def __init__(self, parent_actor, sensor_type='gnss', sensor_params=None):
        self.sensor = None
        self.sensor_type = sensor_type
        self.current_location = None

        if self.sensor_type == 'gnss':
            self.sensor = GnssSensor(parent_actor, sensor_params)
        else:
            raise Exception("(ASTSensorWrapper) No sensor named:" + self.sensor_type)


    def get_location(self):
        """ Convert from sensor observation to CARLA Location."""
        return self.sensor.get_location()


    def get_observation(self):
        """ Get current sensor observation"""
        return self.sensor.get_observation()


    def apply_disturbance(self, disturbance={'x': 0, 'y': 0}):
        """ Apply noise disturbance to sensor observations"""
        location_from_sensor = self.get_location()
        noise = carla.Location(x=disturbance['x'], y=disturbance['y'])
        noisy_location = location_from_sensor + noise
        return noisy_location


    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()




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
