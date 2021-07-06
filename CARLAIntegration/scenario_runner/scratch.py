import carla
import numpy as np
import pandas
from scipy import interpolate

import argparse
import time
import random 
import weakref

client = carla.Client('127.0.0.1', 2222)
client.set_timeout(10.0)

try:
    world = client.get_world() 
    # for _a in world.get_actors():
    #     _a.destroy()
    weather = carla.WeatherParameters(
                cloudiness=100.0,
                precipitation=100.0,
                precipitation_deposits=0.0,
                wind_intensity=100.0,
                sun_azimuth_angle=130.0,
                sun_altitude_angle=68.0)

    world.set_weather(weather)

    position = carla.Vector3D(
        -35,
        0.0,
        0.5
    )
    rotation = carla.Rotation(0.0, 0.0, 0.0)

    blueprints = world.get_blueprint_library().filter(
        'vehicle.' + 'lincoln' + '.*'
    )

    blueprint = random.choice(blueprints)

    color = random.choice(
        blueprint.get_attribute('color').recommended_values
    )

    blueprint.set_attribute('color', color)

    blueprint.set_attribute('role_name', 'autopilot')

    transform = carla.Transform(position, rotation)

    actor = world.try_spawn_actor(blueprint, transform)
    actor.set_autopilot()
    actor.set_simulate_physics(True)

    # Find Trigger Friction Blueprint
    friction_bp = world.get_blueprint_library().find('static.trigger.friction')

    extent = carla.Location(700.0, 700.0, 700.0)

    friction_bp.set_attribute('friction', str(0.05))
    friction_bp.set_attribute('extent_x', str(extent.x))
    friction_bp.set_attribute('extent_y', str(extent.y))
    friction_bp.set_attribute('extent_z', str(extent.z))

    # Spawn Trigger Friction
    position = carla.Vector3D(-25, 10.0, 0.5)
    transform = carla.Transform(position, rotation) 
    fric = world.spawn_actor(friction_bp, transform)

    world.debug.draw_box(box=carla.BoundingBox(transform.location, extent * 1e-2), rotation=transform.rotation, life_time=100, thickness=0.5, color=carla.Color(r=255,g=0,b=0))

    blueprint = world.get_blueprint_library().find('sensor.other.gnss')
    gnss = world.spawn_actor(blueprint, carla.Transform(), attach_to=actor)
    world_ref = weakref.ref(world)
    map = world.get_map()
    map_ref = weakref.ref(map)
    
    def callback(event):
        # print(set([l.type for l in map_ref().get_all_landmarks()]))
        print(event)

    gnss.listen(callback)

    while True:
        world_snapshot = world.wait_for_tick()
finally:
    gnss.stop()
    gnss.destroy()
    actor.destroy()
    fric.destroy()

