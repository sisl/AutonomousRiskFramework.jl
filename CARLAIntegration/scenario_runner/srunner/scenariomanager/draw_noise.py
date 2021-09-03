import carla

def draw_noise(world,vehicle,noise):
    transform = vehicle.get_transform()
    bounding_box = vehicle.bounding_box
    bounding_box.location = transform.location
    bounding_box.location += noise
    # bounding_box.extent = carla.Vector3D(x=1,y=1,z=1)
    # print(transform, bounding_box.location)
    world.debug.draw_box(bounding_box, transform.rotation, life_time=0.2)
