import carla

def draw_noise(world,vehicle,noise):
    transform = vehicle.get_transform()
    bounding_box = check_modify_bb_dimensions(vehicle.bounding_box)
    bounding_box.location = transform.location
    bounding_box.location += noise
    # bounding_box.extent = carla.Vector3D(x=1,y=1,z=1)
    # print(transform, bounding_box.location)
    world.debug.draw_box(bounding_box, transform.rotation, life_time=0.1)

def check_modify_bb_dimensions(bb):
    if bb.extent.x<0.01:
        bb.extent.x = 0.5
    if bb.extent.y<0.01:
        bb.extent.y = 0.5
    if bb.extent.z<0.01:
        bb.extent.z = 0.5
    
    return bb
