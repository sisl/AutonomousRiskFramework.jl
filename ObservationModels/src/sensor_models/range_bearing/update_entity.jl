"""
Update an entity in scene with Range and bearing noise
"""
function update_rb_noise!(ent::Entity, scene::Scene)
    # Predetermined constants
    range_factor = 0.1
    range_cap = 1.5
    range_min = 0.1
    bearing_factor = 0.1
    bearing_cap = 0.3
    bearing_min = 0.005

    ent1_pos = posg(ent)

    for ent2 in scene
        if ent2.id==ent.id
            continue
        end
        ent2_pos = posg(ent2)
        base_measurement = measure_range_bearing(ent1_pos, ent2_pos, [0.0, 0.0])   # Non-noisy measurement
        noise_range = max(min(range_factor*base_measurement.range, range_cap), range_min)
        noise_bearing = max(min(bearing_factor*abs(base_measurement.bearing), bearing_cap), bearing_min)
        range_bearing = RangeAndBearingMeasurement(range=base_measurement.range, bearing=base_measurement.bearing, range_noise=randn()*noise_range, bearing_noise=randn()*noise_bearing)
        
        rel_pos = RB_fix(range_bearing)
        abs_y = -1*rel_pos[1]*cos(ent1_pos.θ) - rel_pos[2]*sin(ent1_pos.θ)
        abs_x = -1*rel_pos[1]*sin(ent1_pos.θ) + rel_pos[2]*cos(ent1_pos.θ)

        Δs = abs_x - (ent2_pos[1] - ent.state.noise.pos[1] - ent1_pos[1])
        Δt = abs_y - (ent2_pos[2] - ent.state.noise.pos[2] - ent1_pos[2])
        noise = Noise(pos=VecE2(Δs, Δt))

        scene[ent2.id] = Entity(update_veh_noise(ent2.state, noise), ent2.def, ent2.id) 
    end    
end

"""
Noise in position from Range and bearing noise
"""
function estimate_rb_noise(ent::Entity, ent2::Entity, scene::Scene, noise::Array{T}) where T
    ent1_pos = posg(ent)

    ent2_pos = posg(ent2)
    range_bearing = measure_range_bearing(ent1_pos, ent2_pos, noise) 
        
    rel_pos = RB_fix(range_bearing)
    abs_y = -1*rel_pos[1]*cos(ent1_pos.θ) - rel_pos[2]*sin(ent1_pos.θ)
    abs_x = -1*rel_pos[1]*sin(ent1_pos.θ) + rel_pos[2]*cos(ent1_pos.θ)

    Δs = abs_x - (ent2_pos[1] - ent.state.noise.pos[1] - ent1_pos[1])
    Δt = abs_y - (ent2_pos[2] - ent.state.noise.pos[2] - ent1_pos[2])
    noise = Noise(pos=VecE2(Δs, Δt))

    noise
end