## Definitions of GPS sensor models and state estimation functions

# Struct for defining range measurements and noise from GPS satellites 
@with_kw struct RangeAndBearingMeasurement <: SensorObservation
    range::Float64 = 0.0    # positive
    bearing::Float64 = 0.0  # [-π/2, π/2]
    range_noise::Float64 = 0.0
    bearing_noise::Float64 = 0.0
end

# Function to compute GPS measurements for x,y coordinate pos from satellite positions and noise
function measure_range_bearing(pos1::VecSE2{Float64}, pos2::VecSE2{Float64}, noise::Array{Float64})
    pos_vec1::VecE2{Float64} = VecE2(pos1.x, pos1.y)
    pos_vec2::VecE2{Float64} = VecE2(pos2.x, pos2.y)

    true_range = dist(pos_vec1, pos_vec2)
    true_bearing = π/2 - pos1.θ + atan(pos_vec2-pos_vec1)

    return RangeAndBearingMeasurement(range=true_range, bearing=true_bearing, range_noise=noise[1], bearing_noise=noise[2])
end

# function to assign noise to measurements based on range/bearing
function measure_range_bearing_noisy(pos1::VecSE2{Float64}, pos2::VecSE2{Float64})
    # Predetermined constants
    range_factor = 0.1
    range_cap = 1.5
    range_min = 0.1
    bearing_factor = 0.1
    bearing_cap = 0.3
    bearing_min = 0.005

    base_measurement = measure_range_bearing(pos1, pos2, [0.0, 0.0])
    noise_range = max(min(range_factor*base_measurement.range, range_cap), range_min)
    noise_bearing = max(min(bearing_factor*abs(base_measurement.bearing), bearing_cap), bearing_min)
    return RangeAndBearingMeasurement(range=base_measurement.range, bearing=base_measurement.bearing, range_noise=randn()*noise_range, bearing_noise=randn()*noise_bearing)
end

# estimate relative position
function RB_fix(range_bearing::RangeAndBearingMeasurement)
    range = range_bearing.range + range_bearing.range_noise
    bearing = range_bearing.bearing + range_bearing.bearing_noise
    return [range*cos(bearing), range*sin(bearing)]
end

# Update entities in scene with Range and bearing noise
function update_rb_noise!(scene::Scene)
    # Predetermined constants
    range_noise = 0.2

    ent1 = scene[1]
    ent1_pos = posg(ent1)

    ent2 = scene[2]
    ent2_pos = posg(ent2)

    range_bearing = measure_range_bearing_noisy(ent1_pos, ent2_pos)
    rel_pos = RB_fix(range_bearing)
    abs_y = -1*rel_pos[1]*cos(ent1_pos.θ) - rel_pos[2]*sin(ent1_pos.θ)
    abs_x = -1*rel_pos[1]*sin(ent1_pos.θ) + rel_pos[2]*cos(ent1_pos.θ)

    Δs = abs_x - (ent2_pos[1] - ent1.state.noise.pos[1] - ent1_pos[1])
    Δt = abs_y - (ent2_pos[2] - ent1.state.noise.pos[2] - ent1_pos[2])
    noise = Noise(pos=VecE2(Δs, Δt))
    scene[2] = Entity(BlinkerState(veh_state=ent2.state.veh_state, blinker=ent2.state.blinker, goals=ent2.state.goals, noise=noise), ent2.def, ent2.id)
end