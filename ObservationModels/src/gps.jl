## Definitions of GPS sensor models and state estimation functions

# Struct that contains satellite positions (in NED w/ height relative to ground) and clock offsets
@with_kw struct Satellite <: Landmark
    pos::VecE3 = VecE3(0,0,0)
    clk_bias::Float64 = 0
    visible::Bool = true
end

# Struct for defining range measurements and noise from GPS satellites 
@with_kw struct GPSRangeMeasurement <: SensorObservation
    sat::Satellite = Satellite()
    range::Float64 = 0.0
    noise::Float64 = 0.0
end

# Function to compute GPS measurements for x,y coordinate pos from satellite positions and noise
function measure_gps(pos_vec::VecE3{T}, noise::Array{T}, satpos::Vector{Satellite}) where T
    ranges = Union{Missing, GPSRangeMeasurement}[]
    n_sats = length(satpos)
    for i in 1:n_sats
        if satpos[i].visible==true
            push!(ranges, GPSRangeMeasurement(sat=satpos[i], range=dist(pos_vec, satpos[i].pos), noise=noise[i]))
        else
            push!(ranges, missing)
        end
    end
    ranges
end

# Function to compute GPS measurements for x,y coordinate pos from satellite positions and noise
function measure_gps(pos_vec::VecE3{T}, noise::Array{T}, buildingmap::Any, satpos::Vector{Satellite}) where T
    # Visibility based on building model
    
    # Sat LOS vector
    n_sats = length(satpos)
    los_angs = Array{T}(undef, n_sats)
    for i in 1:n_sats
        los_angs[i] = atan(satpos[i].pos.y - pos_vec.y, satpos[i].pos.x - pos_vec.x)
    end

    # Building vectors
    n_build = length(buildingmap.buildings)
    build_thresh = Array{T}(undef, 3, n_build)
    for i in 1:n_build
        b = buildingmap.buildings[i]
        build_bound1::VecE3{T} = VecE3(b.pos.x + b.width1/2.0*cos(b.pos.θ), b.pos.y + b.width1/2.0*sin(b.pos.θ), 0.0)
        build_bound2::VecE3{T} = VecE3(b.pos.x - b.width1/2.0*cos(b.pos.θ), b.pos.y - b.width1/2.0*sin(b.pos.θ), 0.0)
        build_cent::VecE3{T} = VecE3(b.pos.x, b.pos.y, 0.0)
        build_thresh[1, i] = atan(build_bound1.y - pos_vec.y, build_bound1.x - pos_vec.x)
        build_thresh[2, i] = atan(build_bound2.y - pos_vec.y, build_bound2.x - pos_vec.x)
        build_thresh[3, i] = dist(build_cent, pos_vec)
    end

    # Detect Visibility
    visibility = trues(n_sats)
    for i in 1:n_sats
        for j in 1:n_build
            h_in = (build_thresh[1, j] - los_angs[i])*(build_thresh[2, j] - los_angs[i]) < 0.0
            too_far = build_thresh[3, j] > 30.0
            if h_in && !too_far
                visibility[i] = false
                noise[i] += 3.0    # bias error from signal reflections
                break
            end
        end
    end

    # Generate measurements
    measure_gps(pos_vec, noise, satpos)
end

# Compatibility functions for passing entities instead of position
function measure_gps(ent::Entity, noise::Array{T}, satpos::Vector{Satellite}) where T
    pos = posg(ent)
    pos_vec::VecE3{T} = VecE3(pos.x, pos.y, 0.0)
    measure_gps(pos_vec, noise, satpos)
end

# Compatibility functions for passing entity positions instead of position vector
function measure_gps(pos::VecSE2{T}, noise::Array{T}, buildingmap::Any, satpos::Vector{Satellite}) where T
    pos_vec::VecE3{T} = VecE3(pos.x, pos.y, 0.0)
    measure_gps(pos_vec, noise, buildingmap, satpos)
end

# Compatibility functions for passing entities instead of position
function measure_gps(ent::Entity, noise::Array{T}, buildingmap::Any, satpos::Vector{Satellite}) where T
    pos = posg(ent)
    measure_gps(pos, noise, buildingmap, satpos)
end

# Determine position fix using GPS measurements from multiple satellites 
function GPS_fix(meas::Array{T}) where {T <: Union{Missing, GPSRangeMeasurement}}
    lam = 1.0  # stepsize
    x = [0.0, 0.0, 0.0, 0.0]
    
    meas = collect(skipmissing(meas))
    
    if length(meas) < 4
        println("Too few measurements!")
    end
    
    function f(x, meas)
        delta_prange = zeros(length(meas), 1)
        for i=1:length(meas)
            satpos = meas[i].sat.pos
            satclk = meas[i].sat.clk_bias
            expected_range = hypot(x[1]-satpos[1], x[2]-satpos[2], x[3]-satpos[3])
            expected_prange = expected_range + x[4] - satclk
            delta_prange[i] = meas[i].range + meas[i].noise - expected_prange
        end
        delta_prange
    end
    
    function df(x, meas)
        jacobian = zeros(length(meas), 4)
        for i=1:length(meas)
            satpos = meas[i].sat.pos
            satclk = meas[i].sat.clk_bias
            expected_range = hypot(x[1]-satpos[1], x[2]-satpos[2], x[3]-satpos[3])
            expected_prange = expected_range + x[4] - satclk
            
            jacobian[i, 1] = -(x[1]-satpos[1])/expected_range
            jacobian[i, 2] = -(x[2]-satpos[2])/expected_range
            jacobian[i, 3] = -(x[3]-satpos[3])/expected_range
            jacobian[i, 4] = -1
        end
        jacobian
    end
    
    delta_x = [1000., 1000., 1000., 1000.]
    
#     while sum(abs.(delta_x)) > 0.001
    for i=1:5
        delta_x = lam*(pinv(df(x, meas))*f(x, meas))
        x = x - delta_x
    end 
    x
end

# Update entity with id 1 (SUT) in scene with GPS noise
function update_gps_noise!(scene::Scene, buildingmap::Any, satpos::Vector{Satellite})
    # Predetermined constants
    range_noise = 0.2

    ent = scene[1]
    ent_pos = posg(ent)
    ranges = measure_gps(ent_pos, randn(length(satpos))*range_noise, buildingmap, satpos)
    gps_ent_pos = GPS_fix(ranges)
    Δs = gps_ent_pos[1] - ent_pos[1]
    Δt = gps_ent_pos[2] - ent_pos[2]
    noise = Noise(pos=VecE2(Δs, Δt))
    scene[1] = Entity(BlinkerState(veh_state=ent.state.veh_state, blinker=ent.state.blinker, goals=ent.state.goals, noise=noise), ent.def, ent.id)
end