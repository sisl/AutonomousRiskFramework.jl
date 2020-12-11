## Definitions of sensor models and state estimation functions

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

# Function to compute GPS measurements for an entity from satellite positions and noise
function measure_gps(ent::Entity, noise::Array{Float64})
    # TODO: move this somewhere outside
    fixed_sat = [
        Satellite(pos=VecE3(-1000.0,-1000.0,1000.0), clk_bias=0.0),
        Satellite(pos=VecE3(1000.0,-1000.0,1000.0), clk_bias=0.0),
        Satellite(pos=VecE3(-1000.0,1000.0,1000.0), clk_bias=0.0),
        Satellite(pos=VecE3(1000.0,1000.0,1000.0), clk_bias=0.0),
        Satellite(pos=VecE3(0.0,0.0,1000.0), clk_bias=0.0)
    ]
    
    ent_pos = posg(ent)
    ranges = Union{Missing, GPSRangeMeasurement}[]
    for i in 1:length(fixed_sat)
        satpos = fixed_sat[i].pos
        if fixed_sat[i].visible==true
            range = hypot(ent_pos.x - satpos.x, ent_pos.y - satpos.y, satpos.z)
            push!(ranges, GPSRangeMeasurement(sat=fixed_sat[i], range=range, noise=noise[i]))
        else
            push!(ranges, missing)
        end
    end
    ranges
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
    for i=1:3
        delta_x = lam*(pinv(df(x, meas))*f(x, meas))
        x = x - delta_x
    end 
    x
end