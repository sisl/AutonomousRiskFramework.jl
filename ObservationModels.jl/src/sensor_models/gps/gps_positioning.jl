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