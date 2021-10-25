"""
RB_fix(range_bearing::RangeAndBearingMeasurement)

estimate relative position from range-bearing measurement
"""
function RB_fix(range_bearing::RangeAndBearingMeasurement)
    range = range_bearing.range + range_bearing.range_noise
    bearing = range_bearing.bearing + range_bearing.bearing_noise
    return [range*cos(bearing), range*sin(bearing)]
end
