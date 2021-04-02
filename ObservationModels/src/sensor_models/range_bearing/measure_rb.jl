"""
measure_range_bearing(pos1::VecSE2{T}, pos2::VecSE2{T}, noise::Array{T}) where T

Function to compute a range-bearing measurement from given positions and noise
"""
function measure_range_bearing(pos1::VecSE2{T}, pos2::VecSE2{T}, noise::Array{T}) where T
    pos_vec1::VecE2{T} = VecE2(pos1.x, pos1.y)
    pos_vec2::VecE2{T} = VecE2(pos2.x, pos2.y)

    true_range = dist(pos_vec1, pos_vec2)
    true_bearing = π/2 - pos1.θ + atan(pos_vec2-pos_vec1)

    return RangeAndBearingMeasurement(range=true_range, bearing=true_bearing, range_noise=noise[1], bearing_noise=noise[2])
end