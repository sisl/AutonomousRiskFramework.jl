"""
RangeAndBearingMeasurement <: SensorObservation
defines a range and bearing measurements with noise
"""
@with_kw struct RangeAndBearingMeasurement{T} <: SensorObservation
    range::T = 0.0    # positive
    bearing::T = 0.0  # [-π/2, π/2]
    range_noise::T = 0.0
    bearing_noise::T = 0.0
end

noise(m::RangeAndBearingMeasurement{T}) where T = T[m.range_noise, m.bearing_noise]
