"""
    Satellite{T<:Real} <: Landmark

denotes a Satellite object
"""
@with_kw struct Satellite{T<:Real} <: Landmark
    pos::VecE3{T} = VecE3(0.0,0.0,0.0)
    clk_bias::T = 0.0
    visible::Bool = true
end

position(sat::Satellite) = sat.pos

"""
    GPSRangeMeasurement{T<:Real} <: SensorObservation
defines a range measurement and noise associated with a GPS satellite 
"""
@with_kw struct GPSRangeMeasurement{T<:Real} <: SensorObservation
    sat::Satellite{T} = Satellite()
    range::T = 0.0
    noise::T = 0.0
end

noise(m::GPSRangeMeasurement{T}) where T = T[m.noise]
