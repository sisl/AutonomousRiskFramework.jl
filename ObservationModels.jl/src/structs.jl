"""
Abstract type for defining landmarks. Consists of `position` method
"""
abstract type Landmark end

position(def::Landmark) = error("position not implemented for landmark of type $(typeof(def))")

"""
Abstract type for defining sensor observations. Consists of `noise` method
"""
abstract type SensorObservation end

noise(def::SensorObservation) = error("noise not implemented for sensor observation of type $(typeof(def))")

"""
Building{T<:Real} <: Landmark

Struct for a building as a Landmark in an environment
"""
mutable struct Building{T<:Real} <: Landmark
    id::Int64
    width1::T # [m]
    width2::T # [m]
    pos::VecSE2{T}
end

Base.show(io::IO, b::Building) = @printf(io, "Building({%.3f, %.3f, %.3f}, %.3f, %.3f)", b.pos.x, b.pos.y, b.pos.θ, b.width1, b.width2)

position(b::Building) = VecE3(b.pos.x, b.pos.y, 0.0)

"""
BuildingMap{T<:Real}

struct for collection of `Building` landmarks. Contains rendering methods for `AutomotiveVisualization`
"""
mutable struct BuildingMap{T<:Real}
    buildings::Vector{Building{T}}
end

BuildingMap{T}() where T = BuildingMap{T}(Building{T}[]) 
BuildingMap() = BuildingMap{Float64}()

Base.show(io::IO, bmap::BuildingMap) = @printf(io, "BuildingMap(%g)", length(bmap.buildings))

function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, buildingmap::BuildingMap)
    for b in buildingmap.buildings
        x_pos = b.pos.x
        y_pos = b.pos.y
        θ_pos = b.pos.θ
        w1 = b.width1/2.0
        pts = Array{Float64}(undef, 2, 2)
        pts[1, 1] = x_pos + w1*cos(θ_pos)
        pts[2, 1] = y_pos + w1*sin(θ_pos)
        pts[1, 2] = x_pos - w1*cos(θ_pos)
        pts[2, 2] = y_pos - w1*sin(θ_pos)
        add_instruction!(rendermodel, render_line, (pts, colorant"maroon", b.width2, Cairo.CAIRO_LINE_CAP_BUTT))
    end

    return rendermodel
end