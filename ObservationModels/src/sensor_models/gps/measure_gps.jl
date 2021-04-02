"""
measure_gps(pos_vec::VecE3{T}, noise::Array{T}, satpos::Vector{Satellite}) where T

function to compute GPS measurements for x,y coordinate pos from satellite positions and noise
"""
function measure_gps(pos_vec::VecE3{T}, noise::Array{T}, satpos::Vector{Satellite{T}}) where T
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

"""
measure_gps(pos_vec::VecE3{T}, noise::Array{T}, buildingmap::BuildingMap{T}, satpos::Vector{Satellite}) where T

function to compute GPS measurements given a building model
"""
function measure_gps(pos_vec::VecE3{T}, noise::Array{T}, buildingmap::BuildingMap{T}, satpos::Vector{Satellite{T}}) where T
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

"""
measure_gps(ent::Entity, noise::Array{T}, satpos::Vector{Satellite}) where T

measure_gps(pos::VecSE2{T}, noise::Array{T}, buildingmap::BuildingMap{T}, satpos::Vector{Satellite}) where T

measure_gps(ent::Entity, noise::Array{T}, buildingmap::BuildingMap{T}, satpos::Vector{Satellite}) where T

compatibility functions
"""
function measure_gps(ent::Entity, noise::Array{T}, satpos::Vector{Satellite{T}}) where T
    pos = posg(ent)
    pos_vec::VecE3{T} = VecE3(pos.x, pos.y, 0.0)
    measure_gps(pos_vec, noise, satpos)
end

function measure_gps(pos::VecSE2{T}, noise::Array{T}, buildingmap::BuildingMap{T}, satpos::Vector{Satellite{T}}) where T
    pos_vec::VecE3{T} = VecE3(pos.x, pos.y, 0.0)
    measure_gps(pos_vec, noise, buildingmap, satpos)
end

function measure_gps(ent::Entity, noise::Array{T}, buildingmap::BuildingMap{T}, satpos::Vector{Satellite{T}}) where T
    pos = posg(ent)
    measure_gps(pos, noise, buildingmap, satpos)
end