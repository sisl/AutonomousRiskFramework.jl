using Pkg
Pkg.activate("ast_obs")
using AutomotiveSimulator, AutomotiveVisualization
using AdversarialDriving
using Printf
using InteractiveUtils
using Interact
using WebIO
using Blink
using ElectronDisplay
using Cairo
using Revise
using ObservationModels
using Parameters
using Random

## Geometry parameters
roadway_length = 500.

####################################################################
"""
Building{T}
"""

mutable struct Building{T<:Real}
    id::Int64
    width1::Float64 # [m]
    width2::Float64 # [m]
    pos::VecSE2{T}
end

Base.show(io::IO, b::Building) = @printf(io, "Building({%.3f, %.3f, %.3f}, %.3f, %.3f)", b.pos.x, b.pos.y, b.pos.θ, b.width1, b.width2)

####################################################################
"""
BuildingMap{T}
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

####################################################################
"""
Generate Buildings
"""

function gen_my_buildings(nbuildings::Int, top::Float64, bottom::Float64)
    buildingmap = BuildingMap()
    id = 1

    # Top buildings
    for i = 1:nbuildings
            origin::VecSE2{Float64} = VecSE2(25.0 + (i-1)*13, top + 1.0, 0.0)
            push!(buildingmap.buildings, Building(id, 10.0, 3.0, origin))
            id += 1
    end

    # Bottom buildings
    for i = 1:nbuildings
        origin::VecSE2{Float64} = VecSE2(25.0 + (i-1)*13, bottom - 1.0, 0.0)
        push!(buildingmap.buildings, Building(id, 10.0, 3.0, origin))
        id += 1
    end

    return buildingmap
end

buildingmap = gen_my_buildings(7, 9.0, -3.0) # num buildings, top and bottom points (meters)

####################################################################
"""
Generate Roadway and scene
"""

roadway = gen_straight_roadway(3, roadway_length) # lanes and length (meters)

init_noise = Noise(pos = (0, 0), vel = 0)

scene = Scene([
	Entity(BlinkerState(VehicleState(VecSE2(10.0, 6.0, 0), roadway, 10.0), false, [], init_noise), VehicleDef(), 1),
	Entity(BlinkerState(VehicleState(VecSE2(30.0, 3.0, 0), roadway, 0.0), false, [], init_noise), VehicleDef(), 2)
]);


####################################################################
"""
Generate satellites
"""

fixed_sats = [
    ObservationModels.Satellite(pos=VecE3(-1e7, 1e7, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(1e7, 1e7, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(-1e7, 0.0, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(100.0, 0.0, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(1e7, 0.0, 1e7), clk_bias=0.0)
        ]

####################################################################
"""
Urban Intelligent Driving Model
"""

# Define a driving model for intelligent Urban driving
@with_kw mutable struct UrbanIDM <: DriverModel{BlinkerVehicleControl}
    idm::IntelligentDriverModel = IntelligentDriverModel(v_des = 15.) # underlying idm
    noisy_observations::Bool = false # Whether or not this model gets noisy observations
    
    next_action::BlinkerVehicleControl = BlinkerVehicleControl() # The next action that the model will do (for controllable vehicles)

    # Describes rules of the road
end

# Sample an action from UrbanIDM model
function Base.rand(rng::AbstractRNG, model::UrbanIDM)
    na = model.next_action
    BlinkerVehicleControl(model.idm.a, na.da, na.toggle_goal, na.toggle_blinker, na.noise)
 end

# Observe function for UrbanIDM
function AutomotiveSimulator.observe!(model::UrbanIDM, input_scene::Scene, roadway::Roadway, egoid::Int64)
    # If this model is susceptible to noisy observations, adjust all the agents by noise
    scene = model.noisy_observations ? noisy_scene!(input_scene, roadway) : input_scene

    # Get the ego and the ego lane
    ego = get_by_id(scene, egoid)
    ego_v = vel(ego)
    li = laneid(ego)

    # Get headway to the forward car
    fore = find_neighbor(scene, roadway, ego, targetpoint_ego = VehicleTargetPointFront(), targetpoint_neighbor = VehicleTargetPointRear())
    fore_v, fore_Δs = isnothing(fore.ind) ? (NaN, Inf) : (vel(scene[fore.ind]), fore.Δs)

    next_idm = track_longitudinal!(model.idm, ego_v, fore_v, fore_Δs)
    model.idm = next_idm
    model
end

####################################################################
"""
Function for generating a noisy entity from the original entity and noise
"""

# Makes a copy of the scene with the noise added to the vehicles in the state
function noisy_scene!(scene::Scene, roadway::Roadway)
    noisy_scene = Scene(Entity)
    # Update SUT with ego-localization noise
    ObservationModels.update_gps_noise!(scene, buildingmap, fixed_sats)
    # Update other agents with range and bearing noise
    ObservationModels.update_rb_noise!(scene)
    for (i,veh) in enumerate(scene)
        push!(noisy_scene, noisy_entity(veh, roadway))
    end
    noisy_scene
end

function noisy_entity(ent, roadway::Roadway)
    f = posf(ent)
    
    # Noise from GPS
    # ent_pos = posg(ent)
    # ranges = ObservationModels.measure_gps(Float32[ent_pos.x, ent_pos.y], zeros(Float32, length(fixed_sats)), buildingmap, fixed_sats)
    # gps_ent_pos = ObservationModels.GPS_fix(ranges)
    # Δs = gps_ent_pos[1] - ent_pos[1]
    # Δt = gps_ent_pos[2] - ent_pos[2]
    
    # Controllable noise
    Δs, Δt = noise(ent).pos
    
    noisy_f = Frenet(roadway[f.roadind.tag], f.s + Δs, f.t + Δt, f.ϕ)
    noisy_g = posg(noisy_f, roadway)
    noisy_v = vel(ent) + noise(ent).vel
    noisy_vs = VehicleState(noisy_g, noisy_f, noisy_v)
    Entity(update_veh_state(ent.state, noisy_vs), ent.def, ent.id)
end

####################################################################
"""
Render instructions for Noisy Vehicle
"""

function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, veh::Entity{BlinkerState, VehicleDef, Int64})
    reg_veh = Entity(veh.state.veh_state, veh.def, veh.id)
    add_renderable!(rendermodel, FancyCar(car=reg_veh))

    noisy_veh = Entity(noisy_entity(veh, roadway).state.veh_state, veh.def, veh.id)
    ghost_color = weighted_color_mean(0.3, colorant"blue", colorant"white")
    add_renderable!(rendermodel, FancyCar(car=noisy_veh, color=ghost_color))

    li = laneid(veh)
    bo = BlinkerOverlay(on = blinker(veh), veh = reg_veh, right=Tint_signal_right[li])
    add_renderable!(rendermodel, bo)
    return rendermodel
end

# function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, veh::Entity{VehicleState, VehicleDef, Int64})
#     add_renderable!(rendermodel, FancyCar(car=veh))
#     return rendermodel
# end

####################################################################
"""
Run Code
"""

models = Dict{Int, DriverModel}(
	1 => UrbanIDM(idm=IntelligentDriverModel(v_des = 15.), noisy_observations=true),
	2 => UrbanIDM(idm=IntelligentDriverModel(v_des = 0.), noisy_observations=false)
)

scenes = simulate(scene, roadway, models, 30, 1.0);

win = Blink.Window()

man = @manipulate for t=slider(1:length(scenes), value=1., label="t")
    camera = StaticCamera(position=VecE2(posg(scenes[t][1]).x, 3.0), zoom=7);
    render([buildingmap, roadway, scenes[t]], camera=camera)
end;

body!(win, man)