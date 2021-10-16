using ObservationModels
using AutomotiveSimulator
using AutomotiveVisualization
using AdversarialDriving
using Distributions
using Parameters
using Random

##############################################################################
"""
Generate Roadway and scene
"""
## Geometry parameters
roadway_length = 100.

roadway = gen_straight_roadway(3, roadway_length) # lanes and length (meters)

init_noise = Noise(pos = (0, 0), vel = 0)

scene = Scene([
	Entity(BlinkerState(VehicleState(VecSE2(10.0, 6.0, 0), roadway, 10.0), false, [], init_noise), VehicleDef(), 1),
	Entity(BlinkerState(VehicleState(VecSE2(30.0, 3.0, 0), roadway, 0.0), false, [], init_noise), VehicleDef(), 2)
]);

####################################################################
"""
Generate satellites and buildings
"""

fixed_sats = [
    ObservationModels.Satellite(pos=VecE3(-1e7, 1e7, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(1e7, 1e7, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(-1e7, 0.0, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(100.0, 0.0, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(1e7, 0.0, 1e7), clk_bias=0.0)
        ]

buildingmap = BuildingMap()

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
    scene = model.noisy_observations ? noisy_scene!(input_scene, roadway, egoid) : input_scene

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
Function for generating noisy entities based on GPS and range-bearing sensor observations
"""

# Makes a copy of the scene with the noise added to the vehicles in the state
function noisy_scene!(scene::Scene, roadway::Roadway, egoid::Int64)
    noisy_scene = Scene(Entity)
    ego = scene[egoid]
    # Update SUT with ego-localization noise
    ObservationModels.update_gps_noise!(ego, scene, buildingmap, fixed_sats)
    # Update other agents with range and bearing noise
    ObservationModels.update_rb_noise!(ego, scene)
    for (i,veh) in enumerate(scene)
        push!(noisy_scene, noisy_entity(veh, roadway))
    end
    noisy_scene
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

####################################################################
"""
Define driving models
"""

models = Dict{Int, DriverModel}(
	1 => UrbanIDM(idm=IntelligentDriverModel(v_des = 15.), noisy_observations=true),
	2 => UrbanIDM(idm=IntelligentDriverModel(v_des = 0.), noisy_observations=false)
)

##############################################################################
"""
Run scene simulations
"""
scenes = Vector{typeof(scene)}()
for i=1:10
    append!(scenes, simulate(scene, roadway, models, 30, 1.0));
end

##############################################################################
"""
Train MDN
"""

feat, y = preprocess_data(1, scenes);

params = MDNParams(batch_size=2, lr=1e-3);

net = construct_mdn(params);

train_nnet!(feat, y, net..., params);
