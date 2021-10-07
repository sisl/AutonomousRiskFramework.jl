####################################################################
"""
Urban Intelligent Driving Model
"""

# Define a driving model for intelligent Urban driving
@with_kw mutable struct UrbanIDM <: DriverModel{BlinkerVehicleControl}
    idm = IntelligentDriverModel(v_des = 15.) # underlying idm
    noisy_observations::Bool = false # Whether or not this model gets noisy observations
    ignore_idm::Bool = false

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
    scene = model.noisy_observations ? noisy_scene!(input_scene, roadway, egoid, model.ignore_idm) : input_scene

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


##############################################################################
"""
Define Agents and MDP
"""
# Construct a regular Blinker vehicle agent
function AdversarialDriving.BlinkerVehicleAgent(get_veh::Function, model::UrbanIDM;
    entity_dim = BLINKERVEHICLE_ENTITY_DIM,
    disturbance_dim=BLINKERVEHICLE_DISTURBANCE_DIM,
    entity_to_vec = BlinkerVehicle_to_vec,
    disturbance_to_vec = BlinkerVehicleControl_to_vec,
    vec_to_entity = vec_to_BlinkerVehicle,
    vec_to_disturbance = vec_to_BlinkerVehicleControl,
    disturbance_model = get_bv_actions())
    Agent(get_veh, model, entity_dim, disturbance_dim, entity_to_vec,
          disturbance_to_vec,  vec_to_entity, vec_to_disturbance, disturbance_model)
end
