####################################################################
"""
Function for generating noisy entities based on GPS and range-bearing sensor observations
"""

# Makes a copy of the scene with the noise added to the vehicles in the state (updates noise in input scene)
function noisy_scene!(scene::Scene, roadway::Roadway, egoid::Int64, ignore_idm::Bool)
    n_scene = Scene(Entity)
    ego = get_by_id(scene, egoid)

    if ignore_idm
        # Update SUT with ego-localization noise
        ObservationModels.update_gps_noise!(ego, scene, buildingmap, fixed_sats)
        # Update other agents with range and bearing noise
        ObservationModels.update_rb_noise!(ego, scene)
    end
    for (i,veh) in enumerate(scene)
        push!(n_scene, noisy_entity(veh, roadway))
    end
    n_scene
end

##############################################################################
"""
Scene stepping for self-noise in SUT
"""

# Step the scene forward by one timestep and return the next state
function AdversarialDriving.step_scene(mdp::AdversarialDriving.AdversarialDrivingMDP, s::Scene, actions::Vector{Disturbance}, rng::AbstractRNG = Random.GLOBAL_RNG)
    entities = []

    # Add noise in SUT
    update_adversary!(sut(mdp), actions[1], s)

    # Loop through the adversaries and apply the instantaneous aspects of their disturbance
    for (adversary, action) in zip(adversaries(mdp), actions[2:end])
        update_adversary!(adversary, action, s)
    end

    # Loop through the vehicles in the scene, apply action and add to next scene
    for (i, veh) in enumerate(s)
        m = model(mdp, veh.id)
        observe!(m, s, mdp.roadway, veh.id)
        a = rand(rng, m)
        bv = Entity(propagate(veh, a, mdp.roadway, mdp.dt), veh.def, veh.id)
        !end_of_road(bv, mdp.roadway, mdp.end_of_road) && push!(entities, bv)
    end
    isempty(entities) ? Scene(typeof(sut(mdp).get_initial_entity())) : Scene([entities...])
end


##############################################################################
"""
Generate Roadway, environment and define Agent functions
"""
## Geometry parameters
roadway_length = 50.

roadway = gen_straight_roadway(3, roadway_length) # lanes and length (meters)

fixed_sats = [
    ObservationModels.Satellite(pos=VecE3(-1e7, 1e7, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(1e7, 1e7, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(-1e7, 0.0, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(100.0, 0.0, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(1e7, 0.0, 1e7), clk_bias=0.0)
        ]

buildingmap = BuildingMap()

get_urban_vehicle_1(;id::Int64, s::Float64, v::Float64, noise::Noise) = (rng::AbstractRNG = Random.GLOBAL_RNG) -> BlinkerVehicle(roadway = roadway,
                                    lane=1, s=s, v = v,
                                    id = id, noise=noise, goals = Int64[],
                                    blinker = false)

get_urban_vehicle_2(;id::Int64, s::Float64, v::Float64, noise::Noise) = (rng::AbstractRNG = Random.GLOBAL_RNG) -> BlinkerVehicle(roadway = roadway,
                                    lane=1, s=s, v = v,
                                    id = id, noise=noise, goals = Int64[],
                                    blinker = false)


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