using Revise
using ObservationModels
using AutomotiveSimulator
using AutomotiveVisualization
using AdversarialDriving
using POMDPStressTesting
using POMDPs, POMDPPolicies, POMDPSimulators
using Distributions
using Parameters
using Random
using WebIO
using Blink
using InteractiveUtils
using Interact

####################################################################
"""
Urban Intelligent Driving Model
"""

# Define a driving model for intelligent Urban driving
@with_kw mutable struct UrbanIDM <: DriverModel{BlinkerVehicleControl}
    idm::IntelligentDriverModel = IntelligentDriverModel(v_des = 15.) # underlying idm
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

##############################################################################
"""
Create Simulation
"""

@with_kw struct AutoRiskParams
    endtime::Real = 30 # Simulate end time
    ignore_sensors::Bool = true # Simulate sensor observations for agents
end;

@with_kw mutable struct AutoRiskSim <: GrayBox.Simulation
    t::Real = 0 # Current time
    params::AutoRiskParams = AutoRiskParams() # Parameters
    
    # Initial Noise
    init_noise_1 = Noise(pos = (0, 0), vel = 0)
    init_noise_2 = Noise(pos = (0, 0), vel = 0)

    # System under test, ego vehicle
    sut = BlinkerVehicleAgent(get_urban_vehicle_1(id=1, s=5.0, v=15.0, noise=init_noise_1),
    UrbanIDM(idm=IntelligentDriverModel(v_des = 15.), noisy_observations=true, ignore_idm=!params.ignore_sensors));

    # Noisy adversary, vehicle
    adversary = BlinkerVehicleAgent(get_urban_vehicle_2(id=2, s=25.0, v=0.0, noise=init_noise_2),
    UrbanIDM(idm=IntelligentDriverModel(v_des = 0.), noisy_observations=false));

    # Adversarial Markov decision process
    problem::MDP = AdversarialDrivingMDP(sut, [adversary], roadway, 0.1)
    state::Scene = rand(initialstate(problem))
    prev_distance::Real = -Inf # Used when agent goes out of frame

    # Noise distributions and disturbances (consistent with output variables in _logpdf)
    xposition_noise_veh::Distribution = Normal(0, 3) # Gaussian noise (notice larger σ)
    yposition_noise_veh::Distribution = Normal(0, 3) # Gaussian noise
    velocity_noise_veh::Distribution = Normal(0, 1e-4) # Gaussian noise

    xposition_noise_sut::Distribution = Normal(0, 3) # Gaussian noise (notice larger σ)
    yposition_noise_sut::Distribution = Normal(0, 3) # Gaussian noise
    velocity_noise_sut::Distribution = Normal(0, 1e-4) # Gaussian noise
    
    disturbances = Disturbance[BlinkerVehicleControl(), BlinkerVehicleControl()] # Initial 0-noise disturbance

    _logpdf::Function = (sample, state) -> 0 # Function for evaluating logpdf  
end;

##############################################################################
"""
Graybox functions
"""

function GrayBox.environment(sim::AutoRiskSim) 
   return GrayBox.Environment(
                            :vel_veh => sim.velocity_noise_veh,
                            :xpos_veh => sim.xposition_noise_veh,
                            :ypos_veh => sim.yposition_noise_veh,
                            :vel_sut => sim.velocity_noise_sut,
                            :xpos_sut => sim.xposition_noise_sut,
                            :ypos_sut => sim.yposition_noise_sut
                        )
end;

function GrayBox.transition!(sim::AutoRiskSim, sample::GrayBox.EnvironmentSample)
    sim.t += 1 # sim.problem.dt # Keep track of time
    noise_veh = Noise(pos = (sample[:xpos_veh].value, sample[:ypos_veh].value), vel = sample[:vel_veh].value) # reversed to match local pedestrain frame
    noise_sut = Noise(pos = (sample[:xpos_sut].value, sample[:ypos_sut].value), vel = sample[:vel_sut].value)
    sim.disturbances[1] = BlinkerVehicleControl(noise=noise_sut)
    sim.disturbances[2] = BlinkerVehicleControl(noise=noise_veh)

    # step agents: given MDP, current state, and current action (i.e. disturbances)
    (sim.state, r) = @gen(:sp, :r)(sim.problem, sim.state, sim.disturbances)

    # return log-likelihood of actions, summation handled by `logpdf()`
    return sim._logpdf(sample, sim.state)::Real
end

##############################################################################
"""
Blackbox functions
"""

function BlackBox.initialize!(sim::AutoRiskSim)
    sim.t = 0
    sim.problem = AdversarialDrivingMDP(sim.sut, [sim.adversary], roadway, 0.1)
    sim.state = rand(initialstate(sim.problem))
    sim.disturbances = Disturbance[BlinkerVehicleControl(), BlinkerVehicleControl()] # noise-less
    sim.prev_distance = -Inf
end

out_of_frame(sim) = length(sim.state.entities) < 2 # either agent went out of frame

function BlackBox.distance(sim::AutoRiskSim)
    if out_of_frame(sim)
        return sim.prev_distance
    else
        vehicle, sut = sim.state.entities
        pos1 = posg(vehicle)
        pos2 = posg(sut)
        return hypot(pos1.x - pos2.x, pos1.y - pos2.y)
    end
end

function BlackBox.isevent(sim::AutoRiskSim)
    if out_of_frame(sim)
        return false
    else
        vehicle, sut = sim.state.entities
        return collision_checker(vehicle, sut)
    end
end

function BlackBox.isterminal(sim::AutoRiskSim)
    return isterminal(sim.problem, sim.state) ||
           out_of_frame(sim) ||
           BlackBox.isevent(sim) ||
           sim.t ≥ sim.params.endtime
end

function BlackBox.evaluate!(sim::AutoRiskSim, sample::GrayBox.EnvironmentSample)
    logprob::Real = GrayBox.transition!(sim, sample) # Step simulation
    d::Real       = BlackBox.distance(sim)           # Calculate miss distance
    event::Bool   = BlackBox.isevent(sim)            # Check event indication
    sim.prev_distance = d                            # Store previous distance
    return (logprob::Real, d::Real, event::Bool)
end

##############################################################################
"""
Generate initial samples
"""

simx = AutoRiskSim();

# Scenes based on current IDM
BlackBox.initialize!(simx);
d = copy(simx.disturbances)
hr = HistoryRecorder(max_steps=simx.params.endtime)
hist_noise = POMDPSimulators.simulate(hr, simx.problem, FunctionPolicy((s) -> d));
idm_scenes = state_hist(hist_noise);

# Sensor noises in the IDM scenes
scenes = Vector{typeof(simx.state)}() 
for epoch=1:20
    for idm_scene in idm_scenes
        scene = copy(idm_scene)
        n_scene = noisy_scene!(scene, roadway, sutid(simx.problem), true)
        push!(scenes, scene)
    end
end

# # Render Test
# win = Blink.Window()
# man = @manipulate for t=slider(1:length(idm_scenes), value=1., label="t")
#     AutomotiveVisualization.render([roadway, buildingmap, idm_scenes[t]])
# end;
# body!(win, man)

##############################################################################
"""
Train MDN
"""

feat, y = preprocess_data(1, scenes);

params = MDNParams(batch_size=2, lr=1e-3);

net = construct_mdn(params);

train_nnet!(feat, y, net..., params);

##############################################################################
"""
Cross Entropy Method
"""

function POMDPStressTesting.cem_losses(d, sample; mdp::ASTMDP, initstate::ASTState)
    sim = mdp.sim
    
    s = initstate
    R = 0 # accumulated reward
    
    BlackBox.initialize!(sim)
    AST.go_to_state(mdp, s)

    # Dummy values for debugging, not used in logprob
    tmp_norm = Normal(0.0, 5.0)
    

    sample_length = length(last(first(sample))) # get length of sample vector ("second" element in pair using "first" key)

    # Compute reward
    for i in 1:sample_length
        env_sample = GrayBox.EnvironmentSample()
        for k in keys(sample)
            value = sample[k][i]
            env_sample[k] = GrayBox.Sample(value, logpdf(tmp_norm, value))
        end
        a = ASTSampleAction(env_sample)
        (s, r) = @gen(:sp, :r)(mdp, s, a)
        R += r
        if BlackBox.isterminal(mdp.sim)
            break
        end
    end

    return -R # negative (loss)
end

##############################################################################
"""
Setup Adaptive Stress Testing
"""

function setup_ast(seed=0)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = AutoRiskSim()

    # Modified logprob from Surrogate probability model
    function logprob(sample, state)
        scenes = [state]
        feat, y = preprocess_data(1, scenes);
        logprobs = evaluate_logprob(feat, y, net..., params)
        return logprobs[1]
        # logpdf(sample)
    end

    sim._logpdf = (sample, state) -> logprob(sample, state)

    # AST MDP formulation object
    mdp::ASTMDP = ASTMDP{ASTSampleAction}(sim)
    mdp.params.debug = true # record metrics
    mdp.params.top_k = 10   # record top k best trajectories
    mdp.params.seed = seed  # set RNG seed for determinism
    mdp.params.collect_data = true # collect supervised dataset (𝐱=disturbances, y=isevent)

    USE_CEM = false

    if USE_CEM
        # Hyperparameters for CEM as the solver
        solver = POMDPStressTesting.CEMSolver(n_iterations=100,
                        #    num_samples=500,
                        #    elite_thresh=3000.,
                        #    min_elite_samples=20,
                        #    max_elite_samples=200,
                        episode_length=sim.params.endtime)
    else
        # Hyperparameters for MCTS-PW as the solver
        solver = MCTSPWSolver(n_iterations=1000,        # number of algorithm iterations
                              exploration_constant=1.0, # UCT exploration
                              k_action=1.0,             # action widening
                              alpha_action=0.5,         # action widening
                              depth=sim.params.endtime) # tree depth
    end
    # Get online planner (no work done, yet)
    planner = solve(solver, mdp)

    return planner
end;

planner = setup_ast(10);

action_trace = search!(planner)

##############################################################################
"""
Evaluate Plan (Metrics)
"""

# episodic_figures(planner.mdp, gui=true); POMDPStressTesting.gcf();
# distribution_figures(planner.mdp, gui=true); POMDPStressTesting.gcf();

playback_trace = playback(planner, action_trace, BlackBox.distance, return_trace=true);

failure_rate = print_metrics(planner).failure_rate

begin
    # TODO: get this index from the `trace` itself
    # findmax(planner.mdp.metrics.reward[planner.mdp.metrics.event])
    # findmax(ast_mdp.metrics.reward[ast_mdp.metrics.event])

    failure_likelihood = NaN
    if any(planner.mdp.metrics.event)
        # Failures were found.
        failure_likelihood =
            round(exp(maximum(planner.mdp.metrics.logprob[planner.mdp.metrics.event])), digits=8)
    end

    # Markdown.parse(string("\$\$p = ", failure_likelihood, "\$\$"))
    print(string("p = ", failure_likelihood, "\n"))
end

##############################################################################
"""
Evaluate Plan (Interactive)
"""

failure_action_set = filter(d->d[2], planner.mdp.dataset)

# [end-1] to remove closure rate from 𝐱 data
displayed_action_trace = convert(Vector{ASTAction}, failure_action_set[1][1][1:end-1]) # TODO: could be empty.

playback_trace = playback(planner, displayed_action_trace, sim->sim.state, return_trace=true)

win = Blink.Window()

man = @manipulate for t=slider(1:length(playback_trace), value=1., label="t")
    AutomotiveVisualization.render([planner.mdp.sim.problem.roadway, buildingmap, playback_trace[min(t, length(playback_trace))]])
end;

body!(win, man)


include("failure_analysis.jl")
plot_failure_distribution(planner.mdp.dataset)

include("../../RiskSimulator.jl/src/RiskMetrics.jl")
metrics = RiskMetrics(cost_data(planner.mdp.dataset), 0.2)
include("../../RiskSimulator.jl/src/plotting.jl")
risk_plot(metrics; mean_y=0.3, var_y=0.22, cvar_y=0.15, α_y=0.2)
