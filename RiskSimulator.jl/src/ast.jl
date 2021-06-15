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
    init_noise_1 = Noise(pos=(0,0), vel=0)
    init_noise_2 = Noise(pos=(0,0), vel=0)

    # Driving scenario
    scenario::Scenario = scenario_t_head_on_turn(init_noise_1=init_noise_1, init_noise_2=init_noise_2)

    # Roadway from scenario
    # roadway::Roadway = multi_lane_roadway() # Default roadway
    roadway::Roadway = scenario.roadway # Default roadway

    # System under test, ego vehicle
    # sut = BlinkerVehicleAgent(get_urban_vehicle_1(id=1, s=5.0, v=15.0, noise=init_noise_1, roadway=roadway),
    # sut = BlinkerVehicleAgent(t_left_to_right(id=1, noise=init_noise_1, roadway=roadway),
    # UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=true, ignore_idm=!params.ignore_sensors));
    sut = scenario.sut

    # Noisy adversary, vehicle
    # adversary = BlinkerVehicleAgent(get_urban_vehicle_2(id=2, s=25.0, v=0.0, noise=init_noise_2, roadway=roadway),
    # UrbanIDM(idm=IntelligentDriverModel(v_des=0.0), noisy_observations=false));
    # adversary = BlinkerVehicleAgent(t_right_to_turn(id=2, noise=init_noise_2, roadway=roadway),
    # UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=false));
    adversary = scenario.adversary

    # Adversarial Markov decision process
    problem::MDP = AdversarialDrivingMDP(sut, [adversary], roadway, 0.1)
    state::Scene = rand(initialstate(problem))
    prev_distance::Real = -1e10 # Used when agent goes out of frame

    # Noise distributions and disturbances (consistent with output variables in _logpdf)
    xposition_noise_veh::Distribution = Normal(0, 3) # Gaussian noise (notice larger σ)
    yposition_noise_veh::Distribution = Normal(0, 3) # Gaussian noise
    velocity_noise_veh::Distribution = Normal(0, 1e-4) # Gaussian noise

    xposition_noise_sut::Distribution = Normal(0, 3) # Gaussian noise (notice larger σ)
    yposition_noise_sut::Distribution = Normal(0, 3) # Gaussian noise
    velocity_noise_sut::Distribution = Normal(0, 1e-4) # Gaussian noise

    disturbances = scenario.disturbances # Initial 0-noise disturbance

    _logpdf::Function = (sample, state) -> 0 # Function for evaluating logpdf
end

"""
Change the SUT for the ego vehicle.
    e.g.,

    - StaticLaneFollowingDriver(0.0), # zero acceleration
    - IntelligentDriverModel(v_des=12.0), # IDM with 12 m/s speed
    - PrincetonDriver(v_des=10.0) # Princeton driver model with selected speed
    - TODO: Robert Dyro's behavior agent.
"""
function AutoRiskSim(system, scenario)
    sim = AutoRiskSim(scenario=scenario)
    sim.sut.model.idm = system
    return sim
end


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
    sim.disturbances[2] = typeof(sim.disturbances[2])(noise=noise_veh) # could be BlinkerVehicleControl or PedestrianControl

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
    sim.problem = AdversarialDrivingMDP(sim.sut, [sim.adversary], sim.roadway, 0.1)
    sim.state = rand(initialstate(sim.problem))
    sim.disturbances = Disturbance[BlinkerVehicleControl(), typeof(sim.disturbances[2])()] # noise-less
    sim.prev_distance = -1e10
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


"""
Find the maximum standard deviation of the GrayBox.environment to be used in the DRL `output_factor`.
"""
function maximum_std(sim::AutoRiskSim)
    env = GrayBox.environment(sim)
    return maximum(std(env[k]) for k in keys(env))
end


##############################################################################
function state2vec(scene::Scene)
    statevec = Float64[]
    for entity in scene.entities
        push!(statevec, entity.state.veh_state.posG...)
        push!(statevec, entity.state.veh_state.v)
    end
    return statevec
end

global STATE_PROXY = :none
function GrayBox.state(sim::AutoRiskSim)
    global STATE_PROXY

    if STATE_PROXY == :distance
        return [BlackBox.distance(sim)]
    elseif STATE_PROXY == :rate
        return [BlackBox.rate(sim.prev_distance, sim)]
    elseif STATE_PROXY == :actual || STATE_PROXY == :true
        return state2vec(sim.state)
    elseif STATE_PROXY == :none
        return nothing
    end
end


"""
Setup Adaptive Stress Testing
"""
function setup_ast(;
        sut=IntelligentDriverModel(v_des=12.0),
        scenario=scenario_hw_stopping(),
        seed=0,
        state_proxy=:none,
        nnobs=true,
        which_solver=:mcts, # :mcts, :cem, :ppo, :random
        noise_adjustment=nothing,
        use_potential_based_shaping=true,
        rollout=AST.rollout)

    global STATE_PROXY

    # Create gray-box simulation object
    sim::GrayBox.Simulation = AutoRiskSim(sut, scenario)

    # Adjust noise before training observation model
    if !isnothing(noise_adjustment)
        noise_adjustment(sim)
    end

    # Train observation model
    if nnobs
        @info "Training observation model."
        net, net_params = training_phase(sim; seed=seed)
    end

    # Modified logprob from Surrogate probability model
    function logprob(sample, state)
        if nnobs
            scenes = [state]
            feat, y = preprocess_data(1, scenes);
            logprobs = evaluate_logprob(feat, y, net..., net_params)
            # TODO: record the logprob(sample) for apples-to-apples comparison.
            return logprobs[1]
        else
            return logpdf(sample)
        end
    end

    sim._logpdf = (sample, state) -> logprob(sample, state)

    # AST MDP formulation object
    mdp::ASTMDP = ASTMDP{ASTSampleAction}(sim)
    mdp.params.debug = true # record metrics
    mdp.params.top_k = 10   # record top k best trajectories
    mdp.params.seed = seed  # set RNG seed for determinism
    mdp.params.collect_data = true # collect supervised dataset (𝐱=disturbances, y=isevent)
    mdp.params.use_potential_based_shaping = use_potential_based_shaping

    STATE_PROXY = state_proxy # set the state-proxy value (needs to be global for method definition)

    @info which_solver

    if which_solver == :cem
        # Hyperparameters for CEM as the solver
        solver = POMDPStressTesting.CEMSolver(n_iterations=100,
                        #    num_samples=500,
                        #    elite_thresh=3000.,
                        #    min_elite_samples=20,
                        #    max_elite_samples=200,
                        episode_length=sim.params.endtime)
    elseif which_solver == :mcts
        # Hyperparameters for MCTS-PW as the solver
        solver = MCTSPWSolver(n_iterations=1000,        # number of algorithm iterations
                              exploration_constant=1.0, # UCT exploration
                              k_action=1.0,             # action widening
                              alpha_action=0.5,         # action widening
                              estimate_value=rollout,   # rollout policy
                              depth=sim.params.endtime) # tree depth
    elseif which_solver == :ppo
        solver = PPOSolver(num_episodes=1000, # Doubled.
                           η=1e-1,
                           episode_length=sim.params.endtime,
                           output_factor=maximum_std(sim),
                           show_progress=false)
    elseif which_solver == :random
        solver = RandomSearchSolver(n_iterations=1000, episode_length=sim.params.endtime)
    else
        error("Solver does not exist ($which_solver)")
    end
    # Get online planner (no work done, yet)
    planner = solve(solver, mdp)

    return planner
end
