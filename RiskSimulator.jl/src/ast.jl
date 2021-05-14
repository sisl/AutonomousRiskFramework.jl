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
    UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=true, ignore_idm=!params.ignore_sensors));

    # Noisy adversary, vehicle
    adversary = BlinkerVehicleAgent(get_urban_vehicle_2(id=2, s=25.0, v=0.0, noise=init_noise_2),
    UrbanIDM(idm=IntelligentDriverModel(v_des=0.0), noisy_observations=false));

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
end

"""
Change the SUT for the ego vehicle.
    e.g.,

    - StaticLaneFollowingDriver(0.0), # zero acceleration
    - IntelligentDriverModel(v_des=12.0), # IDM with 12 m/s speed
    - PrincetonDriver(v_des=10.0) # Princeton driver model with selected speed
    - TODO: Robert Dyro's behavior agent.
"""
function AutoRiskSim(system)
    sim = AutoRiskSim()
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

function setup_ast(net, net_params; sut=IntelligentDriverModel(v_des=12.0), seed=0, nnobs=true)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = AutoRiskSim(sut)

    # Modified logprob from Surrogate probability model
    function logprob(sample, state)
        if nnobs
            scenes = [state]
            feat, y = preprocess_data(1, scenes);
            logprobs = evaluate_logprob(feat, y, net..., net_params)
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