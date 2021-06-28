using CrossEntropyMethod
using Distributions, Random
using Statistics
using Parameters
using POMDPStressTesting

##############################################################################
"""
Create Simulation
"""

@with_kw struct CarlaParams
    endtime::Int64 = 60 # Simulate end time
    episodic_rewards::Bool = false # decision making process with epsidic rewards
    give_intermediate_reward::Bool = false # give log-probability as reward during intermediate gen calls (used only if `episodic_rewards`)
    reward_bonus::Float64 = episodic_rewards ? 100 : 0 # reward received when event is found, multiplicative when using `episodic_rewards`
    use_potential_based_shaping::Bool = true # apply potential-based reward shaping to speed up learning
    pass_seed_action::Bool = false # pass the selected RNG seed to the GrayBox.transition! and BlackBox.evaluate! functions
    discount::Float64 = 1.0 # discount factor (generally 1.0 to not discount later samples in the trajectory)
end;

@with_kw mutable struct CarlaSim <: GrayBox.Simulation
    t::Real = 0 # Current time
    dt::Real = 1 # Time interval
    params::CarlaParams = CarlaParams() # Parameters
    
    # Adversarial Markov decision process
    reset_fn = (x)->0
    step_fn = (x)->0
    state = 0
    running = true
    collision = false
    distance::Real = Inf
    prev_distance::Real = -Inf # Used when agent goes out of frame

    # Noise distributions and disturbances (consistent with output variables in _logpdf)
    xposition_noise_veh::Distribution = Normal(0, 5) # Gaussian noise (notice larger σ)
    yposition_noise_veh::Distribution = Normal(0, 5) # Gaussian noise
    velocity_noise_veh::Distribution = Normal(0, 1e-4) # Gaussian noise

    xposition_noise_sut::Distribution = Normal(0, 5) # Gaussian noise (notice larger σ)
    yposition_noise_sut::Distribution = Normal(0, 5) # Gaussian noise
    velocity_noise_sut::Distribution = Normal(0, 1e-4) # Gaussian noise
    
    disturbances::Vector{Float64} = Float64[0.0, 0.0] # Initial 0-noise disturbance

    _logpdf::Function = (sample, state) -> logpdf(sample) # Function for evaluating logpdf  
end;

##############################################################################
"""
Graybox functions
"""

function GrayBox.environment(sim::CarlaSim) 
   return GrayBox.Environment(
                            :xpos_veh => sim.xposition_noise_veh,
                            :ypos_veh => sim.yposition_noise_veh,
                        )
end;

function GrayBox.transition!(sim::CarlaSim, sample::GrayBox.EnvironmentSample)
    sim.t += sim.dt # Keep track of time
    
    sim.disturbances[1] = sample[:xpos_veh].value
    sim.disturbances[2] = sample[:ypos_veh].value

    # step agents: given MDP, current state, and current action (i.e. disturbances)
    (_running, _collision, sim.distance) = sim.step_fn(sim.disturbances)
    sim.running = _running > 0
    sim.collision = _collision > 0

    # return log-likelihood of actions, summation handled by `logpdf()`
    return sim._logpdf(sample, sim.state)::Real
end

##############################################################################
"""
Blackbox functions
"""

function BlackBox.initialize!(sim::CarlaSim)
    sim.t = 0
    sim.running = true
    sim.state = 0
    sim.reset_fn()
    sim.disturbances = [0, 0] # noise-less
    sim.prev_distance = -Inf
    sim.distance = Inf
    sim.collision = false
end

function BlackBox.distance(sim::CarlaSim)
    return sim.distance
end

function BlackBox.isevent(sim::CarlaSim)
    return sim.collision
end

function BlackBox.isterminal(sim::CarlaSim)
    return BlackBox.isevent(sim) ||
           !sim.running
end

function BlackBox.evaluate!(sim::CarlaSim, sample::GrayBox.EnvironmentSample)
    logprob::Real = GrayBox.transition!(sim, sample) # Step simulation
    d::Real       = BlackBox.distance(sim)           # Calculate miss distance
    event::Bool   = BlackBox.isevent(sim)            # Check event indication
    return (logprob::Real, d::Real, event::Bool)
end

# Pulling from POMDPStressTesting since not exported
function rate(d_prev::Real, sim::CarlaSim)
    return d_prev - BlackBox.distance(sim)
end

# Modified function without MDP
function reward(sim::CarlaSim, logprob::Real, isevent::Bool, isterminal::Bool, miss_distance::Real, rate::Real)
    if sim.params.episodic_rewards
        r = 0
        if isterminal
            r += logprob - miss_distance
            if isevent
                r *= sim.params.reward_bonus # R_E (multiplicative)
            end
        else
            intermediate_reward = mdp.params.give_intermediate_reward ? logprob : 0
            r += intermediate_reward
        end
    else # Standard AST reward function
        # always apply logprob to capture all likelihoods
        r = logprob
        if isterminal && isevent
            r += sim.params.reward_bonus # R_E (additive)
        elseif isterminal && !isevent
            r += -miss_distance # Only add miss distance cost if is terminal and not an event.
        end
        if sim.params.use_potential_based_shaping
            r += rate # potential-based reward shaping
        end
    end

    return r
end

function POMDPStressTesting.cem_losses(d, sample; sim::CarlaSim)
    sample_length = length(last(first(sample)))
    env = GrayBox.environment(sim)
    R = 0 # accumulated reward
    BlackBox.initialize!(sim)

    for i in 1:sample_length
        env_sample = GrayBox.EnvironmentSample()
        for k in keys(sample)
            value = sample[k][i]
            env_sample[k] = GrayBox.Sample(value, logpdf(env[k], value))
        end
        (logprob, miss_distance, _isevent) = BlackBox.evaluate!(sim, env_sample)
        _rate = rate(sim.prev_distance, sim)
        r = reward(sim, logprob, _isevent, BlackBox.isterminal(sim), miss_distance, _rate)
        R += r
        if BlackBox.isterminal(sim)
            break                
        end
    end

    return -R
end

function Statistics.mean(d::Dict{Symbol, Vector{Sampleable}})
    Dict(k => Statistics.mean.(d[k]) for k in keys(d))
end

function run_CEM(step_fn, reset_fn)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = CarlaSim(step_fn=step_fn, reset_fn=reset_fn)
    env::GrayBox.Environment = GrayBox.environment(sim)
    
    # Importance sampling distributions, fill one per time step.
    is_dist_0 = convert(Dict{Symbol, Vector{Sampleable}}, env, sim.params.endtime)

    # Run cross-entropy method using importance sampling
    loss = (d, sample)->POMDPStressTesting.cem_losses(d, sample; sim)
    is_dist_opt = cross_entropy_method(loss,
                                       is_dist_0;
                                       max_iter=3,
                                       N=10,
                                    #    min_elite_samples=planner.solver.min_elite_samples,
                                    #    max_elite_samples=planner.solver.max_elite_samples,
                                    #    elite_thresh=planner.solver.elite_thresh,
                                    #    weight_fn=planner.solver.weight_fn,
                                    #    add_entropy=planner.solver.add_entropy,
                                       verbose=false,
                                       show_progress=true,
                                       batched=false)
    
    failure_path = Statistics.mean(is_dist_opt)
    disturbances = zeros(2, sim.params.endtime)
    disturbances[1, :] = failure_path[:xpos_veh]
    disturbances[2, :] = failure_path[:ypos_veh]
    return disturbances
end