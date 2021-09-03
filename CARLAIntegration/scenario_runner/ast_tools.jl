using Base: Bool
using CrossEntropyMethod
using Distributions, Random
using Statistics
using Parameters
using POMDPStressTesting
# using FFMPEG, Plots
using FileIO
using Dates
# using RiskSimulator
##############################################################################
"""
Create Simulation
"""

@with_kw struct CarlaParams
    endtime::Int64 = 200 # Simulate end time
    episodic_rewards::Bool = false # decision making process with epsidic rewards
    give_intermediate_reward::Bool = false # give log-probability as reward during intermediate gen calls (used only if `episodic_rewards`)
    reward_bonus::Float64 = episodic_rewards ? 100 : 200 # reward received when event is found, multiplicative when using `episodic_rewards`
    use_potential_based_shaping::Bool = false # apply potential-based reward shaping to speed up learning
    pass_seed_action::Bool = false # pass the selected RNG seed to the GrayBox.transition! and BlackBox.evaluate! functions
    discount::Float64 = 1.0 # discount factor (generally 1.0 to not discount later samples in the trajectory)
    debug::Bool = true
    collect_data::Bool = true # collect supervised data {(𝐱=rates, y=isevent), ...} 
    resume::Bool = false
    resume_path = raw"variables\record_2021_07_11_170347.jld2"
end;

@with_kw mutable struct CarlaSim <: GrayBox.Simulation
    t::Real = 0 # Current time
    dt::Real = 1 # Time interval
    params::CarlaParams = CarlaParams() # Parameters
    metrics::AST.ASTMetrics = AST.ASTMetrics() # Metrics to record

    dataset::Vector = [] # (𝐱=disturbances, y=isevent) supervised dataset
    
    # Adversarial Markov decision process
    reset_fn = (x)->0
    step_fn = (x)->0
    state = 0
    running = true
    collision = false
    distance::Real = -1e10
    prev_distance::Real = -1e10 # Used when agent goes out of frame

    # Noise distributions and disturbances (consistent with output variables in _logpdf)
    xposition_noise_veh::Distribution = Normal(0, 4) # Gaussian noise (notice larger σ)
    yposition_noise_veh::Distribution = Normal(0, 4) # Gaussian noise
    velocity_noise_veh::Distribution = Normal(0, 1e-4) # Gaussian noise

    xposition_noise_sut::Distribution = Normal(0, 4) # Gaussian noise (notice larger σ)
    yposition_noise_sut::Distribution = Normal(0, 4) # Gaussian noise
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
    sim.prev_distance = sim.distance
    (_running, _collision, sim.distance) = sim.step_fn(sim.disturbances)
    _rate = rate(sim.prev_distance, sim);
    @show _running, _collision, sim.distance, _rate
    sim.running = _running > 0
    sim.collision = _collision == true

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
    sim.prev_distance = -1e10
    sim.distance = -1e10
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

# Pulling from POMDPStressTesting since not exported and modifying with CarlaSim
function rate(d_prev::Real, sim::CarlaSim)
    return d_prev - BlackBox.distance(sim)
end

###################################################################
"""
Record tools (modified from POMDPStressTesting.jl)
"""
function record(sim::CarlaSim, sym::Symbol, val)
    if sim.params.debug
        push!(getproperty(sim.metrics, sym), val)
    end
end

function record(sim::CarlaSim; logprob=1, prob=exp(logprob), miss_distance=Inf, reward=-Inf, event=false, terminal=false, rate=-Inf)
    record(sim, :prob, prob)
    record(sim, :logprob, logprob)
    record(sim, :miss_distance, miss_distance)
    record(sim, :reward, reward)
    record(sim, :intermediate_reward, reward)
    record(sim, :rate, rate)
    record(sim, :event, event)
    record(sim, :terminal, terminal)
end

function record_returns(sim::CarlaSim)
    # compute returns up to now.
    rewards = sim.metrics.intermediate_reward
    G = returns(rewards, γ=sim.params.discount)
    record(sim, :returns, G)
    sim.metrics.intermediate_reward = [] # reset
end


function returns(R; γ=1)
    T = length(R)
    G = zeros(T)
    for t in reverse(1:T)
        G[t] = t==T ? R[t] : G[t] = R[t] + γ*G[t+1]
    end
    return G
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
    # Record metrics
    record(sim, prob=exp(logprob), logprob=logprob, miss_distance=miss_distance, reward=r, event=isevent, terminal=isterminal, rate=rate)
    if isterminal
        # end of episode
        record_returns(sim)
    end
    return r
end

TMP_DATA_RATE = []
TMP_DATA_DISTANCE = []
function POMDPStressTesting.cem_losses(d, sample; sim::CarlaSim)
    sample_length = length(last(first(sample)))
    env = GrayBox.environment(sim)
    R = 0 # accumulated reward
    BlackBox.initialize!(sim)

    max_rate = 0
    for i in 1:sample_length
        env_sample = GrayBox.EnvironmentSample()
        for k in keys(sample)
            value = sample[k][i]
            env_sample[k] = GrayBox.Sample(value, logpdf(env[k], value))
        end
        (logprob, miss_distance, _isevent) = BlackBox.evaluate!(sim, env_sample)
        _rate = rate(sim.prev_distance, sim)
        max_rate = (max_rate > _rate) || (i<sample_length/1.5) ? max_rate : _rate
        r = reward(sim, logprob, _isevent, BlackBox.isterminal(sim), miss_distance, _rate)
        R += r
        
        # # Data collection of {(𝐱=rates, y=isevent), ...}
        # if sim.params.collect_data
        #     global TMP_DATA_RATE, TMP_DATA_DISTANCE
        #     closure_rate = _rate
        #     distance = BlackBox.distance(sim)
        #     push!(TMP_DATA_RATE, closure_rate)
        #     push!(TMP_DATA_DISTANCE, distance)
        # end
        
        if BlackBox.isterminal(sim)
            # Data collection of {(𝐱=disturbances, y=isevent), ...}
            if sim.params.collect_data
                closure_rate = max_rate
                distance = BlackBox.distance(sim)
                𝐱 = vcat(sample, distance, closure_rate)
                y = BlackBox.isevent(sim)
                push!(sim.dataset, (𝐱, y))
            end
            break                
        end
    end
    # if sim.params.collect_data
    #     global TMP_DATA_RATE, TMP_DATA_DISTANCE
    #     𝐱 = [sample, TMP_DATA_DISTANCE, TMP_DATA_RATE]
    #     y = BlackBox.isevent(sim)
    #     push!(sim.dataset, (𝐱, y))
    #     TMP_DATA_DISTANCE = []
    #     TMP_DATA_RATE = []
    # end

    @show "Reward: ", -R
    return -R
end

function Statistics.mean(d::Dict{Symbol, Vector{Sampleable}})
    Dict(k => Statistics.mean.(d[k]) for k in keys(d))
end

function run_CEM(step_fn, reset_fn, record_step_fn)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = CarlaSim(step_fn=step_fn, reset_fn=reset_fn)
    env::GrayBox.Environment = GrayBox.environment(sim)

    if sim.params.resume
        tmp = load(sim.params.resume_path)
        sim.metrics = tmp["metrics"]
        sim.dataset = tmp["dataset"]
        is_dist_0 = tmp["is_dist_opt"]
    else
        # Importance sampling distributions, fill one per time step.
        is_dist_0 = convert(Dict{Symbol, Vector{Sampleable}}, env, sim.params.endtime)
    end
    
    # Run cross-entropy method using importance sampling
    loss = (d, sample)->POMDPStressTesting.cem_losses(d, sample; sim)
    for i_cem=1:20
        is_dist_opt = cross_entropy_method(loss,
                                        is_dist_0;
                                        max_iter=1,
                                        N=50,
                                        #    min_elite_samples=planner.solver.min_elite_samples,
                                        #    max_elite_samples=planner.solver.max_elite_samples,
                                        elite_thresh=800,
                                        #    weight_fn=planner.solver.weight_fn,
                                        #    add_entropy=planner.solver.add_entropy,
                                        verbose=false,
                                        show_progress=true,
                                        batched=false
                                        )
        # Risk Assessment.
        # ————————————————————————————————————————————————
        fail_metrics = POMDPStressTesting.print_metrics(sim.metrics) 
        @show fail_metrics
        
        timestamp = Dates.format(now(), "_yyyy_mm_dd_HHMMSS")
        save(raw"variables\record"*timestamp*".jld2", Dict("metrics" => sim.metrics, "dataset" => sim.dataset, "is_dist_opt" => is_dist_opt))
        is_dist_0 = is_dist_opt
    end
    # p_closure = plot_closure_rate_distribution(𝒟; reuse=false)
    # savefig(p_closure,raw"plots\closure_rate_distribution"*timestamp*".png")

    # # Plot cost distribution.
    # metrics = risk_assessment(𝒟, α)
    # @show metrics
    # p_risk = plot_risk(metrics; mean_y=0.33, var_y=0.25, cvar_y=0.1, α_y=0.2)
    # savefig(p_risk,raw"plots\risk"*timestamp*".png")
    
    # # Polar plot of risk and failure metrics
    # 𝐰 = ones(7)
    # p_metrics = plot_polar_risk([𝒟], [sim.metrics], ["Carla BaseAgent"]; weights=𝐰, α=α)
    # savefig(p_metrics,raw"plots\polar_risk"*timestamp*".png")
    
    failure_path = Statistics.mean(is_dist_0)
    disturbances = zeros(2, sim.params.endtime)
    disturbances[1, :] = failure_path[:xpos_veh]
    disturbances[2, :] = failure_path[:ypos_veh]
    return disturbances
end