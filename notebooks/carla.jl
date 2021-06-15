### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 92ce9460-f62b-11ea-1a8c-179776b5a0b4
using Revise, Distributions, Parameters, POMDPStressTesting, Latexify, PlutoUI

# ╔═╡ 9946ce80-72df-11eb-2bb8-ab675371090c
using PyPlot, Seaborn # required for episodic_figures

# ╔═╡ 2f501370-7de2-11eb-024b-07af757fc74a
using RollingFunctions

# ╔═╡ 3cd27530-7e49-11eb-23b0-313cf51d1c7f
if false
    using POMDPs, Crux, Flux, POMDPGym

    ## Cartpole - V0
    mdp = GymPOMDP(:CartPole, version = :v0)
    as = actions(mdp)
    S = state_space(mdp)

    Q() = DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)
    V() = ContinuousNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1)))
    A() = DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as)), softmax), as)

    # Solve with REINFORCE
    𝒮_reinforce = PGSolver(π = A(), S = S, N=10000, ΔN = 500, loss = reinforce())
    π_reinforce = solve(𝒮_reinforce, mdp)

    # Solve with A2C
    𝒮_a2c = PGSolver(π = ActorCritic(A(), V()), S = S, N=10000, ΔN = 500, loss = a2c())
    π_a2c = solve(𝒮_a2c, mdp)

    # Solve with PPO
    𝒮_ppo = PGSolver(π = ActorCritic(A(), V()), S = S, N=10000, ΔN = 500, loss = ppo())
    π_ppo = solve(𝒮_ppo, mdp)

    # Solve with DQN
    𝒮_dqn = DQNSolver(π = Q(), S = S, N=10000)
    π_dqn = solve(𝒮_dqn, mdp)

    # Plot the learning curve
    p = plot_learning([𝒮_reinforce, 𝒮_a2c, 𝒮_ppo, 𝒮_dqn], title = "CartPole-V0 Training Curves", labels = ["REINFORCE", "A2C", "PPO", "DQN"])

    # Produce a gif with the final policy
    gif(mdp, π_ppo, "cartpole_policy.gif")

    # Return plot
    p
end

# ╔═╡ 2978b840-f62d-11ea-2ea0-19d7857208b1
md"""
# Black-Box Stress Testing
"""

# ╔═╡ 1e00230e-f630-11ea-3e40-bf8c852f78b8
# begin
#   using Pkg
#   pkg"registry add https://github.com/JuliaPOMDP/Registry"
#   pkg"add https://github.com/sisl/RLInterface.jl"
#   pkg"add https://github.com/sisl/POMDPStressTesting.jl"
# end
md"*Unhide for installation (waiting on Julia registry).*"

# ╔═╡ 40d3b1e0-f630-11ea-2160-01338d9f2209
md"""
To find failures in a black-box autonomous system, we can use the `POMDPStressTesting` package which is part of the POMDPs.jl ecosystem.

Various solvers—which adhere to the POMDPs.jl interface—can be used:
- `MCTSPWSolver` (MCTS with action progressive widening)
- `TRPOSolver` and `PPOSolver` (deep reinforcement learning policy optimization)
- `CEMSolver` (cross-entropy method)
- `RandomSearchSolver`
"""

# ╔═╡ 86f13f60-f62d-11ea-3241-f3f1ffe37d7a
md"""
## Simple Problem: One-Dimensional Walk
We define a simple problem for adaptive stress testing (AST) to find failures. This problem, called Walk1D, samples random walking distances from a standard normal distribution $\mathcal{N}(0,1)$ and defines failures as walking past a certain threshold (which is set to $\pm 10$ in this example). AST will either select the seed which deterministically controls the sampled value from the distribution (i.e. from the transition model) or will directly sample the provided environmental distributions. These action modes are determined by the seed-action or sample-action options. AST will guide the simulation to failure events using a notion of distance to failure, while simultaneously trying to find the set of actions that maximizes the log-likelihood of the samples.
"""

# ╔═╡ d3411dd0-f62e-11ea-27d7-1b2ed8edc415
md"""
## Gray-Box Simulator and Environment
The simulator and environment are treated as gray-box because we need access to the state-transition distributions and their associated likelihoods.
"""

# ╔═╡ e37d7542-f62e-11ea-0b61-513a4b44fc3c
md"""
##### Parameters
First, we define the parameters of our simulation.
"""

# ╔═╡ fd7fc880-f62e-11ea-15ac-f5407aeff2a6
@with_kw mutable struct Walk1DParams
    startx::Float64 = 0   # Starting x-position
    threshx::Float64 = 10 # ± boundary threshold
    endtime::Int64 = 30   # Simulate end time
end;

# ╔═╡ 012c2eb0-f62f-11ea-1637-c113ad01b144
md"""
##### Simulation
Next, we define a `GrayBox.Simulation` structure.
"""

# ╔═╡ 0d7049de-f62f-11ea-3552-214fc4e7ec98
@with_kw mutable struct Walk1DSim <: GrayBox.Simulation
    params::Walk1DParams = Walk1DParams() # Parameters
    x::Float64 = 0 # Current x-position
    t::Int64 = 0 # Current time ±
    rate::Real = 0 # Current rate (TODO: used for GrayBox.state)
    distribution::Distribution = Normal(0, 1) # Transition distribution
end;

# ╔═╡ 11e445d0-f62f-11ea-305c-495272981112
md"""
### GrayBox.environment
Then, we define our `GrayBox.Environment` distributions. When using the `ASTSampleAction`, as opposed to `ASTSeedAction`, we need to provide access to the sampleable environment.
"""

# ╔═╡ 43c8cb70-f62f-11ea-1b0d-bb04a4176730
GrayBox.environment(sim::Walk1DSim) = GrayBox.Environment(:x => sim.distribution)

# ╔═╡ 48a5e970-f62f-11ea-111d-35694f3994b4
md"""
### GrayBox.transition!
We override the `transition!` function from the `GrayBox` interface, which takes an environment sample as input. We apply the sample in our simulator, and return the log-likelihood.
"""

# ╔═╡ 5d0313c0-f62f-11ea-3d33-9ded1fb804e7
function GrayBox.transition!(sim::Walk1DSim, sample::GrayBox.EnvironmentSample)
    sim.t += 1 # Keep track of time
    sim.x += sample[:x].value # Move agent using sampled value from input
    return logpdf(sample)::Real # Summation handled by `logpdf()`
end

# ╔═╡ 6e111310-f62f-11ea-33cf-b5e943b2f088
md"""
## Black-Box System
The system under test, in this case a simple single-dimensional moving agent, is always treated as black-box. The following interface functions are overridden to minimally interact with the system, and use outputs from the system to determine failure event indications and distance metrics.
"""

# ╔═╡ 7c84df7e-f62f-11ea-3b5f-8b090654df19
md"""
### BlackBox.initialize!
Now we override the `BlackBox` interface, starting with the function that initializes the simulation object. Interface functions ending in `!` may modify the `sim` object in place.
"""

# ╔═╡ 9b736bf2-f62f-11ea-0330-69ffafe9f200
function BlackBox.initialize!(sim::Walk1DSim)
    sim.t = 0
    sim.x = sim.params.startx
end

# ╔═╡ a380e250-f62f-11ea-363d-2bf2b59d5eed
md"""
### BlackBox.distance
We define how close we are to a failure event using a non-negative distance metric.
"""

# ╔═╡ be39db60-f62f-11ea-3a5c-bd57114455ff
# BlackBox.distance(sim::Walk1DSim) = sim.params.threshx - abs(sim.x)# max(sim.params.threshx - abs(sim.x), 0)
BlackBox.distance(sim::Walk1DSim) = max(sim.params.threshx - abs(sim.x), 0)

# ╔═╡ c48da3f0-72eb-11eb-3251-01c0c6a588d3
md"""
### BlackBox.rate
Calculate the rate-to-failure.
"""

# ╔═╡ d1846d50-72eb-11eb-26a3-af886fda1ab1
# BlackBox.rate(d_prev::Real, sim::Walk1DSim) = d_prev - BlackBox.distance(sim)

# ╔═╡ bf8917b0-f62f-11ea-0e77-b58065b0da3e
md"""
### BlackBox.isevent
We define an indication that a failure event occurred.
"""

# ╔═╡ c5f03110-f62f-11ea-1119-81f5c9ec9283
BlackBox.isevent(sim::Walk1DSim) = abs(sim.x) ≥ sim.params.threshx

# ╔═╡ c378ef80-f62f-11ea-176d-e96e1be7736e
md"""
### BlackBox.isterminal
Similarly, we define an indication that the simulation is in a terminal state.
"""

# ╔═╡ cb5f7cf0-f62f-11ea-34ca-5f0656eddcd4
function BlackBox.isterminal(sim::Walk1DSim)
    return BlackBox.isevent(sim) || sim.t ≥ sim.params.endtime
end

# ╔═╡ e2f34130-f62f-11ea-220b-c7fc7de2c7e7
md"""
### BlackBox.evaluate!
Lastly, we use our defined interface to evaluate the system under test. Using the input sample, we return the log-likelihood, distance to an event, and event indication.
"""

# ╔═╡ f6213a50-f62f-11ea-07c7-2dcc383c8042
function BlackBox.evaluate!(sim::Walk1DSim, sample::GrayBox.EnvironmentSample)
    d_prev::Real  = BlackBox.distance(sim)           # Calculate previous distance
    logprob::Real = GrayBox.transition!(sim, sample) # Step simulation
    d::Real       = BlackBox.distance(sim)           # Calculate miss distance
    event::Bool   = BlackBox.isevent(sim)            # Check event indication
    # rate::Real    = BlackBox.rate(d_prev, sim)       # Calculate the rate-to-failure
    # sim.rate = rate # TODO: Set elsewhere?

    # TODO: return -rate (instead of d)
    
    # F = d + -rate # TODO: will be negated to be -d + rate
    # return (logprob::Real, F::Real, event::Bool)

    return (logprob::Real, d::Real, event::Bool)
    # return (logprob::Real, -rate::Real, event::Bool)
end

# ╔═╡ 1fca5f1e-72dc-11eb-2e35-5b926d1fd105
md"""
## Fully Observable Simulator State
In some problems, we can have a fully observable simulation state, or can use the distance metric as a state-proxy. In these cases, we define the `GrayBox.state` function to return the current state.
"""

# ╔═╡ 996a0260-72e9-11eb-1846-8f904861d15e
use_state_proxy = :rate # :distance, :rate, :actual

# ╔═╡ 4fe74420-72dc-11eb-063a-1fdd78a5b686
if use_state_proxy == :distance
    GrayBox.state(sim::Walk1DSim) = [BlackBox.distance(sim)]
elseif use_state_proxy == :rate
    GrayBox.state(sim::Walk1DSim) = [sim.rate]
elseif use_state_proxy == :actual
    GrayBox.state(sim::Walk1DSim) = [sim.x]
end

# ╔═╡ ac267500-72e9-11eb-0b4b-415b512b336b
# TODO: BlackBox.rate (automatically created if distance is available)

# ╔═╡ 01da7aa0-f630-11ea-1262-f50453455766
md"""
## AST Setup and Running
Setting up our simulation, we instantiate our simulation object and pass that to the Markov decision proccess (MDP) object of the adaptive stress testing formulation. We use Monte Carlo tree search (MCTS) with progressive widening on the action space as our solver. Hyperparameters are passed to `MCTSPWSolver`, which is a simple wrapper around the POMDPs.jl implementation of MCTS. Lastly, we solve the MDP to produce a planner. Note we are using the `ASTSampleAction`.
"""

# ╔═╡ fdf55130-f62f-11ea-33a4-a783b4d216dc
function setup_ast(seed=2)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = Walk1DSim()

    # AST MDP formulation object
    mdp::ASTMDP = ASTMDP{ASTSampleAction}(sim)
    mdp.params.debug = true # record metrics
    mdp.params.top_k = 100   # record top k best trajectories
    mdp.params.seed = seed  # set RNG seed for determinism
    mdp.params.reward_bonus = 50 # R_E

    # Hyperparameters for MCTS-PW as the solver
    solver = MCTSPWSolver(n_iterations=9000,        # number of algorithm iterations
                          exploration_constant=1.0, # UCT exploration
                          k_action=1.0,             # action widening
                          alpha_action=0.5,         # action widening
                          depth=sim.params.endtime) # tree depth

    # Get online planner (no work done, yet)
    planner = solve(solver, mdp)

    return planner
end;

# ╔═╡ 09c928f0-f631-11ea-3ef7-512a6bececcc
md"""
### Searching for Failures
After setup, we search for failures using the planner and output the best action trace.
"""

# ╔═╡ 17d0ed20-f631-11ea-2e28-3bb9ca9a445f
planner = setup_ast();

# ╔═╡ b9131470-7dda-11eb-232f-c1642f3a7864
mcts_mdp = planner.mdp;

# ╔═╡ 1c47f652-f631-11ea-15f6-1b9b59700f36
action_trace = search!(planner)

# ╔═╡ 21530220-f631-11ea-3994-319c862d51f9
md"""
### Playback
We can also playback specific trajectories and print intermediate $x$-values.
"""

# ╔═╡ 3b282ae0-f631-11ea-309d-639bf4411bb3
playback_trace = playback(planner, action_trace, sim->sim.x, return_trace=true)

# ╔═╡ 7473adb0-f631-11ea-1c87-0f76b18a9ab6
failure_rate = print_metrics(planner)

# ╔═╡ b6244db0-f63a-11ea-3b48-89d427664f5e
md"""
## Other Solvers: Cross-Entropy Method
We can easily take our `ASTMDP` object (`planner.mdp`) and re-solve the MDP using a different solver—in this case the `CEMSolver`.
"""

# ╔═╡ e15cc1b0-f63a-11ea-2401-5321d48118c3
cem_mdp = setup_ast().mdp; # new copy

# ╔═╡ f5a15af0-f63a-11ea-1dd7-593d7cb01ee4
cem_solver = CEMSolver(n_iterations=100, episode_length=cem_mdp.sim.params.endtime)

# ╔═╡ fb3fa610-f63a-11ea-2663-17224dc8aade
cem_planner = solve(cem_solver, cem_mdp);

# ╔═╡ 09c9e0b0-f63b-11ea-2d50-4154e3432fa0
cem_action_trace = search!(cem_planner);

# ╔═╡ 46b40e10-f63b-11ea-2375-1976bb637d51
md"Notice the failure rate is about 10x of `MCTSPWSolver`."

# ╔═╡ 32fd5cf2-f63b-11ea-263a-39f013ef6d68
cem_failure_rate = print_metrics(cem_planner)

# ╔═╡ 734b5320-72dc-11eb-3481-954d7d759154
md"""
## Other Solvers: PPO
Using a neural network that takes in the `GrayBox.state`, we can solve this problem using the PPO algorithm. **Note**: you must define a specific simulator state or this will essentially be random search.
"""

# ╔═╡ b4fa6620-72dd-11eb-121b-f55e92d3a613
ppo_mdp = setup_ast().mdp; # new copy

# ╔═╡ 9f2c3c20-72dc-11eb-31ce-8fde19792bd7
ppo_solver = PPOSolver(num_episodes=10_000) # 10_000

# ╔═╡ a4711ca0-72dc-11eb-2885-e98ad761f964
ppo_planner = solve(ppo_solver, ppo_mdp);

# ╔═╡ b4654780-72dc-11eb-340e-3f57e471c7aa
ppo_action_trace = search!(ppo_planner)

# ╔═╡ fbc67c9e-7e1e-11eb-002e-0148bc603efe
[AST.logpdf(action_trace); # MCTS
 AST.logpdf(cem_action_trace);
 AST.logpdf(ppo_action_trace)]

# ╔═╡ e192fd80-7ded-11eb-0c33-2dfff616be02
playback(ppo_planner, ppo_action_trace, sim->sim.x, return_trace=true)

# ╔═╡ 93e61380-72dd-11eb-094d-5d2f6acf967a
ppo_failure_rate = print_metrics(ppo_planner)

# ╔═╡ 66c23850-72df-11eb-08e2-27b3b5dd523c
distribution_figures(ppo_mdp, logprob=true, bw=[0.1, 0.001]); gcf()

# ╔═╡ 3dd97c7e-72e6-11eb-21bd-57989a9f53ad
exp.(ppo_mdp.metrics.logprob[findall(ppo_mdp.metrics.event)]) |> maximum

# ╔═╡ ed65d060-7dd9-11eb-084a-f924343f592f
episodic_figures(ppo_mdp); gcf()

# ╔═╡ 2f75bd80-7dda-11eb-049d-7f3392f0f2c1
# using RollingFunctions
function episodic_figures_multi(metrics::Vector, labels::Vector{String}, colors::Vector; gui::Bool=true, fillstd::Bool=false, learning_window=100, distance_window=5000, episodic_rewards=false)
    PyPlot.pygui(gui) # Plot with GUI window (if true)
    fig = figure(figsize=(7,9))
    handles = []

    for i in 1:length(metrics)
        miss_distances = metrics[i].miss_distance
        max_iters = length(miss_distances)

        # Font size changes
        plt.rc("axes", titlesize=15, labelsize=13)
        plt.rc("legend", fontsize=12)


        ## Plot 1: Learning curves (reward)
        ax = fig.add_subplot(4,1,1)
        G = mean.(metrics[i].returns)
        G = runmean(G, learning_window)
        title("Learning Curve")
        ax.plot(G, color=colors[i])
        xlabel("Episode")
        ylabel("Undiscounted Returns")
        # xscale("log")

        ## Plot 2: Running miss distance mean
        ax = fig.add_subplot(4,1,2)
        title("Running Miss Distance Mean")

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1.0)

        # rolling_mean = []
        # d_sum = 0
        # for i in 1:max_iters
        #     d_sum += miss_distances[i]
        #     push!(rolling_mean, d_sum/i)
        # end
        rolling_mean = runmean(mean.(metrics[i].miss_distance), distance_window)
        # [mean(miss_distances[1:i]) for i in 1:max_iters]
        ax.plot(rolling_mean, color=colors[i], zorder=2)
        if fillstd # TODO. More efficient approach.
            miss_std_below = [mean(miss_distances[1:i])-std(miss_distances[1:i]) for i in 1:max_iters]
            miss_std_above = [mean(miss_distances[1:i])+std(miss_distances[1:i]) for i in 1:max_iters]
            ax.fill_between(1:max_iters, miss_std_below, miss_std_above, color=colors[i], alpha=0.1)
        end

        ylabel("Miss Distance")
        ax.tick_params(labelbottom=false)
        # xscale("log")

        ## Plot 3: Minimum miss distance
        ax = fig.add_subplot(4,1,3)
        title("Minimum Miss Distance")
        if i == 1
            pl0 = ax.axhline(y=0, color="black", linestyle="--", linewidth=1.0)
            push!(handles, pl0)
        end
        rolling_min = []
        current_min = Inf
        for i in 1:max_iters
            if miss_distances[i] < current_min
                current_min = miss_distances[i]
            end
            push!(rolling_min, current_min)
        end
        pl1 = ax.plot(rolling_min, color=colors[i], label="AST")
        ylabel("Miss Distance")
        push!(handles, pl1[1])

        ax.tick_params(labelbottom=false)
        # xscale("log")

        ## Plot 4: Cumulative failures
        ax = fig.add_subplot(4,1,4)
        E = metrics[i].event
        max_iters = length(E)

        title("Cumulative Number of Failure Events")
        ax.plot(cumsum(E[1:max_iters]), color=colors[i])
        if episodic_rewards
            xlabel("Episode")
        else
            xlabel("Evaluation")
        end
        ylabel("Number of Events")

        yscale("log")
        # xscale("log")
    end
    fig.legend(handles, ["Event Horizon", labels...],
               columnspacing=0.8, loc="lower center", bbox_to_anchor=(0.52, 0), fancybox=true, shadow=false, ncol=5)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.13) # <-- Change the 0.02 to work for your plot.
end

# ╔═╡ 7c75f5a0-7dda-11eb-26d9-efc7685e6fb4
episodic_figures_multi([ppo_mdp.metrics, cem_mdp.metrics, mcts_mdp.metrics], ["PPO", "CEM", "MCTS"], ["darkcyan", "blue", "red"], learning_window=100); gcf()

# ╔═╡ 558ba240-7e2b-11eb-33b2-5f3b56462c39
begin
    figure()
    title("Top-paths log-likelihood")
    plot(map(logpdf, [get_top_path(ppo_mdp, k) for k in 1:10]), c="darkcyan")
    plot(map(logpdf, [get_top_path(cem_mdp, k) for k in 1:10]), c="blue")
    plot(map(logpdf, [get_top_path(mcts_mdp, k) for k in 1:10]), c="red")
    legend(["PPO", "CEM", "MCTS"])
    gcf()
end

# ╔═╡ 5c40ea60-7e25-11eb-180c-8b29613eb5a7
begin
    figure()
    title("Mean log-likelihood (events)")
    plot(runmean(mean.(ppo_mdp.metrics.logprob[findall(ppo_mdp.metrics.event)]), 1000), c="darkcyan")
    plot(runmean(mean.(cem_mdp.metrics.logprob[findall(cem_mdp.metrics.event)]), 1000), c="blue")
    plot(runmean(mean.(mcts_mdp.metrics.logprob[findall(mcts_mdp.metrics.event)]), 1000), c="red")
    legend(["PPO", "CEM", "MCTS"])
    gcf()
end

# ╔═╡ 2b991bbe-7e2b-11eb-154d-d5ac85cafbbd
begin
    figure()
    title("Mean log-likelihood (all)")
    plot(runmean(mean.(ppo_mdp.metrics.logprob), 1000), c="darkcyan")
    plot(runmean(mean.(cem_mdp.metrics.logprob), 1000), c="blue")
    plot(runmean(mean.(mcts_mdp.metrics.logprob), 1000), c="red")
    legend(["PPO", "CEM", "MCTS"])
    gcf()
end

# ╔═╡ dd83f630-7df3-11eb-0207-6f1aa3bae255
begin
    figure()
    title("Learning curve")
    plot(runmean(mean.(ppo_mdp.metrics.returns), 1000), c="darkcyan")
    plot(runmean(mean.(cem_mdp.metrics.returns), 1000), c="blue")
    plot(runmean(mean.(mcts_mdp.metrics.returns), 1000), c="red")
    legend(["PPO", "CEM", "MCTS"])
    gcf()
end

# ╔═╡ 54aa7f4e-7e52-11eb-1690-8782ef1e4032
begin
    clf()
    title("\$p_\\tau(s_T \\in E)\$")
    hist([get_top_path(ppo_mdp,i) |> logpdf |> exp for i in 1:100], alpha=0.5, color="darkcyan")
    hist([get_top_path(cem_mdp,i) |> logpdf |> exp for i in 1:100], alpha=0.5, color="blue")
    hist([get_top_path(mcts_mdp,i) |> logpdf |> exp for i in 1:100], alpha=0.5, color="red")
    gcf()
end

# ╔═╡ f92f8b00-7e53-11eb-2249-6b3bc36a59db
begin
    clf()
    title("\$x \\sim p_\\tau(x \\mid s_T \\in E)\$")
    hist([get_top_path(ppo_mdp,i) |> logpdf |> exp for i in 1:100], color="darkcyan")
    hist([get_top_path(cem_mdp,i) |> logpdf |> exp for i in 1:100], color="blue")
    hist([get_top_path(mcts_mdp,i) |> logpdf |> exp for i in 1:100], color="red")
    gcf()
end

# ╔═╡ 0cfc3520-7e54-11eb-20b2-fd968ef3bc62
begin
    space = []
    for s in 1:100
        l_action_trace = get_top_path(ppo_mdp,s)
        push!(space, [l_action_trace[t].sample[k].value for k in keys(l_action_trace[1].sample) for t in 1:length(l_action_trace)]...)
    end
end

# ╔═╡ e98d3430-7e54-11eb-3301-8d6bd86fc2e2
space

# ╔═╡ a0ab1ae2-7f88-11eb-16d5-5d321709bee3
begin
    figure()
    title("Mean miss distance")
    plot(runmean(mean.(ppo_mdp.metrics.miss_distance), 5000), c="darkcyan")
    plot(runmean(mean.(cem_mdp.metrics.miss_distance), 5000), c="blue")
    plot(runmean(mean.(mcts_mdp.metrics.miss_distance), 5000), c="red")
    legend(["PPO", "CEM", "MCTS"])
    xscale("log")
    xlabel("log evaluation")
    gcf()
end

# ╔═╡ 087e4ae0-72df-11eb-0df5-89b1134dae1c
md"""
## State-proxy
"""

# ╔═╡ 801f8080-f631-11ea-0728-f15dddc3ef5d
md"""
## AST Reward Function
The AST reward function gives a reward of $0$ if an event is found, a reward of negative distance if no event is found at termination, and the log-likelihood during the simulation.
"""

# ╔═╡ 8f06f650-f631-11ea-1c52-697060322173
@latexify function R(p,e,d,τ)
    if τ && e
        return 0
    elseif τ && !e
        return -d
    else
        return log(p)
    end
end

# ╔═╡ 9463f6e2-f62a-11ea-1cef-c3fa7d4f19ad
md"""
## References
1. Robert J. Moss, Ritchie Lee, Nicholas Visser, Joachim Hochwarth, James G. Lopez, and Mykel J. Kochenderfer, "Adaptive Stress Testing of Trajectory Predictions in Flight Management Systems", *Digital Avionics Systems Conference, 2020.*
"""

# ╔═╡ 05be9a80-0877-11eb-3d88-efbd306754a2
PlutoUI.TableOfContents("POMDPStressTesting.jl")

# ╔═╡ Cell order:
# ╟─2978b840-f62d-11ea-2ea0-19d7857208b1
# ╟─1e00230e-f630-11ea-3e40-bf8c852f78b8
# ╠═92ce9460-f62b-11ea-1a8c-179776b5a0b4
# ╟─40d3b1e0-f630-11ea-2160-01338d9f2209
# ╟─86f13f60-f62d-11ea-3241-f3f1ffe37d7a
# ╟─d3411dd0-f62e-11ea-27d7-1b2ed8edc415
# ╟─e37d7542-f62e-11ea-0b61-513a4b44fc3c
# ╠═fd7fc880-f62e-11ea-15ac-f5407aeff2a6
# ╟─012c2eb0-f62f-11ea-1637-c113ad01b144
# ╠═0d7049de-f62f-11ea-3552-214fc4e7ec98
# ╟─11e445d0-f62f-11ea-305c-495272981112
# ╠═43c8cb70-f62f-11ea-1b0d-bb04a4176730
# ╟─48a5e970-f62f-11ea-111d-35694f3994b4
# ╠═5d0313c0-f62f-11ea-3d33-9ded1fb804e7
# ╟─6e111310-f62f-11ea-33cf-b5e943b2f088
# ╟─7c84df7e-f62f-11ea-3b5f-8b090654df19
# ╠═9b736bf2-f62f-11ea-0330-69ffafe9f200
# ╟─a380e250-f62f-11ea-363d-2bf2b59d5eed
# ╠═be39db60-f62f-11ea-3a5c-bd57114455ff
# ╟─c48da3f0-72eb-11eb-3251-01c0c6a588d3
# ╠═d1846d50-72eb-11eb-26a3-af886fda1ab1
# ╟─bf8917b0-f62f-11ea-0e77-b58065b0da3e
# ╠═c5f03110-f62f-11ea-1119-81f5c9ec9283
# ╟─c378ef80-f62f-11ea-176d-e96e1be7736e
# ╠═cb5f7cf0-f62f-11ea-34ca-5f0656eddcd4
# ╟─e2f34130-f62f-11ea-220b-c7fc7de2c7e7
# ╠═f6213a50-f62f-11ea-07c7-2dcc383c8042
# ╟─1fca5f1e-72dc-11eb-2e35-5b926d1fd105
# ╠═996a0260-72e9-11eb-1846-8f904861d15e
# ╠═4fe74420-72dc-11eb-063a-1fdd78a5b686
# ╠═ac267500-72e9-11eb-0b4b-415b512b336b
# ╟─01da7aa0-f630-11ea-1262-f50453455766
# ╠═fdf55130-f62f-11ea-33a4-a783b4d216dc
# ╟─09c928f0-f631-11ea-3ef7-512a6bececcc
# ╠═17d0ed20-f631-11ea-2e28-3bb9ca9a445f
# ╠═b9131470-7dda-11eb-232f-c1642f3a7864
# ╠═1c47f652-f631-11ea-15f6-1b9b59700f36
# ╟─21530220-f631-11ea-3994-319c862d51f9
# ╠═3b282ae0-f631-11ea-309d-639bf4411bb3
# ╠═7473adb0-f631-11ea-1c87-0f76b18a9ab6
# ╟─b6244db0-f63a-11ea-3b48-89d427664f5e
# ╠═e15cc1b0-f63a-11ea-2401-5321d48118c3
# ╠═f5a15af0-f63a-11ea-1dd7-593d7cb01ee4
# ╠═fb3fa610-f63a-11ea-2663-17224dc8aade
# ╠═09c9e0b0-f63b-11ea-2d50-4154e3432fa0
# ╟─46b40e10-f63b-11ea-2375-1976bb637d51
# ╠═32fd5cf2-f63b-11ea-263a-39f013ef6d68
# ╟─734b5320-72dc-11eb-3481-954d7d759154
# ╠═b4fa6620-72dd-11eb-121b-f55e92d3a613
# ╠═9f2c3c20-72dc-11eb-31ce-8fde19792bd7
# ╠═a4711ca0-72dc-11eb-2885-e98ad761f964
# ╠═b4654780-72dc-11eb-340e-3f57e471c7aa
# ╠═fbc67c9e-7e1e-11eb-002e-0148bc603efe
# ╠═e192fd80-7ded-11eb-0c33-2dfff616be02
# ╠═93e61380-72dd-11eb-094d-5d2f6acf967a
# ╠═9946ce80-72df-11eb-2bb8-ab675371090c
# ╠═66c23850-72df-11eb-08e2-27b3b5dd523c
# ╠═3dd97c7e-72e6-11eb-21bd-57989a9f53ad
# ╠═ed65d060-7dd9-11eb-084a-f924343f592f
# ╠═2f501370-7de2-11eb-024b-07af757fc74a
# ╠═2f75bd80-7dda-11eb-049d-7f3392f0f2c1
# ╠═7c75f5a0-7dda-11eb-26d9-efc7685e6fb4
# ╠═558ba240-7e2b-11eb-33b2-5f3b56462c39
# ╠═5c40ea60-7e25-11eb-180c-8b29613eb5a7
# ╠═2b991bbe-7e2b-11eb-154d-d5ac85cafbbd
# ╠═dd83f630-7df3-11eb-0207-6f1aa3bae255
# ╠═54aa7f4e-7e52-11eb-1690-8782ef1e4032
# ╠═f92f8b00-7e53-11eb-2249-6b3bc36a59db
# ╠═0cfc3520-7e54-11eb-20b2-fd968ef3bc62
# ╠═e98d3430-7e54-11eb-3301-8d6bd86fc2e2
# ╠═a0ab1ae2-7f88-11eb-16d5-5d321709bee3
# ╠═087e4ae0-72df-11eb-0df5-89b1134dae1c
# ╠═3cd27530-7e49-11eb-23b0-313cf51d1c7f
# ╠═801f8080-f631-11ea-0728-f15dddc3ef5d
# ╠═8f06f650-f631-11ea-1c52-697060322173
# ╟─9463f6e2-f62a-11ea-1cef-c3fa7d4f19ad
# ╠═05be9a80-0877-11eb-3d88-efbd306754a2
