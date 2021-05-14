using AdversarialDriving
using AutomotiveSimulator
using AutomotiveVisualization
using CrossEntropyVariants
using Crux
using Distributions
using Flux
using Latexify
using Parameters
using PlutoUI
using POMDPPolicies
using POMDPs
using POMDPSimulators
using POMDPStressTesting
using PyPlot
using Random
using Statistics
using STLCG


## Adaptive Stress Testing
# Formulation of the autonomous vehicle risk problem using AST.
# The following code will automatically download any dependent packages
# (see the Pluto console output for debug information).

Random.seed!(0) # reproducibility

## Automotive Driving Problem
### Adversarial Driver
# This section provides the crosswalk example and visualizations
# to understand the problem AST is trying to solve. It involves an
# autonomous vehicle (i.e. ego vehicle) and a noisy pedestrian crossing a crosswalk.

Base.rand(rng::AbstractRNG, s::Scene) = s # TODO: thought this was fixed already?

# include("agent_example.jl")


## Black-Box Stress Testing
# To find failures in a black-box autonomous system,
# we can use the `POMDPStressTesting` package which
# is part of the POMDPs.jl ecosystem.

# Various solvers—which adhere to the POMDPs.jl interface—can be used:
# - `MCTSPWSolver` (MCTS with action progressive widening)
# - `TRPOSolver` and `PPOSolver` (deep reinforcement learning policy optimization)
# - `CEMSolver` (cross-entropy method)
# - `RandomSearchSolver` (standard Monte Carlo random search)

## Working Problem: Pedestrian in a Crosswalk
# We define a simple problem for adaptive stress testing (AST)
# to find failures. This problem, not colliding with a pedestrian
# in a crosswalk, samples random noise disturbances applied to the
# pedestrian's position and velocity from standard normal distributions
# $\mathcal{N}(\mu,\sigma)$. A failure is defined as a collision. AST
# will either select the seed which deterministically controls the sampled
# value from the distribution (i.e. from the transition model) or will
# directly sample the provided environmental distributions. These action
# modes are determined by the seed-action or sample-action options
# (`ASTSeedAction` and `ASTSampleAction`, respectively). AST will guide
# the simulation to failure events using a measure of distance to failure,
# while simultaneously trying to find the set of actions that maximizes
# the log-likelihood of the samples.


# -----

# TODO: replace with "ast.jl"
include("scratch/AVGrayBox.jl")
include("scratch/AVBlackBox.jl")
# include("example.jl")
include("rollouts.jl")


## AST Setup and Running
# Setting up our simulation, we instantiate our simulation object
# and pass that to the Markov decision proccess (MDP) object of the
# adaptive stress testing formulation. We use Monte Carlo tree search (MCTS)
# with progressive widening on the action space as our solver. Hyperparameters
# are passed to `MCTSPWSolver`, which is a simple wrapper around the POMDPs.jl
# implementation of MCTS. Lastly, we solve the MDP to produce a planner.
# Note we are using the `ASTSampleAction`.
function setup_ast(seed=0)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = AutoRiskSim()

    # AST MDP formulation object
    mdp::ASTMDP = ASTMDP{ASTSampleAction}(sim)
    mdp.params.debug = true # record metrics
    mdp.params.top_k = 10   # record top k best trajectories
    mdp.params.seed = seed  # set RNG seed for determinism

    # Hyperparameters for MCTS-PW as the solver
    solver = MCTSPWSolver(n_iterations=10,        # number of algorithm iterations
                          exploration_constant=1.0, # UCT exploration
                          k_action=1.0,             # action widening
                          alpha_action=0.95,         # action widening
                          depth=sim.params.endtime, # tree depth
                          # estimate_value=ϵ_rollout) # rollout function
                          estimate_value=cem_rollout) # rollout function
                          # estimate_value=prior_rollout) # rollout function

    # Get online planner (no work done, yet)
    planner = solve(solver, mdp)

    return planner
end

#### Searching for Failures
# After setup, we search for failures using the planner and output the best action trace.
planner = setup_ast()
action_trace = search!(planner)


### Figures
# These plots show episodic-based metrics, miss distance, and log-likelihood distributions.
# episodic_figures(planner.mdp) POMDPStressTesting.gcf()
# distribution_figures(planner.mdp) POMDPStressTesting.gcf()

#### Playback
# We can also playback specific trajectories and print intermediate distance values.
playback_trace = playback(planner, action_trace, BlackBox.distance, return_trace=true)
failure_rate = print_metrics(planner)
# visualize(planner)

### Other Solvers: Cross-Entropy Method
# We can easily take our `ASTMDP` object (`planner.mdp`) and re-solve the MDP using a different solver—in this case the `CEMSolver`.
ast_mdp = deepcopy(planner.mdp) # re-used from MCTS run.
@info actions(ast_mdp) |> length

begin
    # TODO: get this index from the `trace` itself
    # findmax(ast_mdp.metrics.reward[ast_mdp.metrics.event])
    # findmax(ast_mdp.metrics.reward[ast_mdp.metrics.event])

    failure_likelihood_mcts =
        round(exp(maximum(planner.mdp.metrics.logprob[ast_mdp.metrics.event])), digits=4)
    @info failure_likelihood_mcts
end

cem_solver = CEMSolver(n_iterations=100, episode_length=ast_mdp.sim.params.endtime)
cem_planner = solve(cem_solver, ast_mdp)

run_cem = true
if run_cem
    global cem_action_trace = search!(cem_planner)
end

# Notice the failure rate is higher when using `CEMSolver` than `MCTSPWSolver`.
cem_failure_rate = print_metrics(cem_planner)
# episodic_figures(cem_planner.mdp); POMDPStressTesting.gcf()
# distribution_figures(cem_planner.mdp); POMDPStressTesting.gcf()

## PPO solver
# ppo_solver = PPOSolver(num_episodes=100, episode_length=ast_mdp.sim.params.endtime)
# ast_mdp_ppo = deepcopy(planner.mdp) # re-used from MCTS run.
# ppo_planner = solve(ppo_solver, ast_mdp_ppo)
# run_ppo = false
# if run_ppo
#     global ppo_action_trace = search!(ppo_planner)
# end
# ppo_failure_rate = print_metrics(ppo_planner)

## Random baseline
run_rand = false

if run_rand
    rand_solver = RandomSearchSolver(n_iterations=100,
                                     episode_length=ast_mdp.sim.params.endtime)
    ast_mdp_rand = deepcopy(planner.mdp) # re-used from MCTS run.
    # ast_mdp_rand.params.seed  = 0
    rand_planner = solve(rand_solver, ast_mdp_rand)
    rand_action_trace = search!(rand_planner)
    rand_failure_rate = print_metrics(rand_planner)
end

## Visualization of failure
# We can visualize the failure with the highest likelihood found by AST.

begin
    # TODO: get this index from the `trace` itself
    # findmax(ast_mdp.metrics.reward[ast_mdp.metrics.event])
    # findmax(ast_mdp.metrics.reward[ast_mdp.metrics.event])

    failure_likelihood =
        round(exp(maximum(ast_mdp.metrics.logprob[ast_mdp.metrics.event])), digits=4)

    @info failure_likelihood
end

roadway = cem_planner.mdp.sim.problem.roadway
cem_trace = playback(cem_planner, cem_action_trace, sim->sim.state, return_trace=true)
fail_t = length(cem_trace) # 1:length(cem_trace)
AutomotiveVisualization.render([roadway, crosswalk, cem_trace[fail_t]])
