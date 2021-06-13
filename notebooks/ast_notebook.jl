### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ e59459de-047d-11eb-3252-25dc1bac624c
try
	using Pkg
	using AddPackage
catch
	Pkg.add("AddPackage")
	using AddPackage
end

# ╔═╡ e7783510-047d-11eb-26b0-373b14638ff0
try
	using POMDPStressTesting
catch
	pkg"add https://github.com/JuliaPOMDP/RLInterface.jl"
	pkg"dev https://github.com/sisl/POMDPStressTesting.jl"
	using POMDPStressTesting
end

# ╔═╡ cda24800-0488-11eb-1d7f-8d52b5f6b33e
try
	using AdversarialDriving
catch
	pkg"add https://github.com/sisl/AdversarialDriving.jl"
	using AdversarialDriving
end

# ╔═╡ 92ce9460-f62b-11ea-1a8c-179776b5a0b4
@add using Distributions, Parameters, Random, Latexify, PlutoUI

# ╔═╡ 687a90c0-0480-11eb-1ef4-03e93ffa400c
@add using AutomotiveSimulator, AutomotiveVisualization

# ╔═╡ 3617eb60-0489-11eb-144a-232b222a0365
@add using POMDPs, POMDPPolicies, POMDPSimulators

# ╔═╡ 83e51830-f62a-11ea-215d-a9767d7b07a5
md"""
# Adaptive Stress Testing
Formulation of the autonomous vehicle risk problem using AST. The following code will automatically download any dependent packages (see the Pluto console output for debug information).
"""

# ╔═╡ 9117e2d0-05c5-11eb-3d46-6d4be8e47477
Random.seed!(0); # reproducibility

# ╔═╡ b7f29190-047e-11eb-28b1-559809f831f3
md"**Note**: *AST installation may take a few minutes the first time.*"

# ╔═╡ 92a94b80-05b5-11eb-2f9c-871e36ad52ec
md"Notice we `dev` the `POMDPStressTesting` package, this puts the code in `~/.julia/dev`."

# ╔═╡ 5d084480-0480-11eb-1329-c562528e965c
md"## Automotive Driving Problem"

# ╔═╡ 2e62d8c0-048a-11eb-1822-4b40f0acd39b
md"""
### Adversarial Driver
This section provides the crosswalk example and visualizations to understand the problem AST is trying to solve. It involves an autonomous vehicle (i.e. ego vehicle) and a noisy pedestrian crossing a crosswalk.
"""

# ╔═╡ 5f39b460-0489-11eb-2b4f-8168c3825150
Base.rand(rng::AbstractRNG, s::Scene) = s

# ╔═╡ a4a6b3e0-05b5-11eb-1a28-f528c5f88ee1
md"Again, notice we are `dev`-ing the `AdversarialDriving` package, in case we want to make changes."

# ╔═╡ a9420670-04cf-11eb-33d4-89ba4afbe562
md"##### Agents"

# ╔═╡ cefe6ab0-048a-11eb-0622-4b4e71cb0072
md"Define the system under test (SUT) agent."

# ╔═╡ 2bd501f0-048a-11eb-3921-65a864fa990f
sut_agent = BlinkerVehicleAgent(get_ped_vehicle(id=1, s=5.0, v=15.0),
	                            TIDM(ped_TIDM_template, noisy_observations=true));

# ╔═╡ ddd45450-048a-11eb-0561-e769f54a359c
md"Define the adversary, i.e. a noisy pedestrian"

# ╔═╡ 55b0f332-048a-11eb-0733-5b98489ea1cc
adv_ped = NoisyPedestrianAgent(get_pedestrian(id=2, s=7.0, v=2.0),
	                           AdversarialPedestrian());

# ╔═╡ b51a1960-04cf-11eb-169e-69fc0008aedc
md"##### Markov Decision Process"

# ╔═╡ e7b02210-048a-11eb-33f6-010ce4d1e341
md"Instantiate the MDP structure, using the `ped_roadway` exported by `AdversarialDriving`."

# ╔═╡ 5cc03370-048a-11eb-1d27-c71efabeffdd
ad_mdp = AdversarialDrivingMDP(sut_agent, [adv_ped], ped_roadway, 0.1);

# ╔═╡ bfbba820-04cf-11eb-3879-31d8398c9545
md"##### Behaviors/Actions"

# ╔═╡ 0a936850-048b-11eb-1fc8-13a204c0c7c0
md"Define the action that controls the pedestrian (no noise)."

# ╔═╡ 601e79a0-048a-11eb-3c3e-d7d9fb813922
null_action = Disturbance[PedestrianControl()];

# ╔═╡ 35d2ba20-048b-11eb-0f24-71380027dad4
md"Define the actions that controls a *noisy* pedestrian"

# ╔═╡ 628f0470-048a-11eb-1a11-6bc32f2c3d1c
noisy_action = Disturbance[PedestrianControl(noise=Noise((-10.0, 0.0), -2))];

# ╔═╡ 3cad38a0-05a5-11eb-3f6b-735eb1c3cb59
initialstate(ad_mdp)

# ╔═╡ c6cc6f00-04cf-11eb-263c-c34c6a95db29
md"##### Simulation Histories"

# ╔═╡ 4af33470-048b-11eb-0f8c-f93c7dcdb11b
md"Run a simulation for the nominal behavior (i.e. no noisy)."

# ╔═╡ 6bd32610-048a-11eb-316d-6dd779f7cdc4
# Nominal Behavior
hist = POMDPSimulators.simulate(HistoryRecorder(), ad_mdp,
	                            FunctionPolicy((s) -> null_action));

# ╔═╡ 573ec9b0-048b-11eb-3b97-17d0bbb8d28b
md"Run a simulation with the noisy pedestrian."

# ╔═╡ 766dd700-048a-11eb-0faa-ed69d2203b0a
# Behavior with noise
hist_noise = POMDPSimulators.simulate(HistoryRecorder(), ad_mdp,
	                                  FunctionPolicy((s) -> noisy_action));

# ╔═╡ d45ce322-0489-11eb-2b9d-71a00e65d8b0
ad_scenes = state_hist(hist);

# ╔═╡ fb4732b0-0489-11eb-3e24-3d6ed2221771
ad_scenes_noise = state_hist(hist_noise);

# ╔═╡ 61b258b0-048d-11eb-0e0c-9d3f8c23b0ed
md"##### Distance Metrics"

# ╔═╡ 2d5e59b0-048d-11eb-257d-076254d3488f
function distance(ent1::Entity, ent2::Entity)
	pos1 = posg(ent1)
	pos2 = posg(ent2)
	hypot(pos1.x - pos2.x, pos1.y - pos2.y)
end

# ╔═╡ 900269c0-0489-11eb-0031-c5f78fc2963a
@bind ad_t Slider(1:length(ad_scenes), default=12) # known failure at 12s

# ╔═╡ 4c4b70c0-05a4-11eb-1530-5174e460580b
try
	nominal_vehicle, nominal_predestrian = ad_scenes[ad_t]
	distance(nominal_vehicle, nominal_predestrian)
catch
	NaN
end

# ╔═╡ 73b27d00-0489-11eb-2db1-51b4c3966b8d
AutomotiveVisualization.render([ad_mdp.roadway, crosswalk, ad_scenes[ad_t]])

# ╔═╡ f7f8cb00-0489-11eb-3758-c7ae4acaf16c
begin
	capped_t = min(ad_t, length(ad_scenes_noise))
	AutomotiveVisualization.render([ad_mdp.roadway, crosswalk,
			                        ad_scenes_noise[capped_t]])
end

# ╔═╡ 7221b840-05a4-11eb-1982-2faa93fbd308
noisy_vehicle, noisy_pedestrian = ad_scenes_noise[capped_t];

# ╔═╡ 9dd57770-048b-11eb-0078-8b3a21b9bc4a
distance(noisy_vehicle, noisy_pedestrian)

# ╔═╡ 4d5e0420-05a2-11eb-19c5-8979d9423450
collision_checker(noisy_vehicle, noisy_pedestrian)

# ╔═╡ 2978b840-f62d-11ea-2ea0-19d7857208b1
md"""
# Black-Box Stress Testing
"""

# ╔═╡ 40d3b1e0-f630-11ea-2160-01338d9f2209
md"""
To find failures in a black-box autonomous system, we can use the `POMDPStressTesting` package which is part of the POMDPs.jl ecosystem.

Various solvers—which adhere to the POMDPs.jl interface—can be used:
- `MCTSPWSolver` (MCTS with action progressive widening)
- `TRPOSolver` and `PPOSolver` (deep reinforcement learning policy optimization)
- `CEMSolver` (cross-entropy method)
- `RandomSearchSolver` (standard Monte Carlo random search)
"""

# ╔═╡ 86f13f60-f62d-11ea-3241-f3f1ffe37d7a
md"""
## Working Problem: Pedestrian in a Crosswalk
We define a simple problem for adaptive stress testing (AST) to find failures. This problem, not colliding with a pedestrian in a crosswalk, samples random noise disturbances applied to the pedestrian's position and velocity from standard normal distributions $\mathcal{N}(\mu,\sigma)$. A failure is defined as a collision. AST will either select the seed which deterministically controls the sampled value from the distribution (i.e. from the transition model) or will directly sample the provided environmental distributions. These action modes are determined by the seed-action or sample-action options (`ASTSeedAction` and `ASTSampleAction`, respectively). AST will guide the simulation to failure events using a measure of distance to failure, while simultaneously trying to find the set of actions that maximizes the log-likelihood of the samples.
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
@with_kw struct AutoRiskParams
	endtime::Real = 30 # Simulate end time
end;

# ╔═╡ 012c2eb0-f62f-11ea-1637-c113ad01b144
md"""
##### Simulation
Next, we define a `GrayBox.Simulation` structure.
"""

# ╔═╡ 0d7049de-f62f-11ea-3552-214fc4e7ec98
@with_kw mutable struct AutoRiskSim <: GrayBox.Simulation
    t::Real = 0 # Current time
    params::AutoRiskParams = AutoRiskParams() # Parameters

	# System under test, ego vehicle
	sut = BlinkerVehicleAgent(get_ped_vehicle(id=1, s=5.0, v=15.0),
                              TIDM(ped_TIDM_template, noisy_observations=true))

	# Noisy adversary, pedestrian
	adversary = NoisyPedestrianAgent(get_pedestrian(id=2, s=7.0, v=2.0),
	                                 AdversarialPedestrian())

	# Adversarial Markov decision process
	problem::MDP = AdversarialDrivingMDP(sut, [adversary], ped_roadway, 0.1)
	state::Scene = rand(initialstate(problem))
	prev_distance::Real = -Inf # Used when agent goes out of frame

	# Noise distributions and disturbances
	xposition_noise::Distribution = Normal(0, 5) # Gaussian noise (notice larger σ)
	yposition_noise::Distribution = Normal(0, 1) # Gaussian noise
	velocity_noise::Distribution = Normal(0, 1) # Gaussian noise
	disturbances = Disturbance[PedestrianControl()] # Initial 0-noise disturbance
end;

# ╔═╡ 63326db0-05b9-11eb-0efe-ebd0e7cf3d17
md"**Note**: I avoid `MvNormal` (multivariate Gaussian) for the position noise, I'm submitting a change to the `CrossEntropyMethod` package that fixes this."

# ╔═╡ 11e445d0-f62f-11ea-305c-495272981112
md"""
#### GrayBox.environment
Then, we define our `GrayBox.Environment` distributions. When using the `ASTSampleAction`, as opposed to `ASTSeedAction`, we need to provide access to the sampleable environment.
"""

# ╔═╡ 43c8cb70-f62f-11ea-1b0d-bb04a4176730
function GrayBox.environment(sim::AutoRiskSim)
	return GrayBox.Environment(:xpos => sim.xposition_noise,
							   :ypos => sim.yposition_noise,
		                       :vel => sim.velocity_noise)
end

# ╔═╡ 48a5e970-f62f-11ea-111d-35694f3994b4
md"""
#### GrayBox.transition!
We override the `transition!` function from the `GrayBox` interface, which takes an environment sample as input. We apply the sample in our simulator, take a step, and return the log-likelihood.
"""

# ╔═╡ 5d0313c0-f62f-11ea-3d33-9ded1fb804e7
function GrayBox.transition!(sim::AutoRiskSim, sample::GrayBox.EnvironmentSample)
    sim.t += sim.problem.dt # Keep track of time

	# replace current noise with new sampled noise
	noise = Noise((sample[:xpos].value, sample[:ypos].value), sample[:vel].value)
	sim.disturbances[1] = PedestrianControl(noise=noise)

	# step agents: given MDP, current state, and current action (i.e. disturbances)
	(sim.state, r) = @gen(:sp, :r)(sim.problem, sim.state, sim.disturbances)

	# return log-likelihood of actions, summation handled by `logpdf()`
	return logpdf(sample)::Real
end

# ╔═╡ 4d964d00-05b4-11eb-32d0-11df579faaa9
md"""
## Example
You can use this space to play around with the `GrayBox` and `BlackBox` interface functions.
"""

# ╔═╡ d0c31180-05b0-11eb-159a-2702ed171fcf
simx = AutoRiskSim()

# ╔═╡ 56f103e0-05b4-11eb-2de6-8f6daace22b6
md"**Example**: initializing the AST simulation object."

# ╔═╡ ee99f6f0-05b1-11eb-186d-eb9039f0cfae
md"**Example**: sampling from the environment, applying a state transition, and calculating the distance."

# ╔═╡ 965a6212-05b4-11eb-256a-63b6d10fb951
md"**Example**: or we could call `evaluate!` to do these step for us."

# ╔═╡ 6e111310-f62f-11ea-33cf-b5e943b2f088
md"""
## Black-Box System
The system under test, in this case an autonomous vehicle with sensor noise, is treated as black-box. The following interface functions are overridden to minimally interact with the system, and use outputs from the system to determine failure event indications and distance metrics.
"""

# ╔═╡ 7c84df7e-f62f-11ea-3b5f-8b090654df19
md"""
#### BlackBox.initialize!
Now we override the `BlackBox` interface, starting with the function that initializes the simulation object. Interface functions ending in `!` may modify the `sim` object in place.
"""

# ╔═╡ 9b736bf2-f62f-11ea-0330-69ffafe9f200
function BlackBox.initialize!(sim::AutoRiskSim)
    sim.t = 0
    sim.problem = AdversarialDrivingMDP(sim.sut, [sim.adversary], ped_roadway, 0.1)
	sim.state = rand(initialstate(sim.problem))
	sim.disturbances = Disturbance[PedestrianControl()] # noise-less
	sim.prev_distance = -Inf
end

# ╔═╡ 3df1c8c0-05b4-11eb-0407-89c259b45c10
BlackBox.initialize!(simx);

# ╔═╡ 9d41f840-05c3-11eb-2395-0f4a9f68e3bc
out_of_frame(sim) = length(sim.state.entities) < 2 # either agent went out of frame

# ╔═╡ a380e250-f62f-11ea-363d-2bf2b59d5eed
md"""
#### BlackBox.distance
We define how close we are to a failure event using a non-negative distance metric.
"""

# ╔═╡ be39db60-f62f-11ea-3a5c-bd57114455ff
function BlackBox.distance(sim::AutoRiskSim)
	if out_of_frame(sim)
		return sim.prev_distance
	else
		pedestrian, vehicle = sim.state.entities
		pos1 = posg(pedestrian)
		pos2 = posg(vehicle)
		return hypot(pos1.x - pos2.x, pos1.y - pos2.y)
	end
end

# ╔═╡ adef6630-05b1-11eb-269f-a10c49a437ee
begin
	envsample = rand(GrayBox.environment(simx))
	GrayBox.transition!(simx, envsample)
	BlackBox.distance(simx)
end

# ╔═╡ bf8917b0-f62f-11ea-0e77-b58065b0da3e
md"""
#### BlackBox.isevent
We indicate whether a failure event occurred, using `collision_checker` from `AutomotiveSimulator`.
"""

# ╔═╡ c5f03110-f62f-11ea-1119-81f5c9ec9283
function BlackBox.isevent(sim::AutoRiskSim)
	if out_of_frame(sim)
		return false
	else
		pedestrian, vehicle = sim.state.entities
		return collision_checker(pedestrian, vehicle)
	end
end

# ╔═╡ c378ef80-f62f-11ea-176d-e96e1be7736e
md"""
#### BlackBox.isterminal
Similarly, we define an indication that the simulation is in a terminal state.
"""

# ╔═╡ cb5f7cf0-f62f-11ea-34ca-5f0656eddcd4
function BlackBox.isterminal(sim::AutoRiskSim)
    return isterminal(sim.problem, sim.state) ||
		   out_of_frame(sim) ||
		   BlackBox.isevent(sim) ||
	       sim.t ≥ sim.params.endtime
end

# ╔═╡ e2f34130-f62f-11ea-220b-c7fc7de2c7e7
md"""
#### BlackBox.evaluate!
Lastly, we use our defined interface to evaluate the system under test. Using the input sample, we return the log-likelihood, distance to an event, and event indication.
"""

# ╔═╡ f6213a50-f62f-11ea-07c7-2dcc383c8042
function BlackBox.evaluate!(sim::AutoRiskSim, sample::GrayBox.EnvironmentSample)
    logprob::Real = GrayBox.transition!(sim, sample) # Step simulation
    d::Real       = BlackBox.distance(sim)           # Calculate miss distance
    event::Bool   = BlackBox.isevent(sim)            # Check event indication
    sim.prev_distance = d                            # Store previous distance
	return (logprob::Real, d::Real, event::Bool)
end

# ╔═╡ c6a61f40-05b4-11eb-1f1d-6950aaea7a8d
begin
	envsample2 = rand(GrayBox.environment(simx))
	BlackBox.evaluate!(simx, envsample2) # (log-likelihood, distance, isevent)
end

# ╔═╡ 01da7aa0-f630-11ea-1262-f50453455766
md"""
## AST Setup and Running
Setting up our simulation, we instantiate our simulation object and pass that to the Markov decision proccess (MDP) object of the adaptive stress testing formulation. We use Monte Carlo tree search (MCTS) with progressive widening on the action space as our solver. Hyperparameters are passed to `MCTSPWSolver`, which is a simple wrapper around the POMDPs.jl implementation of MCTS. Lastly, we solve the MDP to produce a planner. Note we are using the `ASTSampleAction`.
"""

# ╔═╡ fdf55130-f62f-11ea-33a4-a783b4d216dc
function setup_ast(seed=0)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = AutoRiskSim()

    # AST MDP formulation object
    mdp::ASTMDP = ASTMDP{ASTSampleAction}(sim)
    mdp.params.debug = true # record metrics
    mdp.params.top_k = 10   # record top k best trajectories
    mdp.params.seed = seed  # set RNG seed for determinism

    # Hyperparameters for MCTS-PW as the solver
    solver = MCTSPWSolver(n_iterations=1000,        # number of algorithm iterations
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
#### Searching for Failures
After setup, we search for failures using the planner and output the best action trace.
"""

# ╔═╡ 17d0ed20-f631-11ea-2e28-3bb9ca9a445f
planner = setup_ast();

# ╔═╡ 1c47f652-f631-11ea-15f6-1b9b59700f36
with_terminal() do
	global action_trace = search!(planner)
end

# ╔═╡ 6ade4d00-05c2-11eb-3732-ff945f7ce127
md"""
### Figures
These plots show episodic-based metrics, miss distance, and log-likelihood distributions.
"""

# ╔═╡ 8efa3720-05be-11eb-2c3e-9519eb7d8e7a
episodic_figures(planner.mdp); POMDPStressTesting.gcf()

# ╔═╡ 22995300-05c2-11eb-3399-574d1fb2ed94
distribution_figures(planner.mdp); POMDPStressTesting.gcf()

# ╔═╡ 21530220-f631-11ea-3994-319c862d51f9
md"""
#### Playback
We can also playback specific trajectories and print intermediate distance values.
"""

# ╔═╡ 3b282ae0-f631-11ea-309d-639bf4411bb3
playback_trace = playback(planner, action_trace, BlackBox.distance, return_trace=true)

# ╔═╡ 7473adb0-f631-11ea-1c87-0f76b18a9ab6
failure_rate = print_metrics(planner)

# ╔═╡ b6244db0-f63a-11ea-3b48-89d427664f5e
md"""
### Other Solvers: Cross-Entropy Method
We can easily take our `ASTMDP` object (`planner.mdp`) and re-solve the MDP using a different solver—in this case the `CEMSolver`.
"""

# ╔═╡ c0cf83e0-05a5-11eb-32b5-6fb00cbc311b
ast_mdp = deepcopy(planner.mdp); # re-used from MCTS run.

# ╔═╡ 824bdde0-05bd-11eb-0594-cddd54c49757
cem_solver = CEMSolver(n_iterations=100, episode_length=ast_mdp.sim.params.endtime)

# ╔═╡ fb3fa610-f63a-11ea-2663-17224dc8aade
cem_planner = solve(cem_solver, ast_mdp);

# ╔═╡ 09c9e0b0-f63b-11ea-2d50-4154e3432fa0
with_terminal() do
	global cem_action_trace = search!(cem_planner)
end

# ╔═╡ 46b40e10-f63b-11ea-2375-1976bb637d51
md"Notice the failure rate is higher when using `CEMSolver` than `MCTSPWSolver`."

# ╔═╡ de88b710-05c5-11eb-1795-a119590ad1c2
cem_failure_rate = print_metrics(cem_planner)

# ╔═╡ 00dd9240-05c1-11eb-3d13-ff544dc94b5d
md"""
## Visualization of failure
We can visualize the failure with the highest likelihood found by AST.
"""

# ╔═╡ 49bb9090-05c4-11eb-1aa9-8b4488a05654
begin
	# TODO: get this index from the `trace` itself
	# findmax(ast_mdp.metrics.reward[ast_mdp.metrics.event])
	# findmax(ast_mdp.metrics.reward[ast_mdp.metrics.event])

	failure_likelihood =
		round(exp(maximum(ast_mdp.metrics.logprob[ast_mdp.metrics.event])), digits=4)

	Markdown.parse(string("\$\$p = ", failure_likelihood, "\$\$"))
end

# ╔═╡ 066f2bd0-05c4-11eb-032f-ad141ecd8070
roadway = cem_planner.mdp.sim.problem.roadway;

# ╔═╡ 937c8392-05c1-11eb-0de5-191f4a5c2d8c
cem_trace = playback(cem_planner, cem_action_trace, sim->sim.state, return_trace=true)

# ╔═╡ 25c274e0-05c1-11eb-3c1d-a591fde9722b
@bind fail_t Slider(1:length(cem_trace), default=length(cem_trace)) # ends in failure

# ╔═╡ 06bf27ee-05c1-11eb-06ed-af4265dee892
AutomotiveVisualization.render([roadway, crosswalk, cem_trace[fail_t]])

# ╔═╡ 801f8080-f631-11ea-0728-f15dddc3ef5d
md"""
## AST Reward Function
For reference, the AST reward function gives a reward of $0$ if an event is found, a reward of negative distance $d$ if no event is found at termination, and the log-likelihood $\log(p)$ during the simulation.
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

# ╔═╡ Cell order:
# ╟─83e51830-f62a-11ea-215d-a9767d7b07a5
# ╠═e59459de-047d-11eb-3252-25dc1bac624c
# ╠═92ce9460-f62b-11ea-1a8c-179776b5a0b4
# ╠═9117e2d0-05c5-11eb-3d46-6d4be8e47477
# ╟─b7f29190-047e-11eb-28b1-559809f831f3
# ╠═e7783510-047d-11eb-26b0-373b14638ff0
# ╟─92a94b80-05b5-11eb-2f9c-871e36ad52ec
# ╟─5d084480-0480-11eb-1329-c562528e965c
# ╠═687a90c0-0480-11eb-1ef4-03e93ffa400c
# ╟─2e62d8c0-048a-11eb-1822-4b40f0acd39b
# ╠═cda24800-0488-11eb-1d7f-8d52b5f6b33e
# ╠═3617eb60-0489-11eb-144a-232b222a0365
# ╠═5f39b460-0489-11eb-2b4f-8168c3825150
# ╟─a4a6b3e0-05b5-11eb-1a28-f528c5f88ee1
# ╟─a9420670-04cf-11eb-33d4-89ba4afbe562
# ╟─cefe6ab0-048a-11eb-0622-4b4e71cb0072
# ╠═2bd501f0-048a-11eb-3921-65a864fa990f
# ╟─ddd45450-048a-11eb-0561-e769f54a359c
# ╠═55b0f332-048a-11eb-0733-5b98489ea1cc
# ╟─b51a1960-04cf-11eb-169e-69fc0008aedc
# ╟─e7b02210-048a-11eb-33f6-010ce4d1e341
# ╠═5cc03370-048a-11eb-1d27-c71efabeffdd
# ╟─bfbba820-04cf-11eb-3879-31d8398c9545
# ╟─0a936850-048b-11eb-1fc8-13a204c0c7c0
# ╠═601e79a0-048a-11eb-3c3e-d7d9fb813922
# ╟─35d2ba20-048b-11eb-0f24-71380027dad4
# ╠═628f0470-048a-11eb-1a11-6bc32f2c3d1c
# ╠═3cad38a0-05a5-11eb-3f6b-735eb1c3cb59
# ╟─c6cc6f00-04cf-11eb-263c-c34c6a95db29
# ╟─4af33470-048b-11eb-0f8c-f93c7dcdb11b
# ╠═6bd32610-048a-11eb-316d-6dd779f7cdc4
# ╟─573ec9b0-048b-11eb-3b97-17d0bbb8d28b
# ╠═766dd700-048a-11eb-0faa-ed69d2203b0a
# ╠═d45ce322-0489-11eb-2b9d-71a00e65d8b0
# ╠═fb4732b0-0489-11eb-3e24-3d6ed2221771
# ╟─61b258b0-048d-11eb-0e0c-9d3f8c23b0ed
# ╠═2d5e59b0-048d-11eb-257d-076254d3488f
# ╠═4c4b70c0-05a4-11eb-1530-5174e460580b
# ╠═7221b840-05a4-11eb-1982-2faa93fbd308
# ╠═9dd57770-048b-11eb-0078-8b3a21b9bc4a
# ╠═4d5e0420-05a2-11eb-19c5-8979d9423450
# ╠═900269c0-0489-11eb-0031-c5f78fc2963a
# ╠═73b27d00-0489-11eb-2db1-51b4c3966b8d
# ╠═f7f8cb00-0489-11eb-3758-c7ae4acaf16c
# ╟─2978b840-f62d-11ea-2ea0-19d7857208b1
# ╟─40d3b1e0-f630-11ea-2160-01338d9f2209
# ╟─86f13f60-f62d-11ea-3241-f3f1ffe37d7a
# ╟─d3411dd0-f62e-11ea-27d7-1b2ed8edc415
# ╟─e37d7542-f62e-11ea-0b61-513a4b44fc3c
# ╠═fd7fc880-f62e-11ea-15ac-f5407aeff2a6
# ╟─012c2eb0-f62f-11ea-1637-c113ad01b144
# ╠═0d7049de-f62f-11ea-3552-214fc4e7ec98
# ╟─63326db0-05b9-11eb-0efe-ebd0e7cf3d17
# ╟─11e445d0-f62f-11ea-305c-495272981112
# ╠═43c8cb70-f62f-11ea-1b0d-bb04a4176730
# ╟─48a5e970-f62f-11ea-111d-35694f3994b4
# ╠═5d0313c0-f62f-11ea-3d33-9ded1fb804e7
# ╟─4d964d00-05b4-11eb-32d0-11df579faaa9
# ╠═d0c31180-05b0-11eb-159a-2702ed171fcf
# ╟─56f103e0-05b4-11eb-2de6-8f6daace22b6
# ╠═3df1c8c0-05b4-11eb-0407-89c259b45c10
# ╟─ee99f6f0-05b1-11eb-186d-eb9039f0cfae
# ╠═adef6630-05b1-11eb-269f-a10c49a437ee
# ╟─965a6212-05b4-11eb-256a-63b6d10fb951
# ╠═c6a61f40-05b4-11eb-1f1d-6950aaea7a8d
# ╟─6e111310-f62f-11ea-33cf-b5e943b2f088
# ╟─7c84df7e-f62f-11ea-3b5f-8b090654df19
# ╠═9b736bf2-f62f-11ea-0330-69ffafe9f200
# ╠═9d41f840-05c3-11eb-2395-0f4a9f68e3bc
# ╟─a380e250-f62f-11ea-363d-2bf2b59d5eed
# ╠═be39db60-f62f-11ea-3a5c-bd57114455ff
# ╟─bf8917b0-f62f-11ea-0e77-b58065b0da3e
# ╠═c5f03110-f62f-11ea-1119-81f5c9ec9283
# ╟─c378ef80-f62f-11ea-176d-e96e1be7736e
# ╠═cb5f7cf0-f62f-11ea-34ca-5f0656eddcd4
# ╟─e2f34130-f62f-11ea-220b-c7fc7de2c7e7
# ╠═f6213a50-f62f-11ea-07c7-2dcc383c8042
# ╟─01da7aa0-f630-11ea-1262-f50453455766
# ╠═fdf55130-f62f-11ea-33a4-a783b4d216dc
# ╟─09c928f0-f631-11ea-3ef7-512a6bececcc
# ╠═17d0ed20-f631-11ea-2e28-3bb9ca9a445f
# ╠═1c47f652-f631-11ea-15f6-1b9b59700f36
# ╟─6ade4d00-05c2-11eb-3732-ff945f7ce127
# ╠═8efa3720-05be-11eb-2c3e-9519eb7d8e7a
# ╠═22995300-05c2-11eb-3399-574d1fb2ed94
# ╟─21530220-f631-11ea-3994-319c862d51f9
# ╠═3b282ae0-f631-11ea-309d-639bf4411bb3
# ╠═7473adb0-f631-11ea-1c87-0f76b18a9ab6
# ╟─b6244db0-f63a-11ea-3b48-89d427664f5e
# ╠═c0cf83e0-05a5-11eb-32b5-6fb00cbc311b
# ╠═824bdde0-05bd-11eb-0594-cddd54c49757
# ╠═fb3fa610-f63a-11ea-2663-17224dc8aade
# ╠═09c9e0b0-f63b-11ea-2d50-4154e3432fa0
# ╟─46b40e10-f63b-11ea-2375-1976bb637d51
# ╠═de88b710-05c5-11eb-1795-a119590ad1c2
# ╟─00dd9240-05c1-11eb-3d13-ff544dc94b5d
# ╟─49bb9090-05c4-11eb-1aa9-8b4488a05654
# ╠═066f2bd0-05c4-11eb-032f-ad141ecd8070
# ╠═937c8392-05c1-11eb-0de5-191f4a5c2d8c
# ╠═25c274e0-05c1-11eb-3c1d-a591fde9722b
# ╠═06bf27ee-05c1-11eb-06ed-af4265dee892
# ╟─801f8080-f631-11ea-0728-f15dddc3ef5d
# ╠═8f06f650-f631-11ea-1c52-697060322173
