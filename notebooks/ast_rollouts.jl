### A Pluto.jl notebook ###
# v0.12.16

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

# ╔═╡ 92ce9460-f62b-11ea-1a8c-179776b5a0b4
@add using Distributions, Parameters, Random, Latexify, PlutoUI

# ╔═╡ e7783510-047d-11eb-26b0-373b14638ff0
try
	using POMDPStressTesting
catch
	pkg"add https://github.com/JuliaPOMDP/RLInterface.jl"
	pkg"dev https://github.com/sisl/POMDPStressTesting.jl"
	using POMDPStressTesting
end

# ╔═╡ 687a90c0-0480-11eb-1ef4-03e93ffa400c
@add using AutomotiveSimulator, AutomotiveVisualization

# ╔═╡ cda24800-0488-11eb-1d7f-8d52b5f6b33e
try
	using AdversarialDriving
catch
	pkg"dev https://github.com/sisl/AdversarialDriving.jl"
	using AdversarialDriving
end

# ╔═╡ 3617eb60-0489-11eb-144a-232b222a0365
@add using POMDPs, POMDPPolicies, POMDPSimulators

# ╔═╡ ae68e2e2-3bde-11eb-2133-41c25803770a
using STLCG

# ╔═╡ e66d5b60-2614-11eb-0dba-9f6829ce2fe2
using Statistics

# ╔═╡ 9061cd00-2b87-11eb-05e2-eb9b27484486
using PyPlot

# ╔═╡ de0497b0-3be3-11eb-096c-99c1a584ca68
using CrossEntropyVariants

# ╔═╡ 99e92652-3be7-11eb-0fb2-316c55af79a7
@add using DeepQLearning

# ╔═╡ 7a71fe60-3be6-11eb-1fe7-7bd3ab22ffc9
using Flux

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
# Base.rand(rng::AbstractRNG, s::Scene) = s

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
# ad_scenes = state_hist(hist); # TODO: fix this bug???
ad_scenes = [s.s for s in hist];

# ╔═╡ fb4732b0-0489-11eb-3e24-3d6ed2221771
# ad_scenes_noise = state_hist(hist_noise);
ad_scenes_noise = [s.s for s in hist_noise];

# ╔═╡ 84985160-3be0-11eb-274f-9579e1337cc3
hist_noise

# ╔═╡ 61b258b0-048d-11eb-0e0c-9d3f8c23b0ed
md"##### Distance Metrics"

# ╔═╡ ab394dd0-3bde-11eb-266a-ade6c9ff5697
md"""
###### Distance using STL
"""

# ╔═╡ b67e4880-3bde-11eb-185e-313a83ee528e
pkg"dev ../STLCG.jl/."

# ╔═╡ b2869690-3be0-11eb-0199-8f72d717fe61
md"""
Notes:
- Use STL to define collision (i.e. is robustness zero or negative?)
- Use robustness as the distance metric.
"""

# ╔═╡ e1090570-3be0-11eb-285d-8303a3401d8a
collisionₛₜₗ = Always(subformula=GreaterThan(:d, 0), interval=nothing)

# ╔═╡ 2d5e59b0-048d-11eb-257d-076254d3488f
function distance(ent1::Entity, ent2::Entity)
	pos1 = posg(ent1)
	pos2 = posg(ent2)
	d = hypot(pos1.x - pos2.x, pos1.y - pos2.y)
	return first(ρt(collisionₛₜₗ, [d]')) # robustness (i.e. distance)
end

# ╔═╡ 0626f970-3be1-11eb-101d-09cadd79b879
# function collision_stl(pos1, pos2)
	ρt(collisionₛₜₗ, [10]')
# end

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
		return distance(pedestrian, vehicle)
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
		# return BlackBox.distance(sim) <= 0
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

# ╔═╡ 8f4abd70-2491-11eb-1044-0f3fdced32b9
md"""
## Rollouts
- **$$Q$$-rollout**: explore based on existing $$Q$$-values
- **$$\epsilon$$-greedy rollout**: take random action with probability $$\epsilon$$, best action otherwise
- **CEM-rollout**: use low-level CEM optimization approach to select rollout action
- Gaussian process-based $$Q$$-function approximation
- Neural network-based $$Q$$-function approximation
    -  $$Q(d,a)$$ encoding instead of $$Q(s,a)$$
"""

# ╔═╡ a0660a70-2616-11eb-384b-f7998bf64235
# html"<style>ul li p {margin: 0} ol li p {margin: 0}</style>"# bulleted list spacing

# ╔═╡ ce9b7d70-2b8a-11eb-08d1-93a7132feafe
global final_is_distrs = Any[nothing]

# ╔═╡ f57a2ce0-2b8d-11eb-0abb-b71e527b3dad
final_is_distrs[1]

# ╔═╡ f943e670-2b8a-11eb-0419-8f1987e9b052
# convert(Vector{GrayBox.Environment}, final_is_distrs, 29)

# ╔═╡ 33dd9eb2-2b8c-11eb-3968-bf149aa4c850
# is_dist_0 = convert(Dict{Symbol, Vector{Sampleable}}, GrayBox.environment(ast_mdp.sim), 10)

# ╔═╡ 515394e0-2b8c-11eb-0365-7384df7c294c
# samples = rand(Random.GLOBAL_RNG, is_dist_0, 10)

# ╔═╡ 6668c490-2b8c-11eb-0e93-bf92bc74d37e
# losses_fn = (d, samples) -> [POMDPStressTesting.cem_losses(d, samples; mdp=ast_mdp, initstate=initialstate(ast_mdp))]

# ╔═╡ a07ec352-2b8c-11eb-2196-3b7ecb053b74
# losses = losses_fn(is_dist_0, samples)

# ╔═╡ 923e33e0-2491-11eb-1b9c-27f4842ad081
function cem_rollout(mdp::ASTMDP, s::ASTState, d::Int64)
	USE_PRIOR = false
	cem_mdp = mdp # deepcopy(mdp)
	prev_top_k = cem_mdp.params.top_k
	q_value = 0

	if USE_PRIOR # already computed importance sampling distribution
		is_distrs = final_is_distrs[1] # TODO: put this in `mdp`
	else
		cem_solver = CEMSolver(n_iterations=10,
							   num_samples=20,
							   episode_length=d,
							   show_progress=false)
		cem_mdp.params.top_k = 0
		cem_planner = solve(cem_solver, cem_mdp)
		is_distrs = convert(Vector{GrayBox.Environment}, search!(cem_planner, s), d)
		global final_is_distrs[1] = is_distrs
	end

	USE_MEAN = true # use the mean of the importance sampling distr, instead of rand.
	
	AST.go_to_state(mdp, s) # Records trace through this call

	for i in 1:length(is_distrs) # TODO: handle min(d, length) to select is_dist associated with `d`
		is_distr = is_distrs[1]
		if USE_MEAN
			sample = mean(is_distr)
		else
			sample = rand(is_distr)
		end
		# @info sample
		# @info is_distr
		a::ASTAction = ASTSampleAction(sample)
		# a::ASTAction = ASTSampleAction(rand(GrayBox.environment(mdp.sim)))
		# AST.random_action(mdp)
		(s, r) = @gen(:sp, :r)(cem_mdp, s, a, Random.GLOBAL_RNG)
		q_value = r + discount(cem_mdp)*q_value
		# AST.go_to_state(mdp, s) # Records trace through this call
	end
	# AST.go_to_state(mdp, s) # Records trace through this call
	cem_mdp.params.top_k = prev_top_k

	return q_value
end


# ╔═╡ 91d48ec0-2614-11eb-30a6-33c89c9c07ef
D = Dict{Symbol,Distributions.Sampleable}(:vel => Distributions.Normal{Float64}(0.16191185557003204, 0.00010103246108517094),:xpos => Distributions.Normal{Float64}(-7.717689089890023, 5.7750315962668e-5),:ypos => Distributions.Normal{Float64}(0.8894044320100376, 3.3435841468310024e-6))

# ╔═╡ bdef4c70-2614-11eb-1e70-51a2f4844295
function Statistics.mean(d::Dict)
	meand = Dict()
	for k in keys(d)
		m = mean(d[k])
		meand[k] = GrayBox.Sample(m, logpdf(d[k], m))
	end
	return meand
end

# ╔═╡ b8875e20-2615-11eb-0f24-d700ce3fa5ab
logpdf(D[:vel], 0.16191185557003204)

# ╔═╡ e2e4f7f0-2614-11eb-0221-2166dd21d555
rand(D)

# ╔═╡ 1485bdce-2615-11eb-2551-0bcf8c4477fa
mean(D)

# ╔═╡ dc2c8920-2536-11eb-1625-ab5ee68e2cce
md"""
**Ideas:**
- change CEM initial distribution (heavier)
- using the GP $$Q$$-values in the SELECTION process
"""

# ╔═╡ 91ad8ed0-24a0-11eb-2518-450a0f95159f
@with_kw mutable struct BestAction
	a = nothing
	r = -Inf
end

# ╔═╡ 01b0f140-24a1-11eb-2b51-c17654f8f698
global BEST_ACTION = BestAction()

# ╔═╡ 61c885de-24a4-11eb-232e-5df113729f2d
BEST_ACTION

# ╔═╡ dc2340f2-249f-11eb-0fab-b9545ba763f2
function record_best_action(mdp, a, r)
	global BEST_ACTION
	if r > BEST_ACTION.r # less than for distance `d`, greater than for reward `r`
		BEST_ACTION.a = a
		BEST_ACTION.r = r
	end
end

# ╔═╡ 92e3e160-249f-11eb-0d10-c3c67a74428e
function ϵ_rollout(mdp::ASTMDP, s::ASTState, d::Int64; ϵ=0.5)
    if d == 0 || isterminal(mdp, s)
        AST.go_to_state(mdp, s) # Records trace through this call
        return 0.0
    else
		if rand() < ϵ
			a::ASTAction = AST.random_action(mdp)
		else
			a = ASTSampleAction(BEST_ACTION.a)
		end

        (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + AST.discount(mdp)*ϵ_rollout(mdp, sp, d-1; ϵ=ϵ)

        return q_value
    end
end


# ╔═╡ fad2ab80-2b84-11eb-017a-6905ab6071b7
global 𝒟 = Tuple{Tuple{Real,ASTAction}, Real}[]

# ╔═╡ 3ffe0970-2b85-11eb-3fc2-cfd5d7ae02d7
𝒟

# ╔═╡ 9357b470-2b87-11eb-2477-cdc2d6db8846
begin
	x = [d for ((d,a), q) in 𝒟]
	y = [q for ((d,a), q) in 𝒟]
end

# ╔═╡ bd6e16a0-2b87-11eb-0cba-b16b63f314ce
begin
	PyPlot.svg(false)
	clf()
	hist2D(x, y)
	xlabel(L"d")
	ylabel(L"Q")
	gcf()
end

# ╔═╡ bdcba74e-2b84-11eb-07ac-c16716b887e9
function prior_rollout(mdp::ASTMDP, s::ASTState, d::Int64)
    if d == 0 || isterminal(mdp, s)
        AST.go_to_state(mdp, s) # Records trace through this call
        return 0.0
    else
		a::ASTAction = AST.random_action(mdp)
		distance = BlackBox.distance(mdp.sim)

        (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + AST.discount(mdp)*prior_rollout(mdp, sp, d-1)

		push!(𝒟, ((distance, a), q_value))

        return q_value
    end
end


# ╔═╡ 6784331e-249b-11eb-1c7c-85f91c2a0964
function AST.search!(planner::CEMPlanner, s::ASTState)
    mdp::ASTMDP = planner.mdp
    return action(planner, s)
end


# ╔═╡ 6dab3da0-2498-11eb-1446-2fbf5c3fbb17
function Base.convert(::Type{Vector{GrayBox.Environment}}, distr::Dict{Symbol, Vector{Sampleable}}, max_steps::Integer=1)
    env_vector = GrayBox.Environment[]
	for t in 1:max_steps
		env = GrayBox.Environment()
		for k in keys(distr)
			env[k] = distr[k][t]
		end
		push!(env_vector, env)
	end
	return env_vector::Vector{GrayBox.Environment}
end

# ╔═╡ c1b76a12-3be3-11eb-24f9-87990bc4141b
md"""
## Cross-Entropy Surrogate Method
"""

# ╔═╡ c59df310-3be3-11eb-0e26-fb3a6fbb0c07
pkg"add https://github.com/mossr/CrossEntropyVariants.jl"

# ╔═╡ cff128d0-3be5-11eb-01a3-65e012391e48
md"""
## Neural Network Q-Approximator
- Use distance $d$ as a _state proxy_
- Approximate $Q(d,a)$ using a neural network (DQN?)
- Collect data: $\mathcal{D} = (d, a) \to Q$
- Train network: input $d$ output action $a$ (DQN) or input $(d,a)$ output $Q$
"""

# ╔═╡ 9579dac0-3be6-11eb-228d-c7a452e9914d
@with_kw mutable struct Args
	α::Float64 = 3e-4      # learning rate
	epochs::Int = 20       # number of epochs
	device::Function = cpu # gpu or cpu device
	throttle::Int = 1      # throttle print every X seconds
end

# ╔═╡ 7beae2c0-3be6-11eb-0e3d-5ba4d0c3c354
model = Chain(Dense(1+6, 32), Dense(32, 1)) # d + |A| -> Q

# ╔═╡ dc769d50-3be6-11eb-3478-453ba24f4e7d
𝒟[1][1][2].sample

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

# ╔═╡ 0d159de0-284f-11eb-230b-a1feaa0b0581
visualize(planner)

# ╔═╡ b6244db0-f63a-11ea-3b48-89d427664f5e
md"""
### Other Solvers: Cross-Entropy Method
We can easily take our `ASTMDP` object (`planner.mdp`) and re-solve the MDP using a different solver—in this case the `CEMSolver`.
"""

# ╔═╡ c0cf83e0-05a5-11eb-32b5-6fb00cbc311b
ast_mdp = deepcopy(planner.mdp); # re-used from MCTS run.

# ╔═╡ 17cd6400-3be8-11eb-30f5-8d31eadaa535
actions(ast_mdp) |> length

# ╔═╡ 5043e8c0-284e-11eb-0c6d-7f3940b0a940
begin
	# TODO: get this index from the `trace` itself
	# findmax(ast_mdp.metrics.reward[ast_mdp.metrics.event])
	# findmax(ast_mdp.metrics.reward[ast_mdp.metrics.event])

	failure_likelihood_mcts =
		round(exp(maximum(planner.mdp.metrics.logprob[ast_mdp.metrics.event])), digits=4)

	Markdown.parse(string("\$\$p_\\text{likely} = ", failure_likelihood_mcts, "\$\$"))
end

# ╔═╡ 824bdde0-05bd-11eb-0594-cddd54c49757
cem_solver = CEMSolver(n_iterations=100, episode_length=ast_mdp.sim.params.endtime)

# ╔═╡ fb3fa610-f63a-11ea-2663-17224dc8aade
cem_planner = solve(cem_solver, ast_mdp);

# ╔═╡ ac2ec420-24a2-11eb-3cd4-b3751126845c
md"Run CEM? $(@bind run_cem CheckBox())"

# ╔═╡ 09c9e0b0-f63b-11ea-2d50-4154e3432fa0
with_terminal() do
	if run_cem
		global cem_action_trace = search!(cem_planner)
	end
end

# ╔═╡ d4817b20-2493-11eb-2f0b-b18bd7f364e4
run_cem ? cem_action_trace : nothing

# ╔═╡ 46b40e10-f63b-11ea-2375-1976bb637d51
md"Notice the failure rate is higher when using `CEMSolver` than `MCTSPWSolver`."

# ╔═╡ de88b710-05c5-11eb-1795-a119590ad1c2
cem_failure_rate = print_metrics(cem_planner)

# ╔═╡ 6b6fe810-24a2-11eb-2de0-5de07707e7c4
episodic_figures(cem_planner.mdp); POMDPStressTesting.gcf()

# ╔═╡ 7412e7b0-24a2-11eb-0523-9bb85e449a80
distribution_figures(cem_planner.mdp); POMDPStressTesting.gcf()

# ╔═╡ 38a4f220-2b89-11eb-14c6-c18aee509c28
md"""
## PPO solver
"""

# ╔═╡ 3b4ae4d0-2b89-11eb-0176-b3b84ddc6ec3
ppo_solver = PPOSolver(num_episodes=100, episode_length=ast_mdp.sim.params.endtime)

# ╔═╡ 6e7d2020-2b89-11eb-2153-236afd953dcd
ast_mdp_ppo = deepcopy(planner.mdp); # re-used from MCTS run.

# ╔═╡ 69815690-2b89-11eb-3fbf-4b94773309da
ppo_planner = solve(ppo_solver, ast_mdp_ppo);

# ╔═╡ e7a24060-2b8f-11eb-17ce-9751327ccc5a
md"Run PPO? $(@bind run_ppo CheckBox())"

# ╔═╡ 8134b0c0-2b89-11eb-09f3-e50f52093132
with_terminal() do
	if run_ppo
		global ppo_action_trace = search!(ppo_planner)
	end
end

# ╔═╡ a4fab5e2-2b89-11eb-1d20-b31c4761a77e
ppo_failure_rate = print_metrics(ppo_planner)

# ╔═╡ 30b19e00-2b8a-11eb-1d25-91d098b53ac7
md"""
## Random baseline
"""

# ╔═╡ fbff1c90-2b8f-11eb-1da5-91bb366a9f7e
md"Run random baseline? $(@bind run_rand CheckBox())"

# ╔═╡ 371953a2-2b8a-11eb-3f00-9b04999863b7
if run_rand
	rand_solver = RandomSearchSolver(n_iterations=100,
		                             episode_length=ast_mdp.sim.params.endtime)
	ast_mdp_rand = deepcopy(planner.mdp) # re-used from MCTS run.
	# ast_mdp_rand.params.seed  = 0
	rand_planner = solve(rand_solver, ast_mdp_rand)
	rand_action_trace = search!(rand_planner)
	rand_failure_rate = print_metrics(rand_planner)
end

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
# @latexify 
function R(p,e,d,τ)
    if τ && e
        return 0
    elseif τ && !e
        return -d
    else
        return log(p)
    end
end

# ╔═╡ f6213a50-f62f-11ea-07c7-2dcc383c8042
function BlackBox.evaluate!(sim::AutoRiskSim, sample::GrayBox.EnvironmentSample)
    logprob::Real = GrayBox.transition!(sim, sample) # Step simulation
    d::Real       = BlackBox.distance(sim)           # Calculate miss distance
    event::Bool   = BlackBox.isevent(sim)            # Check event indication
    sim.prev_distance = d                            # Store previous distance
	r = R(exp(logprob), event, d, BlackBox.isterminal(sim))
	record_best_action(sim, sample, r)
	return (logprob::Real, d::Real, event::Bool)
end

# ╔═╡ c6a61f40-05b4-11eb-1f1d-6950aaea7a8d
begin
	envsample2 = rand(GrayBox.environment(simx))
	BlackBox.evaluate!(simx, envsample2) # (log-likelihood, distance, isevent)
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
# ╠═84985160-3be0-11eb-274f-9579e1337cc3
# ╟─61b258b0-048d-11eb-0e0c-9d3f8c23b0ed
# ╠═2d5e59b0-048d-11eb-257d-076254d3488f
# ╟─ab394dd0-3bde-11eb-266a-ade6c9ff5697
# ╠═b67e4880-3bde-11eb-185e-313a83ee528e
# ╠═ae68e2e2-3bde-11eb-2133-41c25803770a
# ╟─b2869690-3be0-11eb-0199-8f72d717fe61
# ╠═e1090570-3be0-11eb-285d-8303a3401d8a
# ╠═0626f970-3be1-11eb-101d-09cadd79b879
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
# ╟─8f4abd70-2491-11eb-1044-0f3fdced32b9
# ╠═a0660a70-2616-11eb-384b-f7998bf64235
# ╠═ce9b7d70-2b8a-11eb-08d1-93a7132feafe
# ╠═f57a2ce0-2b8d-11eb-0abb-b71e527b3dad
# ╠═f943e670-2b8a-11eb-0419-8f1987e9b052
# ╠═33dd9eb2-2b8c-11eb-3968-bf149aa4c850
# ╠═515394e0-2b8c-11eb-0365-7384df7c294c
# ╠═6668c490-2b8c-11eb-0e93-bf92bc74d37e
# ╠═a07ec352-2b8c-11eb-2196-3b7ecb053b74
# ╠═923e33e0-2491-11eb-1b9c-27f4842ad081
# ╠═91d48ec0-2614-11eb-30a6-33c89c9c07ef
# ╠═e66d5b60-2614-11eb-0dba-9f6829ce2fe2
# ╠═bdef4c70-2614-11eb-1e70-51a2f4844295
# ╠═b8875e20-2615-11eb-0f24-d700ce3fa5ab
# ╠═e2e4f7f0-2614-11eb-0221-2166dd21d555
# ╠═1485bdce-2615-11eb-2551-0bcf8c4477fa
# ╠═dc2c8920-2536-11eb-1625-ab5ee68e2cce
# ╠═91ad8ed0-24a0-11eb-2518-450a0f95159f
# ╠═01b0f140-24a1-11eb-2b51-c17654f8f698
# ╠═61c885de-24a4-11eb-232e-5df113729f2d
# ╠═dc2340f2-249f-11eb-0fab-b9545ba763f2
# ╠═92e3e160-249f-11eb-0d10-c3c67a74428e
# ╠═fad2ab80-2b84-11eb-017a-6905ab6071b7
# ╠═3ffe0970-2b85-11eb-3fc2-cfd5d7ae02d7
# ╠═9061cd00-2b87-11eb-05e2-eb9b27484486
# ╠═9357b470-2b87-11eb-2477-cdc2d6db8846
# ╠═bd6e16a0-2b87-11eb-0cba-b16b63f314ce
# ╠═bdcba74e-2b84-11eb-07ac-c16716b887e9
# ╠═6784331e-249b-11eb-1c7c-85f91c2a0964
# ╠═6dab3da0-2498-11eb-1446-2fbf5c3fbb17
# ╟─c1b76a12-3be3-11eb-24f9-87990bc4141b
# ╠═c59df310-3be3-11eb-0e26-fb3a6fbb0c07
# ╠═de0497b0-3be3-11eb-096c-99c1a584ca68
# ╟─cff128d0-3be5-11eb-01a3-65e012391e48
# ╠═99e92652-3be7-11eb-0fb2-316c55af79a7
# ╠═7a71fe60-3be6-11eb-1fe7-7bd3ab22ffc9
# ╠═9579dac0-3be6-11eb-228d-c7a452e9914d
# ╠═7beae2c0-3be6-11eb-0e3d-5ba4d0c3c354
# ╠═17cd6400-3be8-11eb-30f5-8d31eadaa535
# ╠═dc769d50-3be6-11eb-3478-453ba24f4e7d
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
# ╟─5043e8c0-284e-11eb-0c6d-7f3940b0a940
# ╠═0d159de0-284f-11eb-230b-a1feaa0b0581
# ╟─b6244db0-f63a-11ea-3b48-89d427664f5e
# ╠═c0cf83e0-05a5-11eb-32b5-6fb00cbc311b
# ╠═824bdde0-05bd-11eb-0594-cddd54c49757
# ╠═fb3fa610-f63a-11ea-2663-17224dc8aade
# ╟─ac2ec420-24a2-11eb-3cd4-b3751126845c
# ╠═09c9e0b0-f63b-11ea-2d50-4154e3432fa0
# ╠═d4817b20-2493-11eb-2f0b-b18bd7f364e4
# ╟─46b40e10-f63b-11ea-2375-1976bb637d51
# ╠═de88b710-05c5-11eb-1795-a119590ad1c2
# ╠═6b6fe810-24a2-11eb-2de0-5de07707e7c4
# ╠═7412e7b0-24a2-11eb-0523-9bb85e449a80
# ╟─38a4f220-2b89-11eb-14c6-c18aee509c28
# ╠═3b4ae4d0-2b89-11eb-0176-b3b84ddc6ec3
# ╠═6e7d2020-2b89-11eb-2153-236afd953dcd
# ╠═69815690-2b89-11eb-3fbf-4b94773309da
# ╟─e7a24060-2b8f-11eb-17ce-9751327ccc5a
# ╠═8134b0c0-2b89-11eb-09f3-e50f52093132
# ╠═a4fab5e2-2b89-11eb-1d20-b31c4761a77e
# ╟─30b19e00-2b8a-11eb-1d25-91d098b53ac7
# ╟─fbff1c90-2b8f-11eb-1da5-91bb366a9f7e
# ╠═371953a2-2b8a-11eb-3f00-9b04999863b7
# ╟─00dd9240-05c1-11eb-3d13-ff544dc94b5d
# ╟─49bb9090-05c4-11eb-1aa9-8b4488a05654
# ╠═066f2bd0-05c4-11eb-032f-ad141ecd8070
# ╠═937c8392-05c1-11eb-0de5-191f4a5c2d8c
# ╠═25c274e0-05c1-11eb-3c1d-a591fde9722b
# ╠═06bf27ee-05c1-11eb-06ed-af4265dee892
# ╟─801f8080-f631-11ea-0728-f15dddc3ef5d
# ╠═8f06f650-f631-11ea-1c52-697060322173
