### A Pluto.jl notebook ###
# v0.12.4

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

# ╔═╡ b53f20fa-0e6d-11eb-3cd7-272757e1ac50
using Distributions, Parameters, Random, Latexify, PlutoUI

# ╔═╡ ae6d06a0-0e79-11eb-2a02-8137d81d74aa
using AutomotiveSimulator, AutomotiveVisualization

# ╔═╡ b8e93e70-0e79-11eb-340c-15748cb271ad
using AdversarialDriving

# ╔═╡ c9b0f3f2-0e79-11eb-3c7f-cb38421a6b48
using POMDPs, POMDPPolicies, POMDPSimulators

# ╔═╡ cf485b4c-0e79-11eb-213e-9d2d5fb826a4
Base.rand(rng::AbstractRNG, s::Scene) = s

# ╔═╡ d6f02cba-0e79-11eb-3dc4-517cdccba76c
sut_agent = BlinkerVehicleAgent(get_ped_vehicle(id=1, s=5.0, v=15.0),
	                            TIDM(ped_TIDM_template, noisy_observations=true));

# ╔═╡ 2793c72a-12f9-11eb-00b1-a5a26e2d7066
get_pedestrian_noisy(;id::Int64, s::Float64, v::Float64, noise::Noise) = (rng::AbstractRNG = Random.GLOBAL_RNG) -> NoisyPedestrian(roadway = AdversarialDriving.ped_roadway, lane = 2, s=s, v=v, id=id, noise=noise)

# ╔═╡ f12c039c-0e79-11eb-3e5c-55484b035f0f
adv_ped = NoisyPedestrianAgent(get_pedestrian_noisy(id=2, s=7.0, v=2.0, noise=Noise((-2, 0), 0)), AdversarialPedestrian());

# ╔═╡ ff489d50-0e79-11eb-3e55-1deee5a04fba
ad_mdp = AdversarialDrivingMDP(sut_agent, [adv_ped], ped_roadway, 0.1);

# ╔═╡ 1aee52d4-0e7a-11eb-1bc7-addbd951e23a
AutomotiveVisualization.render([ad_mdp.roadway, crosswalk, initialstate(ad_mdp)])

# ╔═╡ 96027338-0e7a-11eb-0e6a-2baa0a1e630d
ped_state, veh_state = initialstate(ad_mdp)

# ╔═╡ 60c4a1c6-12fa-11eb-1c6c-75ac0acb8034


# ╔═╡ 60c3aa6e-12fa-11eb-0f9b-370d4a017078


# ╔═╡ e55eea10-0e7a-11eb-1073-d71128bfaec4
md"""
## Working Problem: Pedestrian in a Crosswalk
We define a simple problem for adaptive stress testing (AST) to find failures. This problem, not colliding with a pedestrian in a crosswalk, samples random noise disturbances applied to the pedestrian's position and velocity from standard normal distributions $\mathcal{N}(\mu,\sigma)$. A failure is defined as a collision. AST will either select the seed which deterministically controls the sampled value from the distribution (i.e. from the transition model) or will directly sample the provided environmental distributions. These action modes are determined by the seed-action or sample-action options (`ASTSeedAction` and `ASTSampleAction`, respectively). AST will guide the simulation to failure events using a measure of distance to failure, while simultaneously trying to find the set of actions that maximizes the log-likelihood of the samples.
"""

# ╔═╡ b027ddac-0e7a-11eb-175d-3f260321a8d6
@with_kw struct AutoRiskParams
	endtime::Real = 30 # Simulate end time
end;

# ╔═╡ fa0dbfec-0e7d-11eb-3de0-2d7fb92df880
@bind da_noise Slider(1:2000, default=1) # Pedestrian action control

# ╔═╡ 86a67188-0fcd-11eb-02b3-d1baacae0b47
@bind a_noise Slider(1:2000, default=1) # Pedestrian action control

# ╔═╡ 6a5d8fa2-0fd4-11eb-0aa3-cffea5d2722a
@bind n1 Slider(-10:10, default=0) # Pedestrian action control

# ╔═╡ 79bf4600-0fd4-11eb-2df5-0da874116ddd
@bind n2 Slider(-10:100, default=0) # Pedestrian action control

# ╔═╡ 80ae4c7c-0fd4-11eb-2618-1be6e60b4b4e
@bind n3 Slider(-10:10, default=0) # Pedestrian action control

# ╔═╡ f9c05642-0e7a-11eb-24a8-ddf3c687f43b
noisy_action = Disturbance[PedestrianControl(a=VecE2(a_noise, 0), da=VecE2(da_noise, 0), noise=Noise((n1, n2), n3))]

# ╔═╡ 841d11a6-12f2-11eb-074d-b1b88d357e2a


# ╔═╡ e4a1320e-1000-11eb-2d4a-a9e28fc5eaa7
# function noisy_policy(s)
# 	for (i, veh) in enumerate(s)
# 		m = m = model(mdp, veh.id)
# 	end
# end
# ped_state

# ╔═╡ 0430a75c-0e7c-11eb-0d70-87eaa4c576af
# Behavior with noise
hist_noise = POMDPSimulators.simulate(HistoryRecorder(), ad_mdp,
	                                  FunctionPolicy((s) -> noisy_action));

# ╔═╡ f74d632a-0fe0-11eb-0ea2-1fa8d8081c64
map(x -> (x.entities[1].state.veh_state.posG[2], AdversarialDriving.noisy_entity(x.entities[1], ad_mdp.roadway).state.veh_state.posG[2], noise(x.entities[1]).pos[1]) , POMDPSimulators.stepthrough(ad_mdp, FunctionPolicy((s) -> noisy_action), "s", max_steps=5))

# ╔═╡ 62f3ce2e-0e7c-11eb-06ea-61fdf15abd60
ad_scenes_noise = state_hist(hist_noise);

# ╔═╡ 1e953e34-0fd5-11eb-27ce-77708031b68e
@bind ad_t Slider(1:length(ad_scenes_noise), default=12)

# ╔═╡ 2eb81816-0fd5-11eb-1f68-05e8ae158473
AutomotiveVisualization.render([ad_mdp.roadway, crosswalk,
			                        ad_scenes_noise[ad_t]])

# ╔═╡ 46c1cc9c-0fd6-11eb-190e-95c0ca7b3ec1
# Instructions for rendering the noisy pedestrian
function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, ped::Entity{NoisyPedState, VehicleDef, Int64})
    reg_ped = Entity(ped.state.veh_state, ped.def, ped.id)
    add_renderable!(rendermodel, FancyPedestrian(ped=reg_ped))
	noisy_ped = Entity(noisy_entity(ped, ad_mdp.roadway).state.veh_state, ped.def, ped.id)
	ghost_color = weighted_color_mean(0.3, colorant"blue", colorant"white")
	add_renderable!(rendermodel, FancyPedestrian(ped=noisy_ped, color=ghost_color))
    return rendermodel
end

# ╔═╡ 8c7cd740-0fec-11eb-2614-41c4a9503f48
# The function that propagates the PedestrianControl action
function AdversarialDriving.propagate(ped::Entity{NoisyPedState, D, I}, action::PedestrianControl, roadway::Roadway, Δt::Float64) where {D, I}
    starting_lane = laneid(ped)
    vs_entity = Entity(ped.state.veh_state, ped.def, ped.id)
    a_lat_lon = reverse(action.a + action.da)
    vs = propagate(vs_entity, LatLonAccel(a_lat_lon...), roadway, Δt)
	upd_noise = Noise((ped.state.noise.pos[1] + action.noise.pos[1] + action.noise.vel*Δt, action.noise.pos[2]), action.noise.vel)
    nps = NoisyPedState(AdversarialDriving.set_lane(vs, laneid(ped), roadway), upd_noise)
    @assert starting_lane == laneid(nps)
    nps
end

# ╔═╡ f21fdcc0-0fef-11eb-3dd6-e54db7bb748c
propagate(ped_state, noisy_action[1], ad_mdp.roadway, 1.0).noise.pos[1]

# ╔═╡ db40b35a-12f6-11eb-327e-6164bdc6cecf
function AdversarialDriving.update_adversary!(adversary::Agent, action::Disturbance, s::Scene)
    index = findfirst(id(adversary), s)
    isnothing(index) && return nothing # If the adversary is not in the scene then don't update
    adversary.model.next_action = action # Set the adversaries next action
    # veh = s[index] # Get the actual entity
    # state_type = typeof(veh.state) # Find out the type of its state
    # s[index] =  Entity(state_type(veh.state, noise = action.noise), veh.def, veh.id) # replace the entity in the scene
end

# ╔═╡ ec1af75a-0fff-11eb-010e-1396205fd6e1
begin
	index = findfirst(id(adversaries(ad_mdp)[1]), initialstate(ad_mdp))
	adversaries(ad_mdp)[1].model.next_action
end

# ╔═╡ 387b519e-0e7a-11eb-0d62-81968dbe7c06
md"""
#### Utility functions
"""

# ╔═╡ 501ce8ee-0e7a-11eb-0bfa-a94ed9e54903
function distance(ent1::Entity, ent2::Entity)
	pos1 = posg(ent1)
	pos2 = posg(ent2)
	hypot(pos1.x - pos2.x, pos1.y - pos2.y)
end

# ╔═╡ 7b08a89a-0e7a-11eb-207f-79c0c6df16f4
distance(veh_state, ped_state)

# ╔═╡ Cell order:
# ╠═b53f20fa-0e6d-11eb-3cd7-272757e1ac50
# ╠═ae6d06a0-0e79-11eb-2a02-8137d81d74aa
# ╠═b8e93e70-0e79-11eb-340c-15748cb271ad
# ╠═c9b0f3f2-0e79-11eb-3c7f-cb38421a6b48
# ╠═cf485b4c-0e79-11eb-213e-9d2d5fb826a4
# ╠═d6f02cba-0e79-11eb-3dc4-517cdccba76c
# ╠═2793c72a-12f9-11eb-00b1-a5a26e2d7066
# ╠═f12c039c-0e79-11eb-3e5c-55484b035f0f
# ╠═ff489d50-0e79-11eb-3e55-1deee5a04fba
# ╠═1aee52d4-0e7a-11eb-1bc7-addbd951e23a
# ╠═96027338-0e7a-11eb-0e6a-2baa0a1e630d
# ╠═60c4a1c6-12fa-11eb-1c6c-75ac0acb8034
# ╠═60c3aa6e-12fa-11eb-0f9b-370d4a017078
# ╠═7b08a89a-0e7a-11eb-207f-79c0c6df16f4
# ╟─e55eea10-0e7a-11eb-1073-d71128bfaec4
# ╠═b027ddac-0e7a-11eb-175d-3f260321a8d6
# ╠═fa0dbfec-0e7d-11eb-3de0-2d7fb92df880
# ╠═86a67188-0fcd-11eb-02b3-d1baacae0b47
# ╠═6a5d8fa2-0fd4-11eb-0aa3-cffea5d2722a
# ╠═79bf4600-0fd4-11eb-2df5-0da874116ddd
# ╠═80ae4c7c-0fd4-11eb-2618-1be6e60b4b4e
# ╠═f9c05642-0e7a-11eb-24a8-ddf3c687f43b
# ╠═841d11a6-12f2-11eb-074d-b1b88d357e2a
# ╠═e4a1320e-1000-11eb-2d4a-a9e28fc5eaa7
# ╠═0430a75c-0e7c-11eb-0d70-87eaa4c576af
# ╠═f74d632a-0fe0-11eb-0ea2-1fa8d8081c64
# ╠═62f3ce2e-0e7c-11eb-06ea-61fdf15abd60
# ╠═1e953e34-0fd5-11eb-27ce-77708031b68e
# ╠═2eb81816-0fd5-11eb-1f68-05e8ae158473
# ╠═46c1cc9c-0fd6-11eb-190e-95c0ca7b3ec1
# ╠═8c7cd740-0fec-11eb-2614-41c4a9503f48
# ╠═f21fdcc0-0fef-11eb-3dd6-e54db7bb748c
# ╠═db40b35a-12f6-11eb-327e-6164bdc6cecf
# ╠═ec1af75a-0fff-11eb-010e-1396205fd6e1
# ╟─387b519e-0e7a-11eb-0d62-81968dbe7c06
# ╠═501ce8ee-0e7a-11eb-0bfa-a94ed9e54903
