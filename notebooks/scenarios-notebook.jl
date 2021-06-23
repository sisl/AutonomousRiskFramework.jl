### A Pluto.jl notebook ###
# v0.14.8

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

# ╔═╡ 576e7079-56d3-45f9-a4da-3bccde50455b
begin
	using Revise
	using AutomotiveSimulator
	using AutomotiveVisualization
	using AdversarialDriving
	using RiskSimulator
end

# ╔═╡ b91adfce-0482-4ba9-9923-36447d3d5be4
using IntelligentDriving

# ╔═╡ 1b064105-d928-4832-a7df-c89d07a4e048
using PlutoUI

# ╔═╡ fa36e749-a4fc-4d20-a7b2-ee8c31de696f
AutomotiveVisualization.colortheme["background"] = colorant"white";

# ╔═╡ a044340e-beaa-4953-8914-1e35ee4e3a81
AutomotiveVisualization.set_render_mode(:fancy);

# ╔═╡ 8ef9e8b2-e8d7-451d-90e2-f6350aeae46e
begin
    r = 5.0
    w = DEFAULT_LANE_WIDTH
    l = 25
end

# ╔═╡ 8791fb30-d376-495d-9fce-964d3fe6519a
md"""
# Cross intersection
"""

# ╔═╡ eb446feb-52fa-4555-9be2-c5da8ea68324
Revise.retry()

# ╔═╡ 72cb2bec-8b11-461a-836b-457dcfa95f69
roadwayX = x_intersection(r=r, w=w, l=25)

# ╔═╡ 6b79e36c-89e9-4cab-9edd-d090cdc23f4b
USE_TIDM = true

# ╔═╡ cc54e9f3-d3db-4a14-b7cf-3fbdd2de862f
XIDM_template = get_XIDM_template();

# ╔═╡ 0508608b-d3bc-4ae9-ac5b-62eec7d22018
models = Dict{Int, AutomotiveSimulator.DriverModel}(
    1 => USE_TIDM ? TIDM(XIDM_template) : IntelligentDriverModel(), 
    2 => USE_TIDM ? TIDM(XIDM_template) : IntelligentDriverModel(), 
)

# ╔═╡ c120e7b5-8e94-4a3b-a6ce-b85197229722
# begin
# 	models[1].idm = MPCDriver()
# 	set_desired_speed!(models[1].idm, 12.0)
# end

# ╔═╡ a7e2f3a9-7326-4fd5-8577-3fcf30f08f3f
begin
	timestep = 0.1
	nticks = 80
end

# ╔═╡ 78413728-8e4d-4f5f-a3bf-96105661b213
# Computes the time it takes to cover a given distance, assuming the current acceleration of the provided idm
function AdversarialDriving.time_to_cross_distance_const_acc(veh::Entity, mpc::MPCDriver, ds::Float64)
    v = vel(veh)
    d = v^2 + 2*mpc.a_lon*ds
    d < 0 && return 0 # We have already passed the point we are trying to get to
    if isnothing(mpc.v_ref)
		vf = sqrt(d)
	else
		vf = min(mpc.v_ref, sqrt(d))
	end
    2*ds/(vf + v)
end


# ╔═╡ becc1d57-878b-4c0a-93a1-cec012a7a7ae
begin
	veh0X = cross_left_to_right(id=1, roadway=roadwayX)()
	veh1X = cross_bottom_to_top(id=2, roadway=roadwayX)()
    sceneX = Scene([veh0X, veh1X])
    scenesX = AutomotiveSimulator.simulate(sceneX, roadwayX, models, nticks, timestep)
end;

# ╔═╡ 38fe26a3-5eef-41ce-be19-3dd6ef9b5ea4
dt = 1

# ╔═╡ dbb4a522-3462-466d-b45b-0488e528ac77
@bind sim_t Slider(1:dt:length(scenesX))

# ╔═╡ 097f734a-15df-42d0-9619-1432e5b78554
render([roadwayX, scenesX[sim_t], (VelocityArrow(entity=x) for x in scenesX[sim_t])...])

# ╔═╡ 7fb572f5-b825-405d-b48a-461ff2d29b3a
function distance(scene)
    return get_distance(
        Entity(scene[1].state.veh_state, VehicleDef(), 1),
        Entity(scene[2].state.veh_state, VehicleDef(), 2))
end

# ╔═╡ 0997e090-810e-4472-b7e7-1defbe2a3ec4
if USE_TIDM
    distance(scenesX[sim_t])
else
    get_distance(scenesX[sim_t][1], scenesX[sim_t][2])
end

# ╔═╡ e3bfdd81-8973-41f2-aabe-d3441e64f5b7
any_collides(scenesX[sim_t])

# ╔═╡ 68fd7750-afbb-42a2-baa6-845f7bcd9f80
md"""
# T-intersection (head-on turn)
"""

# ╔═╡ 6627e48b-e5cf-4e1e-8ef3-13ac587e8be2
roadwayT = t_intersection()

# ╔═╡ 399a553b-0216-4c53-878d-7b605a6e8e53
begin
	veh0T = t_left_to_right(id=1, roadway=roadwayT)()
	veh1T = t_right_to_turn(s=20.0, id=2, roadway=roadwayT)()
    sceneT = Scene([veh0T, veh1T])
    scenesT = AutomotiveSimulator.simulate(sceneT, roadwayT, models, nticks, timestep)
end;

# ╔═╡ f55dcdd5-3a6e-46ab-a3d4-22696b759fc7
@bind sim_tT Slider(1:dt:length(scenesT))

# ╔═╡ e3d4acbd-373c-4998-ae04-34dd04a8179a
render([roadwayT, scenesT[sim_tT], (VelocityArrow(entity=x) for x in scenesT[sim_tT])...])

# ╔═╡ d7678894-f592-401a-89ef-986dafe3ca8e
md"""
# T-intersection (left turn)
"""

# ╔═╡ 335ad618-e749-4b8e-898a-c1cc597c97d6
begin
	veh0TT = t_left_to_right(id=1, roadway=roadwayT)()
	veh1TT = t_bottom_to_turn_left(id=2, roadway=roadwayT)()
    sceneTT = Scene([veh0TT, veh1TT])
    scenesTT = AutomotiveSimulator.simulate(sceneTT, roadwayT, models, nticks, timestep)
end;

# ╔═╡ 7e216135-4ad9-4271-b236-c53f6f82c5ea
@bind sim_tTT Slider(1:dt:length(scenesTT))

# ╔═╡ 4e5e67a7-a615-4598-8c5b-28201b178d92
render([roadwayT, scenesTT[sim_tTT], (VelocityArrow(entity=x) for x in scenesTT[sim_tTT])...])

# ╔═╡ e5cac1ca-306b-45fb-ad4f-6bea83044acc
md"""
# Stopped on highway
"""

# ╔═╡ 554224d1-60aa-44a1-b497-b6896a9a4bf0
roadwayHW = multi_lane_roadway()

# ╔═╡ eb572535-28d7-4726-bc8f-4fd1549cafc0
begin
	# TODO.
    modelsHW = deepcopy(models)
	modelsHW[1].goals[1] = [1]
	modelsHW[2].goals[1] = [1]
	modelsHW[2].idm.v_des = 0.0
	
	veh0HW = hw_behind(id=1, roadway=roadwayHW)()
	veh1HW = hw_stopping(id=2, roadway=roadwayHW)()
    sceneHW = Scene([veh0HW, veh1HW])
    scenesHW = AutomotiveSimulator.simulate(sceneHW, roadwayHW, modelsHW, nticks, timestep)
end;

# ╔═╡ bbb12a96-1b03-450f-a5d2-ae518067e570
@bind sim_tHW Slider(1:dt:length(scenesHW))

# ╔═╡ 7ea726be-f103-43fb-9185-7835bf67e2e8
md"`Render circle.`"

# ╔═╡ 6e3eb2e4-5100-4826-a5ca-6ea59478c491
struct RenderableCircle
    pos::VecE2
    radius::Float64
    color::Colorant
end

# ╔═╡ 0fd123ae-6dd4-4a0d-bd5d-ec2eb4af9757
obstacle = RenderableCircle(VecE2(5.0r,0), 1/2, RGB(0.8,0.2,0.2));

# ╔═╡ d7f68a57-b800-4685-9224-ede9b4e4740c
function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, circle::RenderableCircle)
    # add the desired render instructions to the rendermodel
    add_instruction!(
        rendermodel, AutomotiveVisualization.render_circle,
        (circle.pos.x, circle.pos.y, circle.radius, circle.color),
        coordinate_system=:scene
    )
    return rendermodel
end

# ╔═╡ b04e403d-590b-4448-8383-c1fe81332f31
render([roadwayHW, scenesHW[sim_tHW], obstacle, (VelocityArrow(entity=x) for x in scenesHW[sim_tHW])...])

# ╔═╡ 462f0e84-cfe1-4ca5-b8fc-5ce9999303d0
md"""
# Highway merge
"""

# ╔═╡ 2ed86ca9-1286-4266-80e3-08025bbadfa3
roadwayM = merging_roadway()

# ╔═╡ bebe9e7f-a39c-4763-8e58-b271c3d284fa
md"""
TODO: `propagate` fork.
"""

# ╔═╡ 5fced5f7-8aef-4ecb-92a5-af977b8f1058
function AutomotiveSimulator.propagate(veh::Entity{BlinkerState, D, I}, action::BlinkerVehicleControl, roadway::Roadway, Δt::Float64) where {D,I}
    # set the new goal
    vs = veh.state.veh_state
    if action.toggle_goal
        gs = goals(veh)
        curr_index = findfirst(gs .== laneid(veh))
        @assert !isnothing(curr_index)
        new_goal = gs[curr_index % length(gs) + 1]
        if can_have_goal(veh, new_goal, roadway)
            vs = set_lane(vs, new_goal, roadway)
        end
    end

    starting_lane = laneid(vs)

    # Update the kinematics of the vehicle (don't allow v < 0)
    vs_entity = Entity(vs, veh.def, veh.id)
    vs = propagate(vs_entity, LaneFollowingAccel(action.a + action.da), roadway, Δt)

    # Set the blinker state and return
    new_blink = action.toggle_blinker ? !blinker(veh) : blinker(veh)
    bs = BlinkerState(vs, new_blink, goals(veh), action.noise)
    # @assert starting_lane == laneid(bs) # TODO!
    bs
end

# ╔═╡ 2dbb611e-d8e2-4bbc-a47e-4618ff141203
begin
	# TODO.
    modelsM = deepcopy(models)
    map(k->delete!(modelsM[1].goals, k), 4:7)
    map(k->delete!(modelsM[2].goals, k), 4:7)
    modelsM[1].goals[3] = [3]
    modelsM[2].goals[3] = [3]

	veh0M = hw_straight(id=1, roadway=roadwayM)()
	veh1M = hw_merging(id=2, roadway=roadwayM)()
	sceneM = Scene([veh0M, veh1M])
    scenesM = AutomotiveSimulator.simulate(sceneM, roadwayM, modelsM, nticks, timestep)
end;

# ╔═╡ 68e8360b-73c8-4d48-872c-c0d2e1491ba9
@bind sim_tM Slider(1:dt:length(scenesM))

# ╔═╡ 21c5cc0d-bdca-4c9c-b03d-bd3329a0606a
render([roadwayM, scenesM[sim_tM], (VelocityArrow(entity=x) for x in scenesM[sim_tM])...])

# ╔═╡ 4210934c-b8bf-4a6f-9ec1-aaf56dbf9c4b
md"""
# Crosswalk
"""

# ╔═╡ 45dac8b0-7224-4e4a-9507-98119718e2ab
roadwayCW, crosswalk = crosswalk_roadway()

# ╔═╡ ceaffb0c-75bd-44cf-b629-dd43ab5efe23
begin
	# TODO.
	modelsCW = deepcopy(models)
	# NOTE: ped_TIDM_template or get_XIDM_template()
    modelsCW[1] = TIDM(get_XIDM_template()) # Important
    modelsCW[2] = AdversarialPedestrian() # Important

	veh0CW = cw_left_to_right(id=1, roadway=roadwayCW)()
	ped1CW = cw_pedestrian_walking_up(id=2, roadway=roadwayCW)()
	
    sceneCW = Scene([veh0CW, ped1CW])
    scenesCW = AutomotiveSimulator.simulate(sceneCW, roadwayCW, modelsCW, nticks, timestep)
end;

# ╔═╡ 706a5bb6-180f-40f7-8985-1bbe1c463b75
@bind sim_tCW Slider(1:dt:length(scenesCW))

# ╔═╡ a06a35d3-6611-4740-b43c-2c21084c4cc8
render([roadwayCW, crosswalk, scenesCW[sim_tCW], (VelocityArrow(entity=x) for x in scenesCW[sim_tCW])...])

# ╔═╡ f33ef352-c881-4788-9957-db843d07d3f1
PlutoUI.TableOfContents()

# ╔═╡ Cell order:
# ╠═576e7079-56d3-45f9-a4da-3bccde50455b
# ╠═b91adfce-0482-4ba9-9923-36447d3d5be4
# ╠═fa36e749-a4fc-4d20-a7b2-ee8c31de696f
# ╠═a044340e-beaa-4953-8914-1e35ee4e3a81
# ╠═8ef9e8b2-e8d7-451d-90e2-f6350aeae46e
# ╟─8791fb30-d376-495d-9fce-964d3fe6519a
# ╠═eb446feb-52fa-4555-9be2-c5da8ea68324
# ╠═72cb2bec-8b11-461a-836b-457dcfa95f69
# ╠═6b79e36c-89e9-4cab-9edd-d090cdc23f4b
# ╠═cc54e9f3-d3db-4a14-b7cf-3fbdd2de862f
# ╠═0508608b-d3bc-4ae9-ac5b-62eec7d22018
# ╠═c120e7b5-8e94-4a3b-a6ce-b85197229722
# ╠═a7e2f3a9-7326-4fd5-8577-3fcf30f08f3f
# ╠═78413728-8e4d-4f5f-a3bf-96105661b213
# ╠═becc1d57-878b-4c0a-93a1-cec012a7a7ae
# ╠═1b064105-d928-4832-a7df-c89d07a4e048
# ╠═38fe26a3-5eef-41ce-be19-3dd6ef9b5ea4
# ╠═dbb4a522-3462-466d-b45b-0488e528ac77
# ╠═097f734a-15df-42d0-9619-1432e5b78554
# ╠═7fb572f5-b825-405d-b48a-461ff2d29b3a
# ╠═0997e090-810e-4472-b7e7-1defbe2a3ec4
# ╠═e3bfdd81-8973-41f2-aabe-d3441e64f5b7
# ╟─68fd7750-afbb-42a2-baa6-845f7bcd9f80
# ╠═6627e48b-e5cf-4e1e-8ef3-13ac587e8be2
# ╠═399a553b-0216-4c53-878d-7b605a6e8e53
# ╠═f55dcdd5-3a6e-46ab-a3d4-22696b759fc7
# ╠═e3d4acbd-373c-4998-ae04-34dd04a8179a
# ╟─d7678894-f592-401a-89ef-986dafe3ca8e
# ╠═335ad618-e749-4b8e-898a-c1cc597c97d6
# ╠═7e216135-4ad9-4271-b236-c53f6f82c5ea
# ╠═4e5e67a7-a615-4598-8c5b-28201b178d92
# ╟─e5cac1ca-306b-45fb-ad4f-6bea83044acc
# ╠═554224d1-60aa-44a1-b497-b6896a9a4bf0
# ╠═eb572535-28d7-4726-bc8f-4fd1549cafc0
# ╠═bbb12a96-1b03-450f-a5d2-ae518067e570
# ╟─7ea726be-f103-43fb-9185-7835bf67e2e8
# ╠═6e3eb2e4-5100-4826-a5ca-6ea59478c491
# ╠═0fd123ae-6dd4-4a0d-bd5d-ec2eb4af9757
# ╠═d7f68a57-b800-4685-9224-ede9b4e4740c
# ╠═b04e403d-590b-4448-8383-c1fe81332f31
# ╟─462f0e84-cfe1-4ca5-b8fc-5ce9999303d0
# ╠═2ed86ca9-1286-4266-80e3-08025bbadfa3
# ╟─bebe9e7f-a39c-4763-8e58-b271c3d284fa
# ╠═5fced5f7-8aef-4ecb-92a5-af977b8f1058
# ╠═2dbb611e-d8e2-4bbc-a47e-4618ff141203
# ╠═68e8360b-73c8-4d48-872c-c0d2e1491ba9
# ╠═21c5cc0d-bdca-4c9c-b03d-bd3329a0606a
# ╟─4210934c-b8bf-4a6f-9ec1-aaf56dbf9c4b
# ╠═45dac8b0-7224-4e4a-9507-98119718e2ab
# ╠═ceaffb0c-75bd-44cf-b629-dd43ab5efe23
# ╠═706a5bb6-180f-40f7-8985-1bbe1c463b75
# ╠═a06a35d3-6611-4740-b43c-2c21084c4cc8
# ╠═f33ef352-c881-4788-9957-db843d07d3f1
