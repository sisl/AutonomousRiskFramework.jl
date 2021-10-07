### A Pluto.jl notebook ###
# v0.14.7

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

# ╔═╡ 9ab2bb62-b97c-11eb-2bd2-01a3bbe4935c
using AutomotiveSimulator

# ╔═╡ c3c35502-6151-4fc2-9a0b-0fcd598a9586
using AutomotiveVisualization

# ╔═╡ 1615fc72-13ed-423b-a554-984e4077da92
using AdversarialDriving

# ╔═╡ 4fddbf48-49a4-4167-a35f-31bfd59da89a
using PlutoUI

# ╔═╡ 09af83f7-765b-42e9-97c6-7d31d25f0d3c
using AutoUrban

# ╔═╡ 273bd194-6b77-4df9-bd0f-485be0567754
begin
	include("..\\RiskSimulator.jl\\src\\scenarios\\scenarios.jl")
	r = 5.0
	w = DEFAULT_LANE_WIDTH
	l = 25
end

# ╔═╡ 9458472a-3311-474e-a68b-fa6f5fddd5db
begin
	include("..\\RiskSimulator.jl\\src\\scenarios\\scenarios.jl")
	roadwayTC = traffic_circle_roadway()
end

# ╔═╡ e738e8ec-b31c-4dc7-9f77-71a728743d86
AutomotiveVisualization.colortheme["background"] = colorant"white";

# ╔═╡ a1f73e35-287a-4d1f-9909-868c3e023903
AutomotiveVisualization.set_render_mode(:fancy);

# ╔═╡ f7dc489c-e159-4b8a-bb67-9fb23bb588de
md"""
# Cross intersection
"""

# ╔═╡ f885b321-8835-4a43-80b2-faf9f484dbfb
roadway = x_intersection(r=r, w=w, l=25)

# ╔═╡ b6e8449f-e875-46a7-8f46-fc3324ceaf4b
USE_TIDM = true

# ╔═╡ bbab7f17-2ade-4fe3-96b0-9ac32cd3da4b
XIDM_template = get_XIDM_template();

# ╔═╡ 09c11427-c21f-4f79-84db-f4fdb3ff3d82
models = Dict{Int, AutomotiveSimulator.DriverModel}(
	1 => USE_TIDM ? TIDM(XIDM_template) : IntelligentDriverModel(), 
	2 => USE_TIDM ? TIDM(XIDM_template) : IntelligentDriverModel(), 
)

# ╔═╡ 21de40d9-f8a8-431f-b1db-f9280c94d562
begin
	timestep = 0.1
	nticks = 80

	state0 = VehicleState(VecSE2(-4r,0.0,0.0), roadway[LaneTag(2,1)], roadway, 8.0)
	state1 = VehicleState(VecSE2(r+w, -4r, π/2), roadway[LaneTag(7,1)], roadway, 8.0)
	
	if USE_TIDM
		vs0 = BlinkerState(state0, false, [1], Noise())
		vs1 = BlinkerState(state1, false, [1], Noise())
	else
		vs0 = state0
		vs1 = state1
	end
	scene = Scene([Entity(vs0, VehicleDef(), 1), Entity(vs1, VehicleDef(), 2)])
	scenes = simulate(scene, roadway, models, nticks, timestep)
end;

# ╔═╡ 300c337e-1ca5-4882-90da-b2781f2a30ca
dt = 1

# ╔═╡ f90ac295-3dcf-434a-916e-ae04d1fc0252
@bind sim_t Slider(1:dt:length(scenes))

# ╔═╡ ed325984-9bc6-4c1b-a4b1-3a2507f766b3
render([roadway, scenes[sim_t]])

# ╔═╡ dc1ad6fa-8b76-4662-b0fc-744076fa0d95
function distance(scene)
	return get_distance(
		Entity(scene[1].state.veh_state, VehicleDef(), 1),
		Entity(scene[2].state.veh_state, VehicleDef(), 2))
end

# ╔═╡ b8948cf0-db38-4ce2-956b-8ae11e665fc4
if USE_TIDM
	distance(scenes[sim_t])
else
	get_distance(scenes[sim_t][1], scenes[sim_t][2])
end

# ╔═╡ 2b99c564-2907-4db9-9901-4be7326da039
any_collides(scenes[sim_t])

# ╔═╡ 5cf252da-54e3-4c89-bc6b-eaf54364b603
md"""
# T-intersection (head-on turn)
"""

# ╔═╡ c36c14a0-22fc-4017-969f-3992c866cb89
roadwayT = t_intersection(r=r, w=w, l=25)

# ╔═╡ 5384a3dc-a071-48e5-832d-0f6f18f9b329
begin
    state0_T = VehicleState(VecSE2(-4r,0.0,0.0), roadwayT[LaneTag(2,1)], roadwayT, 8.0)
    state1_T = VehicleState(VecSE2(4r, w, π), roadwayT[LaneTag(4,1)], roadwayT, 8.0)
    
    if USE_TIDM
        vs0_T = BlinkerState(state0_T, false, [1], Noise())
        vs1_T = BlinkerState(state1_T, false, [1], Noise())
    else
        vs0_T = state0_T
        vs1_T = state1_T
    end
    sceneT = Scene([Entity(vs0_T, VehicleDef(), 1), Entity(vs1_T, VehicleDef(), 2)])
    scenesT = simulate(sceneT, roadwayT, models, nticks, timestep)
end;

# ╔═╡ fb031b3c-0bdc-4d1d-9559-ce81d83c4773
@bind sim_tT Slider(1:dt:length(scenesT))

# ╔═╡ 5e608abd-3bb2-4774-87f6-c756c459a338
render([roadwayT, scenesT[sim_tT]])

# ╔═╡ 8aa2228c-b307-4b02-9405-02cfe3d2513a
md"""
# T-intersection (left turn)
"""

# ╔═╡ 6d4e221e-0daa-4c1e-9869-f3d128db9d47
begin
    state0_TT = VehicleState(VecSE2(-4r,0.0,0.0), roadwayT[LaneTag(2,1)], roadwayT, 8.0)
    state1_TT = VehicleState(VecSE2(r+w, -5r, π/2), roadwayT[LaneTag(6,1)], roadwayT, 8.0)

    if USE_TIDM
        vs0_TT = BlinkerState(state0_TT, false, [1], Noise())
        vs1_TT = BlinkerState(state1_TT, false, [1], Noise())
    else
        vs0_TT = state0_TT
        vs1_TT = state1_TT
    end
    sceneTT = Scene([Entity(vs0_TT, VehicleDef(), 1), Entity(vs1_TT, VehicleDef(), 2)])
    scenesTT = simulate(sceneTT, roadwayT, models, nticks, timestep)
end;

# ╔═╡ fc224d9f-a9fe-4ad6-8c69-65769a774f2a
@bind sim_tTT Slider(1:dt:length(scenesTT))

# ╔═╡ 78aa48cc-ccf7-435f-aa1c-019db846439f
render([roadwayT, scenesTT[sim_tTT]])

# ╔═╡ c0e52aca-048e-400c-9cf8-5b1d63e175ff
md"""
# Stopped on highway
"""

# ╔═╡ ac759406-d597-4503-86df-4bbb7135e1c5
roadwayHW = multi_lane_roadway()

# ╔═╡ 0a9b9b26-fe44-479a-bd4f-6714859d130e
begin
    state0_HW = VehicleState(VecSE2(r,0.0,0.0), roadwayHW[LaneTag(1,1)], roadwayHW, 8.0)
    state1_HW = VehicleState(VecSE2(4r, 0, 0), roadwayHW[LaneTag(1,1)], roadwayHW, 8.0)

    if false # USE_TIDM
        vs0_HW = BlinkerState(state0_HW, false, [1], Noise())
        vs1_HW = BlinkerState(state1_HW, false, [1], Noise())
    else
        vs0_HW = state0_HW
        vs1_HW = state1_HW
    end
	modelsHW = deepcopy(models)
	modelsHW[1] = IntelligentDriverModel()
	modelsHW[2] = IntelligentDriverModel(v_des=0)
	# modelsHW[2].idm.v_des=0
    sceneHW = Scene([Entity(vs0_HW, VehicleDef(), 1), Entity(vs1_HW, VehicleDef(), 2)])
    scenesHW = simulate(sceneHW, roadwayHW, modelsHW, nticks, timestep)
end;

# ╔═╡ 3279458c-e35a-4c90-aa8e-568170e7deef
@bind sim_tHW Slider(1:dt:length(scenesHW))

# ╔═╡ 84725684-09f6-4c81-b369-daa74bb2e1af
struct RenderableCircle
    pos::VecE2
    radius::Float64
    color::Colorant
end

# ╔═╡ 51077e41-40a0-488e-aac3-ab8e569a19a1
obstacle = RenderableCircle(VecE2(6.05r,0), 1/2, RGB(0.8,0.2,0.2));

# ╔═╡ e1eb954b-a92a-427f-8848-3a51eaecf4ca
function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, circle::RenderableCircle)
    # add the desired render instructions to the rendermodel
    add_instruction!(
        rendermodel, AutomotiveVisualization.render_circle,
        (circle.pos.x, circle.pos.y, circle.radius, circle.color),
        coordinate_system=:scene
    )
    return rendermodel
end

# ╔═╡ 8f49a05e-77b0-4750-bf4a-de227f514721
render([roadwayHW, scenesHW[sim_tHW], obstacle])

# ╔═╡ ff83dae1-32be-448d-a976-da6a8f0c474d
md"""
# Crosswalk
"""

# ╔═╡ b5d9dc08-9a74-4d22-856e-5a795d68294a
roadwayCW, crosswalk = crosswalk_roadway()

# ╔═╡ 640fee3f-2473-436a-9903-40dd30b322de
begin
    state0_CW = VehicleState(VecSE2(r,0.0,0.0), roadwayCW[LaneTag(1,1)], roadwayCW, 8.0)
    state1_CW = VehicleState(VecSE2(5r, -r, π/2), roadwayCW[LaneTag(2,1)], roadwayCW, 8.0)

    if USE_TIDM
        vs0_CW = BlinkerState(state0_CW, false, [1], Noise())
        vs1_CW = NoisyPedState(state1_CW, Noise())
    else
        vs0_CW = state0_CW
        vs1_CW = state1_CW
    end
    modelsCW = deepcopy(models)
	modelsCW[1] = TIDM(ped_TIDM_template) # Important
    modelsCW[2] = AdversarialPedestrian() # Important

    sceneCW = Scene([Entity(vs0_CW, VehicleDef(), 1), Entity(vs1_CW, VehicleDef(AgentClass.PEDESTRIAN, 1.0, 1.0), 2)])
    scenesCW = simulate(sceneCW, roadwayCW, modelsCW, nticks, timestep)
end;

# ╔═╡ bf3d94c0-853b-4128-bd0f-fefe727c7e4b
@bind sim_tCW Slider(1:dt:length(scenesCW))

# ╔═╡ 50505804-6d5e-4a42-83c6-90c189f333cc
render([roadwayCW, crosswalk, scenesCW[sim_tCW]])

# ╔═╡ e471dabf-98b9-4da7-8647-9e4444586390
md"""
# Highway merge
"""

# ╔═╡ 8a4db9c8-9da2-4432-a884-f05a0a4dda0a
roadwayM = merging_roadway()

# ╔═╡ 168035dd-2450-4edb-81b1-644c0f6e93f2
md"""
TODO: `propagate` fork.
"""

# ╔═╡ e3468c7f-ec57-441e-935c-fdf93df4d624
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

# ╔═╡ b3a5633e-69e7-47f8-a774-c6cc2d8cf70e
begin
    state0_M = VehicleState(VecSE2(r/2,0.0,0.0), roadwayM[LaneTag(1,1)], roadwayM, 8.0)
    state1_M = VehicleState(VecSE2(2r,-w,0), roadwayM[LaneTag(3,1)], roadwayM, 8.0)

    if USE_TIDM
        vs0_M = BlinkerState(state0_M, false, [1], Noise())
        vs1_M = BlinkerState(state1_M, false, [1], Noise())
    else
        vs0_M = state0_M
        vs1_M = state1_M
    end
	modelsM = deepcopy(models)
	map(k->delete!(modelsM[1].goals, k), 4:7)
	map(k->delete!(modelsM[2].goals, k), 4:7)
	modelsM[1].goals[3] = [3]
	modelsM[2].goals[3] = [3]
    sceneM = Scene([Entity(vs0_M, VehicleDef(), 1), Entity(vs1_M, VehicleDef(), 2)])
    scenesM = simulate(sceneM, roadwayM, modelsM, nticks, timestep)
end;

# ╔═╡ 01169ef1-d9b6-4ea5-b565-c8bfaed2bc0c
@bind sim_tM Slider(1:dt:length(scenesM))

# ╔═╡ 884c0c94-5be9-4de4-8d49-5b9b10bdccb8
render([roadwayM, scenesM[sim_tM], (VelocityArrow(entity=x) for x in scenesM[sim_tM])...])

# ╔═╡ 0135ceb2-43fc-4154-89ba-9175cd55a307
md"""
---
---
---
"""

# ╔═╡ 9b83bbc4-fefb-425e-94c1-6127ac15b39a
md"""
# Traffic circle merge (UNUSED)
"""

# ╔═╡ edcc81c0-6d9f-4d30-bb03-161d58f5fd4b
begin
    state0_TC = VehicleState(VecSE2(-6w,5w,-π/2), roadwayTC[LaneTag(1,1)], roadwayTC, 8.0)
    state1_TC = VehicleState(VecSE2(-4r,-w,0), roadwayTC[LaneTag(7,1)], roadwayTC, 8.0)

    if USE_TIDM
        vs0_TC = BlinkerState(state0_TC, false, [1], Noise())
        vs1_TC = BlinkerState(state1_TC, false, [1], Noise())
    else
        vs0_TC = state0_TC
        vs1_TC = state1_TC
    end
    sceneTC = Scene([Entity(vs0_TC, VehicleDef(), 1), Entity(vs1_TC, VehicleDef(), 2)])
    scenesTC = simulate(sceneTC, roadwayTC, models, nticks, timestep)
end;

# ╔═╡ 1a552db3-65fa-41eb-a9be-48453bbdd4d0
@bind sim_tTC Slider(1:dt:length(scenesTC))

# ╔═╡ 22ba800b-a6db-41b8-b31f-c926fa26fee6
render([roadwayTC, scenesTC[sim_tTC], (VelocityArrow(entity=x) for x in scenesTC[sim_tTC])...])

# ╔═╡ 5c1774b9-b5a3-40fc-bc28-9f6981937598
begin
	roadway_urban = Roadway()
	origin = VecSE2(-50.0,0.0,0.0)
	laneLength = 30.0
	nlanes = 4
	add_line!(origin,nlanes,laneLength,roadway_urban)
	origin = VecSE2(0.0,-20.0,-0.4*pi)
	laneLength = 30.0
	nlanes = 4
	add_line!(origin,nlanes,laneLength,roadway_urban)
	origin = VecSE2(15.0,8.0,0.2*pi)
	laneLength = 30.0
	nlanes = 4
	add_line!(origin,nlanes,laneLength,roadway_urban)
	origin = VecSE2(5.0,20.0,0.4*pi)
	laneLength = 30.0
	nlanes = 4
	add_line!(origin,nlanes,laneLength,roadway_urban)
	roadway_urban
end

# ╔═╡ 74312aba-6062-4d40-99e6-d04ea1204a13
md"""
# Junction (UNUSED)
"""

# ╔═╡ f5982a13-0d25-42b5-8053-f58668740064
render([roadway_urban])

# ╔═╡ 7c27804d-8c42-4d67-9df9-ce7874b24c68
# Specify connections
# A Junction contains several Connections
# Connection(1,3) means connect all lanes from segment 1 to segment 3
# Connection(1,2,0,[(1,1),(2,2)]) means connect segment 1 and 2 from lane 1 in segment 1 to lane 2 in segment 2, lane 2 in segment 1 to lane 2 in segment2
junctions = [
	Junction([
		Connection(1,2,0,[(1,1),(2,2)]),
		Connection(1,3),
		Connection(1,4)
	])
]

# ╔═╡ b79c7d56-d0dd-4910-ba3b-59b1b737219d
#Add all junctions
for junction in junctions
    add_junction!(junction,roadway_urban)
end

# ╔═╡ 04f2cc03-2b9c-4544-8d3c-2a9caccfec4c
render([roadway_urban])

# ╔═╡ 1ccb168e-0739-4482-a6a7-853ffd2ff83b
md"""
# Two lane roadway (UNUSED)
"""

# ╔═╡ cf674376-95a3-4e87-a605-533ea7fddc48
roadway2 = two_lane_roadway(w=w, l=25)

# ╔═╡ c57360f5-92c5-4fa6-be9d-adc119077159
begin
	state0_2lane = VehicleState(VecSE2(-4r,0.0,0.0), roadway2[LaneTag(1,1)], roadway2, 8.0)
	state1_2lane = VehicleState(VecSE2(6r, w, π), roadway2[LaneTag(2,1)], roadway2, 8.0)
	
	if USE_TIDM
		vs0_2lane = BlinkerState(state0_2lane, false, [1], Noise())
		vs1_2lane = BlinkerState(state1_2lane, false, [1], Noise())
	else
		vs0_2lane = state0_2lane
		vs1_2lane = state1_2lane
	end
	scene2 = Scene([Entity(vs0_2lane, VehicleDef(), 1), Entity(vs1_2lane, VehicleDef(), 2)])
	scenes2 = simulate(scene2, roadway2, models, nticks, timestep)
end;

# ╔═╡ db0296c6-1f39-4a71-bf41-b6cd9497d6ef
@bind sim_t2 Slider(1:dt:length(scenes2))

# ╔═╡ f59666c9-99c6-4de3-9dae-1cc31df0dee6
render([roadway2, scenes2[sim_t2]])

# ╔═╡ 369e4698-1657-4d97-bf6c-fc8f0e736401
PlutoUI.TableOfContents()

# ╔═╡ Cell order:
# ╠═9ab2bb62-b97c-11eb-2bd2-01a3bbe4935c
# ╠═c3c35502-6151-4fc2-9a0b-0fcd598a9586
# ╠═1615fc72-13ed-423b-a554-984e4077da92
# ╠═e738e8ec-b31c-4dc7-9f77-71a728743d86
# ╠═a1f73e35-287a-4d1f-9909-868c3e023903
# ╠═273bd194-6b77-4df9-bd0f-485be0567754
# ╟─f7dc489c-e159-4b8a-bb67-9fb23bb588de
# ╠═f885b321-8835-4a43-80b2-faf9f484dbfb
# ╠═b6e8449f-e875-46a7-8f46-fc3324ceaf4b
# ╠═bbab7f17-2ade-4fe3-96b0-9ac32cd3da4b
# ╠═09c11427-c21f-4f79-84db-f4fdb3ff3d82
# ╠═21de40d9-f8a8-431f-b1db-f9280c94d562
# ╠═4fddbf48-49a4-4167-a35f-31bfd59da89a
# ╠═300c337e-1ca5-4882-90da-b2781f2a30ca
# ╠═f90ac295-3dcf-434a-916e-ae04d1fc0252
# ╠═ed325984-9bc6-4c1b-a4b1-3a2507f766b3
# ╠═dc1ad6fa-8b76-4662-b0fc-744076fa0d95
# ╠═b8948cf0-db38-4ce2-956b-8ae11e665fc4
# ╠═2b99c564-2907-4db9-9901-4be7326da039
# ╟─5cf252da-54e3-4c89-bc6b-eaf54364b603
# ╠═c36c14a0-22fc-4017-969f-3992c866cb89
# ╟─5384a3dc-a071-48e5-832d-0f6f18f9b329
# ╠═fb031b3c-0bdc-4d1d-9559-ce81d83c4773
# ╠═5e608abd-3bb2-4774-87f6-c756c459a338
# ╟─8aa2228c-b307-4b02-9405-02cfe3d2513a
# ╟─6d4e221e-0daa-4c1e-9869-f3d128db9d47
# ╠═fc224d9f-a9fe-4ad6-8c69-65769a774f2a
# ╠═78aa48cc-ccf7-435f-aa1c-019db846439f
# ╟─c0e52aca-048e-400c-9cf8-5b1d63e175ff
# ╠═ac759406-d597-4503-86df-4bbb7135e1c5
# ╟─0a9b9b26-fe44-479a-bd4f-6714859d130e
# ╠═3279458c-e35a-4c90-aa8e-568170e7deef
# ╟─84725684-09f6-4c81-b369-daa74bb2e1af
# ╟─51077e41-40a0-488e-aac3-ab8e569a19a1
# ╟─e1eb954b-a92a-427f-8848-3a51eaecf4ca
# ╠═8f49a05e-77b0-4750-bf4a-de227f514721
# ╟─ff83dae1-32be-448d-a976-da6a8f0c474d
# ╠═b5d9dc08-9a74-4d22-856e-5a795d68294a
# ╟─640fee3f-2473-436a-9903-40dd30b322de
# ╠═bf3d94c0-853b-4128-bd0f-fefe727c7e4b
# ╠═50505804-6d5e-4a42-83c6-90c189f333cc
# ╟─e471dabf-98b9-4da7-8647-9e4444586390
# ╠═8a4db9c8-9da2-4432-a884-f05a0a4dda0a
# ╟─168035dd-2450-4edb-81b1-644c0f6e93f2
# ╟─e3468c7f-ec57-441e-935c-fdf93df4d624
# ╟─b3a5633e-69e7-47f8-a774-c6cc2d8cf70e
# ╠═01169ef1-d9b6-4ea5-b565-c8bfaed2bc0c
# ╠═884c0c94-5be9-4de4-8d49-5b9b10bdccb8
# ╟─0135ceb2-43fc-4154-89ba-9175cd55a307
# ╟─9b83bbc4-fefb-425e-94c1-6127ac15b39a
# ╠═9458472a-3311-474e-a68b-fa6f5fddd5db
# ╟─edcc81c0-6d9f-4d30-bb03-161d58f5fd4b
# ╠═1a552db3-65fa-41eb-a9be-48453bbdd4d0
# ╠═22ba800b-a6db-41b8-b31f-c926fa26fee6
# ╠═09af83f7-765b-42e9-97c6-7d31d25f0d3c
# ╠═5c1774b9-b5a3-40fc-bc28-9f6981937598
# ╟─74312aba-6062-4d40-99e6-d04ea1204a13
# ╠═f5982a13-0d25-42b5-8053-f58668740064
# ╠═7c27804d-8c42-4d67-9df9-ce7874b24c68
# ╠═b79c7d56-d0dd-4910-ba3b-59b1b737219d
# ╠═04f2cc03-2b9c-4544-8d3c-2a9caccfec4c
# ╟─1ccb168e-0739-4482-a6a7-853ffd2ff83b
# ╠═cf674376-95a3-4e87-a605-533ea7fddc48
# ╠═c57360f5-92c5-4fa6-be9d-adc119077159
# ╠═db0296c6-1f39-4a71-bf41-b6cd9497d6ef
# ╠═f59666c9-99c6-4de3-9dae-1cc31df0dee6
# ╠═369e4698-1657-4d97-bf6c-fc8f0e736401
