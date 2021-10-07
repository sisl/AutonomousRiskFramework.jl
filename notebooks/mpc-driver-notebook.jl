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

# ╔═╡ 71611590-cec1-11eb-0f9a-979a2941b1b1
begin
	using Revise
	using Distributions
	using Cairo, Gtk
	using AutomotiveSimulator, AutomotiveVisualization
	using IntelligentDriving
	ID = IntelligentDriving
	
	include("..\\IntelligentDriving.jl\\test\\MPCDriver\\draw_utils.jl")
end

# ╔═╡ 879d79f6-5ff2-4d09-b9a7-c3d1c084edcd
using PlutoUI

# ╔═╡ 87d1f23c-179b-4300-b1ef-9e59df7b08c5
begin
	roadway = gen_stadium_roadway(3)
	w = DEFAULT_LANE_WIDTH
	scene = Scene([
	  Entity(
		VehicleState(VecSE2(30.0, -w, 0.0), roadway, 10.0),
		VehicleDef(),
		:alice,
	  ),
	  Entity(
		VehicleState(VecSE2(40.0, 0.0, 0.0), roadway, 20.0),
		VehicleDef(),
		:bob,
	  ),
	  Entity(
		VehicleState(VecSE2(10.0, -w, 0.0), roadway, 15.0),
		VehicleDef(),
		:charlie,
	  ),
	])
	car_colors = get_pastel_car_colors(scene)

	renderables = vcat(
	  [roadway],
	  [FancyCar(car = veh, color = car_colors[veh.id]) for veh in scene],
	)
	snapshot = render(renderables)
end

# ╔═╡ 6738c170-d78b-4415-bfb3-b28e3bc302fc
begin
	timestep = 0.1
	nticks = 500
end

# ╔═╡ dd89470e-73ae-4063-b400-6d29b5624f24
begin
	models = Dict{Symbol,DriverModel}(
	  :alice => LatLonSeparableDriver( # produces LatLonAccels
	    ProportionalLaneTracker(), # lateral model
	    IntelligentDriverModel(), # longitudinal model
	  ),
	  #:bob => Tim2DDriver(mlane = MOBIL()),
	  :bob => LatLonSeparableDriver( # produces LatLonAccels
	    ProportionalLaneTracker(), # lateral model
	    IntelligentDriverModel(), # longitudinal model
	  ),
	  :charlie => MPCDriver(),
	)
	
	set_desired_speed!(models[:alice], 12.0)
	set_desired_speed!(models[:bob], 12.0)
	set_desired_speed!(models[:charlie], 15.0)
end;

# ╔═╡ 49e83e29-e2a5-40be-8873-007ca547803e
scenes = simulate(scene, roadway, models, nticks, timestep);

# ╔═╡ 3d04bb96-ebfd-4a9e-9807-305575583fff
@bind t Slider(1:length(scenes))

# ╔═╡ fa50f32d-ce5d-44fc-85e3-94425e3be3a5
render([roadway, scenes[t]])

# ╔═╡ Cell order:
# ╠═71611590-cec1-11eb-0f9a-979a2941b1b1
# ╠═87d1f23c-179b-4300-b1ef-9e59df7b08c5
# ╠═6738c170-d78b-4415-bfb3-b28e3bc302fc
# ╠═dd89470e-73ae-4063-b400-6d29b5624f24
# ╠═49e83e29-e2a5-40be-8873-007ca547803e
# ╠═879d79f6-5ff2-4d09-b9a7-c3d1c084edcd
# ╟─3d04bb96-ebfd-4a9e-9807-305575583fff
# ╠═fa50f32d-ce5d-44fc-85e3-94425e3be3a5
