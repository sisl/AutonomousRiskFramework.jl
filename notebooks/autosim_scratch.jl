### A Pluto.jl notebook ###
# v0.11.14

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

# ╔═╡ 728689fe-05a3-11eb-3591-6b47f8f373c0
try
	using Pkg
	using AddPackage
catch
	Pkg.add("AddPackage")
	using AddPackage
end

# ╔═╡ 7462eb22-05a3-11eb-2385-ff5a8db350a2
@add using AutomotiveSimulator, AutomotiveVisualization

# ╔═╡ ad81b89e-05a3-11eb-3580-fbc9fa14a929
@add using PlutoUI

# ╔═╡ fafdf400-05a2-11eb-037e-69fd345b618e
md"# Automotive simulator *(scratch)*"

# ╔═╡ 86363640-05a3-11eb-01c2-a5cb49b6a9dd
md"### Roadway (stadium)"

# ╔═╡ 971b9eee-05a3-11eb-0441-6db5042cb234
roadway = gen_stadium_roadway(2)

# ╔═╡ 99dd08e0-05a3-11eb-189c-7956508628ae
md"""
### Scenes
Includes different `Entity` objects, i.e. agents/vehicles/pedestrians.
"""

# ╔═╡ 9b9df2c0-05a3-11eb-2045-05924665bb60
w = DEFAULT_LANE_WIDTH

# ╔═╡ 9d760e20-05a3-11eb-3b57-e506a95c2e03
scene = Scene([
	Entity(VehicleState(VecSE2(10.0, 0, 0), roadway,  8.0), VehicleDef(), 1),
	Entity(VehicleState(VecSE2(30.0, 0, 0), roadway,  6.0), VehicleDef(), 2)
]);

# ╔═╡ 9f52e470-05a3-11eb-04e2-27ed7cc709b9
camera = StaticCamera(position=VecE2(50, 30), zoom=4,
	                  canvas_height=400, canvas_width=700);

# ╔═╡ a0fe7190-05a3-11eb-218a-193e36fb3cc8
snapshot = AutomotiveVisualization.render([roadway, scene], camera=camera)

# ╔═╡ a348afb2-05a3-11eb-2643-21599e1d7b03
md"### Simulation"

# ╔═╡ a516e000-05a3-11eb-29f9-552e0c96ed62
models = Dict{Int, DriverModel}(
	1 => IntelligentDriverModel(v_des=4.0),
	2 => Tim2DDriver(mlon=IntelligentDriverModel(v_des=4.0), mlane=TimLaneChanger())
)

# ╔═╡ a709af50-05a3-11eb-2067-d3caded88595
nticks, timestep = 2000, 1.0;

# ╔═╡ a9024afe-05a3-11eb-3eab-05b039811dcf
scenes = AutomotiveSimulator.simulate(scene, roadway, models, nticks, timestep);

# ╔═╡ abf54c40-05a3-11eb-1e2c-516d71bec964
md"### Animation"

# ╔═╡ aee03730-05a3-11eb-0054-a19718e42a0e
dt = 1

# ╔═╡ b0951320-05a3-11eb-354e-df59ae0591e2
@bind sim_t Slider(1:dt:length(scenes))

# ╔═╡ b2915850-05a3-11eb-3c2b-7b00e9106b82
AutomotiveVisualization.render([roadway, scenes[sim_t]], camera=camera)

# ╔═╡ b5fda840-05a3-11eb-244d-1b30f733dbc1
md"### Distance Metric"

# ╔═╡ b79ba0d0-05a3-11eb-3b37-4754bb2511c1
get_distance(scenes[sim_t][1], scenes[sim_t][2])

# ╔═╡ Cell order:
# ╟─fafdf400-05a2-11eb-037e-69fd345b618e
# ╠═728689fe-05a3-11eb-3591-6b47f8f373c0
# ╠═7462eb22-05a3-11eb-2385-ff5a8db350a2
# ╟─86363640-05a3-11eb-01c2-a5cb49b6a9dd
# ╠═971b9eee-05a3-11eb-0441-6db5042cb234
# ╟─99dd08e0-05a3-11eb-189c-7956508628ae
# ╠═9b9df2c0-05a3-11eb-2045-05924665bb60
# ╠═9d760e20-05a3-11eb-3b57-e506a95c2e03
# ╠═9f52e470-05a3-11eb-04e2-27ed7cc709b9
# ╠═a0fe7190-05a3-11eb-218a-193e36fb3cc8
# ╟─a348afb2-05a3-11eb-2643-21599e1d7b03
# ╠═a516e000-05a3-11eb-29f9-552e0c96ed62
# ╠═a709af50-05a3-11eb-2067-d3caded88595
# ╠═a9024afe-05a3-11eb-3eab-05b039811dcf
# ╟─abf54c40-05a3-11eb-1e2c-516d71bec964
# ╠═ad81b89e-05a3-11eb-3580-fbc9fa14a929
# ╠═aee03730-05a3-11eb-0054-a19718e42a0e
# ╠═b0951320-05a3-11eb-354e-df59ae0591e2
# ╠═b2915850-05a3-11eb-3c2b-7b00e9106b82
# ╟─b5fda840-05a3-11eb-244d-1b30f733dbc1
# ╠═b79ba0d0-05a3-11eb-3b37-4754bb2511c1
