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

# ╔═╡ 3bca7980-fc80-11ea-0bde-b5e9b35bdc9b
try using AddPackage catch; using Pkg; Pkg.add("AddPackage") end

# ╔═╡ 4e5ea39e-fc80-11ea-103a-f3a1cea1550a
@add using AutomotiveSimulator, AutomotiveVisualization

# ╔═╡ 13f1e010-fc85-11ea-1a3d-ddb4f27dde6b
@add using PlutoUI

# ╔═╡ 15cecfb0-fc80-11ea-259e-0dbe9a2978e0
md"""
# Automotive driving example
Using the *straight roadways* example from [AutomotiveSimulator.jl](https://sisl.github.io/AutomotiveSimulator.jl/dev/)
"""

# ╔═╡ 5bfe16ce-fc85-11ea-00e8-79b38894afdf
# pkg"add https://github.com/mossr/BeautifulAlgorithms.jl"

# ╔═╡ 65be6d50-fc85-11ea-3178-41eac8a5d2d9
md"""
## Roadway
"""

# ╔═╡ adf91900-fc83-11ea-384f-412e4b296dd3
roadway = gen_straight_roadway(3, 200.0) # lanes and length (meters)

# ╔═╡ 79a703e0-fc85-11ea-3858-3d92dbd58550
md"""
## Scenes
Includes different `Entity` objects, i.e. agents/vehicles/pedestrians.
"""

# ╔═╡ 3ea1dfa0-fc84-11ea-384a-4f9b2742ab1d
scene = Scene([
	Entity(VehicleState(VecSE2(10.0, 0, 0), roadway, 8.0), VehicleDef(), 1),
	Entity(VehicleState(VecSE2(50.0, 0, 0), roadway, 12.5), VehicleDef(), 2),
	Entity(VehicleState(VecSE2(150.0, 0, 0), roadway, 6.0), VehicleDef(), 3)
]);

# ╔═╡ 8f5de740-fc84-11ea-0fbe-e154f6d379a6
md"Order does not necessarily match ID."

# ╔═╡ 82dee5f0-fc84-11ea-2abf-9746b8cff50e
vechile1 = get_by_id(scene, 1)

# ╔═╡ 8d628c70-fc84-11ea-0243-99cb4e676b35
camera = StaticCamera(position=VecE2(95.0, 3.0), zoom=4, canvas_height=100);

# ╔═╡ a87e6600-fc84-11ea-2e77-c7336634e6c6
snapshot = render([roadway, scene], camera=camera)

# ╔═╡ e29d21e0-fc85-11ea-0fcd-217f25b30fc4
md"## Simulation"

# ╔═╡ ed8c8280-fc85-11ea-3726-fd3e9452dd42
models = Dict{Int, LaneFollowingDriver}(
	1 => StaticLaneFollowingDriver(0.0), # zero acceleration
	2 => IntelligentDriverModel(v_des=12.0), # IDM with 12 m/s speed
	3 => PrincetonDriver(v_des=10.0) # Princeton driver model with selected speed
)

# ╔═╡ 1c9cd980-fc86-11ea-2573-79eb8a0a17be
nticks, timestep = 200, 1.0;

# ╔═╡ 41251d2e-fc86-11ea-1028-415f6f3994a0
scenes = simulate(scene, roadway, models, nticks, timestep);

# ╔═╡ af80d500-fc84-11ea-18c8-81031a0c6534
md"""
## Animation
"""

# ╔═╡ 838a4450-fc88-11ea-215e-5d246a672a9e
dt = 1

# ╔═╡ 486ff4f0-fc88-11ea-3f52-79c9604b9bb2
@bind sim_t Slider(1:dt:length(scenes))

# ╔═╡ 37463040-fc88-11ea-1048-01928fd99839
render([roadway, scenes[sim_t]], camera=camera)

# ╔═╡ Cell order:
# ╟─15cecfb0-fc80-11ea-259e-0dbe9a2978e0
# ╠═5bfe16ce-fc85-11ea-00e8-79b38894afdf
# ╠═3bca7980-fc80-11ea-0bde-b5e9b35bdc9b
# ╠═4e5ea39e-fc80-11ea-103a-f3a1cea1550a
# ╟─65be6d50-fc85-11ea-3178-41eac8a5d2d9
# ╠═adf91900-fc83-11ea-384f-412e4b296dd3
# ╟─79a703e0-fc85-11ea-3858-3d92dbd58550
# ╠═3ea1dfa0-fc84-11ea-384a-4f9b2742ab1d
# ╟─8f5de740-fc84-11ea-0fbe-e154f6d379a6
# ╠═82dee5f0-fc84-11ea-2abf-9746b8cff50e
# ╠═8d628c70-fc84-11ea-0243-99cb4e676b35
# ╠═a87e6600-fc84-11ea-2e77-c7336634e6c6
# ╟─e29d21e0-fc85-11ea-0fcd-217f25b30fc4
# ╠═ed8c8280-fc85-11ea-3726-fd3e9452dd42
# ╠═1c9cd980-fc86-11ea-2573-79eb8a0a17be
# ╠═41251d2e-fc86-11ea-1028-415f6f3994a0
# ╟─af80d500-fc84-11ea-18c8-81031a0c6534
# ╠═13f1e010-fc85-11ea-1a3d-ddb4f27dde6b
# ╠═838a4450-fc88-11ea-215e-5d246a672a9e
# ╠═486ff4f0-fc88-11ea-3f52-79c9604b9bb2
# ╠═37463040-fc88-11ea-1048-01928fd99839
