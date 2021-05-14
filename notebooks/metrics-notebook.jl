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

# ╔═╡ baaee774-3b4e-4741-a6b6-44f8236053a8
begin
	using Distributions
	using Plots
	using PGFPlotsX
	using StatsBase
	using LinearAlgebra
	using PlutoUI
	using LaTeXStrings
end

# ╔═╡ 389837c0-a3b5-11eb-0525-d9e8c8498141
# gr()
# pgfplotsx()
pyplot() # polar plot xtick labels

# ╔═╡ 15ee1d99-c94b-450e-ab82-2f4ca45e625e
begin
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amsmath}")
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amsfonts}")
end;

# ╔═╡ b6de8810-a3b4-11eb-0623-6ffb251e8436
begin
	N = 20
	N_fail = 2
	N_succ = N-N_fail
end;

# ╔═╡ bc657010-a3b3-11eb-140c-791072671c01
P = Beta(N_fail, N_succ) # proxy (this would be unknown to us)

# ╔═╡ e125b3b0-a3b3-11eb-1428-2dfc7b0cd20a
Z = 10rand(P, 100_000) # cost data

# ╔═╡ e26f6763-04f4-46f1-9a29-57aae8bf91b4
md"""
# Risk Metrics
"""

# ╔═╡ 4d0d1157-4eb6-4d16-910d-6869b217b153
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 1e1a03cb-d46c-4c76-82f7-f7a5480afab0
plotting = ingredients("../RiskSimulator.jl/src/visualization/plotting.jl");

# ╔═╡ 771f0b67-e4fa-43e4-a314-e384c61bf80a
risk = ingredients("../RiskSimulator.jl/src/risk_assessment.jl");

# ╔═╡ c192175d-b436-4c2b-93af-b1e4b5e6cfa1
md"""
$$\operatorname{VaR}_\alpha(Z) := \min\left\{ z \mid \mathcal{P}(Z > z) \le \alpha \right\}\tag{lowest cost given probability threshold}$$
"""

# ╔═╡ 5ed4cc85-5286-44ea-93e0-1dd509d3de54
md"""
$$\begin{align}
	\operatorname{CVaR}_\alpha(Z) &:= \frac{1}{\alpha}\int_{1-\alpha}^{1} \operatorname{VaR}_{1-\tau}(Z) d\tau\tag{expected cost under conditional distribution}\\
	&:= \mathbb{E}_{\mathcal{P}'}[Z] \tag{where $\mathcal{P}'(z) := \mathbb{P}\left\{z \mid \mathcal{P}(Z > z) \le \alpha\right\}$}
\end{align}$$
"""

# ╔═╡ db2081d4-2041-4157-be16-b00f961f1c9b
md"""
We accept $\alpha$ percentage of the risk.
"""

# ╔═╡ d73331fc-4dea-4456-98f8-d260e320cc09
@bind α Slider(0.001:0.001:1, default=0.08) 

# ╔═╡ 84e246f5-544a-450a-8f45-9bb8f0108994
metrics = risk.RiskMetrics(Z, α);

# ╔═╡ 5d3a146c-f6bb-4d2a-9dfd-610e5af4b032
risk.latex_metrics(metrics)

# ╔═╡ fc7b737a-fc8a-4750-b0fc-5e10bd0b454f
plotting.risk_plot(metrics)

# ╔═╡ c002adfa-8b2a-48a1-b118-4f2a5117b753
md"""
# Polar Area Plots
"""

# ╔═╡ d5d2c789-c0c3-4152-8390-453a38fb6f9c
md"""
We accept $\alpha$ percentage of risk, so when changing $\alpha$:
- If we lower $\alpha$, we are saying "we want to accept _less_ risk", so the area-under-curve (AUC) will be larger, i.e., more measured risk.

- If we increase $\alpha$, we can accept _more_ risk, so the AUC will be smaller, i.e., less measured risk. 
"""

# ╔═╡ 89fa3dff-2393-4d91-84dd-e0500ea8fe1a
md"Visualize risk area? $(@bind viz CheckBox())"

# ╔═╡ 341354fd-4291-46e2-9bf2-f6b8a4c83984
𝐰 = [1, 1, 1, 1];

# ╔═╡ 3d7c55d4-50e7-4650-b1ec-13969c4afa2d
viz && plotting.plot_risk_metrics([metrics], ["SUT"]; weights=𝐰)

# ╔═╡ 9feb4911-db29-45d2-ad20-6193e8319381
plotting.risk_area([metrics]; weights=𝐰)

# ╔═╡ c989db45-0ca4-45f1-8208-9075eb1b9734
md"""
# Sample Polar Plots
"""

# ╔═╡ 4457f416-6474-4996-8293-f2201abf7584
n = 2

# ╔═╡ 5e629873-19c7-4b58-903e-45c683290ec5
Y = plotting.copyend(rand(n))

# ╔═╡ b6fe7af9-0320-4b6a-a3ac-efabd397602b
Y2 = plotting.copyend(rand(n))

# ╔═╡ 88c79cfb-8176-46b4-8d72-645c2fc07259
Y3 = plotting.copyend(rand(n))

# ╔═╡ 1526d4ab-d3b7-4033-a983-6f7642a7f857
viz && plotting.plot_metrics([Y,Y2,Y3], ["one","two","three"], ["SUT1","SUT2","SUT3"])

# ╔═╡ Cell order:
# ╠═baaee774-3b4e-4741-a6b6-44f8236053a8
# ╟─389837c0-a3b5-11eb-0525-d9e8c8498141
# ╠═15ee1d99-c94b-450e-ab82-2f4ca45e625e
# ╠═b6de8810-a3b4-11eb-0623-6ffb251e8436
# ╠═bc657010-a3b3-11eb-140c-791072671c01
# ╠═e125b3b0-a3b3-11eb-1428-2dfc7b0cd20a
# ╟─e26f6763-04f4-46f1-9a29-57aae8bf91b4
# ╟─4d0d1157-4eb6-4d16-910d-6869b217b153
# ╠═1e1a03cb-d46c-4c76-82f7-f7a5480afab0
# ╠═771f0b67-e4fa-43e4-a314-e384c61bf80a
# ╟─c192175d-b436-4c2b-93af-b1e4b5e6cfa1
# ╟─5ed4cc85-5286-44ea-93e0-1dd509d3de54
# ╠═84e246f5-544a-450a-8f45-9bb8f0108994
# ╟─db2081d4-2041-4157-be16-b00f961f1c9b
# ╠═d73331fc-4dea-4456-98f8-d260e320cc09
# ╠═5d3a146c-f6bb-4d2a-9dfd-610e5af4b032
# ╠═fc7b737a-fc8a-4750-b0fc-5e10bd0b454f
# ╟─c002adfa-8b2a-48a1-b118-4f2a5117b753
# ╟─d5d2c789-c0c3-4152-8390-453a38fb6f9c
# ╟─89fa3dff-2393-4d91-84dd-e0500ea8fe1a
# ╠═341354fd-4291-46e2-9bf2-f6b8a4c83984
# ╠═3d7c55d4-50e7-4650-b1ec-13969c4afa2d
# ╠═9feb4911-db29-45d2-ad20-6193e8319381
# ╟─c989db45-0ca4-45f1-8208-9075eb1b9734
# ╠═4457f416-6474-4996-8293-f2201abf7584
# ╠═5e629873-19c7-4b58-903e-45c683290ec5
# ╠═b6fe7af9-0320-4b6a-a3ac-efabd397602b
# ╠═88c79cfb-8176-46b4-8d72-645c2fc07259
# ╠═1526d4ab-d3b7-4033-a983-6f7642a7f857
