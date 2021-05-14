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

# ╔═╡ e1cc5eae-c7dd-4e14-ac53-c76dc83cf831
begin
	function ingredients(path::String)
		# this is from the Julia source code (evalfile in base/loading.jl)
		# but with the mod. that it returns the module instead of the last object
		name = Symbol(basename(path))
		m = Module(name)
		Core.eval(m,
			Expr(:toplevel,
				 :(eval(x) = $(Expr(:core, :eval))($name, x)),
				 :(include(x) = $(Expr(:top, :include))($name, x)),
				 :(include(mapexpr::Function, x) =
					$(Expr(:top, :include))(mapexpr, $name, x)),
				 :(include($path))))
		return m
	end

	# using PyPlot, Seaborn
	ast = ingredients("../ObservationModels/examples/organized_ast.jl")
	plotting = ingredients("../RiskSimulator.jl/src/visualization/plotting.jl")

	using Distributions
	using PlutoUI
end

# ╔═╡ ce186b02-7e6a-4f46-8d5b-548a3689eebe
using MixFit

# ╔═╡ 345067fa-8d53-4847-944d-e373f1c0dd01
md"""
## End-to-End Autonomous Risk Framework
A demo of the end-to-end risk assessment framework comparing the Intelligent Driver Model (IDM) with the Princeton Driver Model.
"""

# ╔═╡ 696414a6-f2ac-437f-a0a0-c8e13d35377e
md"""
# Intelligent Driver Model
"""

# ╔═╡ 7e0a2028-5b33-4050-9b3e-ef7f29c2de3d
md"""
Set RNG seeds for determinism.
"""

# ╔═╡ 929c42ed-11c3-48f9-ae8a-74cbff0f8586
begin
	SEED = 1000
	ast.Random.seed!(SEED);
end;

# ╔═╡ 0aabf8be-93fa-4022-afb9-d381dd14ba5d
md"""
## Phase 1: Observation model fitting
"""

# ╔═╡ 12adfc80-775b-4105-8fff-52d4d5e4d274
md"""
Train a surrogate model (using a neural network in this case) to replace the observation model.
"""

# ╔═╡ d89e1d69-9339-4c68-a1db-c87bb463ee12
SEED; net, net_params = ast.training_phase();

# ╔═╡ a58bb08d-1507-41d4-b9d5-620e52700a67
md"""
## Phase 2: Standard AST Failure Search
"""

# ╔═╡ 89c59d7d-d0d6-425f-8cdb-b4b10cc0a211
md"""
1. Select system under test (i.e., autonomous vehicle platform)
"""

# ╔═╡ 727b5568-89ec-4c98-b6b3-08aa03442777
system = ast.IntelligentDriverModel(v_des=12.0);

# ╔═╡ d54a2257-8ff2-4ce5-b80e-ba2434e14d2f
md"""
2. Set-up AST search problem.
"""

# ╔═╡ f3459a95-fd43-495a-8c5b-362dc4f35536
planner = ast.setup_ast(net, net_params; sut=system, seed=SEED);

# ╔═╡ 5308a9c1-82fb-424b-bb0b-26d2fe64b04a
md"""
3. Run AST search.
"""

# ╔═╡ 18d03acb-5dcf-474f-a8cb-2c7f96ad99af
action_trace = ast.search!(planner);

# ╔═╡ f599cb73-cd85-4f6e-96f6-c63e6802a03c
md"""
#### Failure metrics
We can capture **failure metrics**:
- failure rate (i.e., biased $p(\text{fail})$)
- first failure index (i.e., "ease" of finding failures)
- number of failures
- highest log-likelihood of failure
"""

# ╔═╡ 65e3ea70-493f-4d48-a056-daec44993e01
failure_metrics = ast.failure_metrics(planner);

# ╔═╡ 0f7dd3f7-e595-4d4f-9770-387b9f58ad32
ast.POMDPStressTesting.latex_metrics(failure_metrics)

# ╔═╡ 6f302f7c-e2fd-4267-a8e6-57ea493c2699
md"""
Let $\alpha$ be the risk threshold over cost distribution (i.e., severity of failure distribution).
"""

# ╔═╡ 746c7128-c854-4cdd-82a8-0566287b1c35
α = 0.2;

# ╔═╡ 0a40b9d0-835a-4986-85b3-26a2ee96aeea
md"""
We accept $\alpha$ percentage of risk, so when changing $\alpha$:
- If we lower $\alpha$, we are saying "we want to accept _less_ risk", so the area-under-curve (AUC) will be larger, i.e., more measured risk.

- If we increase $\alpha$, we can accept _more_ risk, so the AUC will be smaller, i.e., less measured risk. 
"""

# ╔═╡ 02ff1404-1914-44da-a5ea-de59e854f702
md"""
Collect cost value $Z$ (i.e., severity of failure). We use the _rate_ value already collected and used by AST:

$$\text{rate} = d_{t+1} - d_t$$

where $d$ is the miss distance metric already collected by AST (preserving the black-box assumption).
"""

# ╔═╡ 7560d798-3e9d-4800-9f47-73c9b03eed8f
𝒟 = planner.mdp.dataset;

# ╔═╡ a43e2484-ff74-4d7b-a3e7-fc13cb9558a1
plotting.plot_closure_rate_distribution(𝒟; reuse=false)

# ╔═╡ 9045b21a-fc55-414e-a490-5f568a22efb0
md"""
#### Risk metrics
Now we can also collect the **risk metrics**:
- Expected cost $\mathbb{E}[Z]$ (i.e., mean cost)
- Value at Risk $\operatorname{VaR}$
- Conditional Value at Risk $\operatorname{CVaR}$
- Worst case cost
"""

# ╔═╡ 9641d717-229e-48cc-a105-455ea250ddaf
md"""
Collect cost/severity data $Z$.
"""

# ╔═╡ 2538080d-ea56-46ae-b4eb-ecf4a2807b6b
Z = ast.cost_data(𝒟);

# ╔═╡ 9c250ac1-ddf9-46ee-9d53-a43ff55ca765
metrics = ast.RiskMetrics(Z, α);

# ╔═╡ 9c54880f-a81d-456f-aad4-b89bc429f2c8
ast.latex_metrics(metrics)

# ╔═╡ 347023a3-4f9a-4f1d-8def-1c88d955301e
plotting.plot_risk(metrics; mean_y=0.33, var_y=0.25, cvar_y=0.1, α_y=0.2)

# ╔═╡ cc812055-735b-4fb6-9300-d5c14a3531a0
md"Vizualize failure? $(@bind viz CheckBox())"

# ╔═╡ de9ffff5-93f5-4cd7-a402-4d6b81b8fee1
viz && ast.visualize_most_likely_failure(planner, ast.buildingmap);

# ╔═╡ 06691cee-80cf-4863-a52f-4ad6aeff8deb
md"""
# Princeton Driver Model
"""

# ╔═╡ b0f742ab-1144-4a77-bea7-1ad28bcfba42
md"""
## Phase 2: Standard AST Failure Search
"""

# ╔═╡ 4993fbcb-8ce6-4b2b-9001-c774825b0811
md"""
Now we can change the SUT to use the Princeton Driver Model.
"""

# ╔═╡ 791b9070-c99e-4df1-9d56-ffe01aaed609
system2 = ast.PrincetonDriver(v_des=2.2);

# ╔═╡ 4e4801f5-7125-4d1b-bb14-650254ac68a9
md"""
And set-up our AST search problem, passing a different value for `sut`.
"""

# ╔═╡ 68d0160f-91c2-4a99-afd5-c5548b668c4a
planner2 = ast.setup_ast(net, net_params; sut=system2, seed=SEED);

# ╔═╡ 40eca7fb-cbe4-4908-a522-d6747d46efe1
md"""
And finally, run the AST search.
"""

# ╔═╡ 2252129b-407d-4df3-ad6d-bd7fff7d29ce
action_trace2 = ast.search!(planner2);

# ╔═╡ a82c4d5e-8beb-4e91-ae2e-07972d1a8882
𝒟2 = planner2.mdp.dataset;

# ╔═╡ 0f93a146-1a9e-466b-b44a-e6a702d076cf
plotting.plot_closure_rate_distribution(𝒟2; reuse=false)

# ╔═╡ 01c76e6f-caca-4042-a70f-2516c6f66f36
md"""
Collect **failure metrics**.
"""

# ╔═╡ 8d77bc0f-f78e-4b80-901b-69a8d0770e2f
failure_metrics2 = ast.failure_metrics(planner2);

# ╔═╡ 888cb1a8-f18f-460e-ae79-93505c45045e
ast.POMDPStressTesting.latex_metrics(failure_metrics2)

# ╔═╡ dd0b95ed-7955-4f0e-bd0c-8ed62bcce9cf
md"""
Collect **risk metrics**.
"""

# ╔═╡ c3d482ba-7129-40eb-8dbc-d46b1bee04d7
Z2 = ast.cost_data(planner2.mdp.dataset);

# ╔═╡ fffdded8-69a1-4361-90a8-1a9de0256965
metrics2 = ast.RiskMetrics(Z2, α);

# ╔═╡ f30bd6a9-7fdf-448c-b28b-662b5afe8e5a
ast.latex_metrics(metrics2)

# ╔═╡ 5046d560-2deb-4702-8272-bc9b27ef8462
plotting.plot_risk(metrics2; mean_y=0.18, var_y=0.15, cvar_y=0.1, α_y=0.13)

# ╔═╡ c1ded51d-dacd-4908-90a7-99a20283ea19
md"Vizualize failure? $(@bind viz2 CheckBox())"

# ╔═╡ ce5ee9cf-f614-4b48-b1e1-58791f7256ad
viz2 && ast.visualize_most_likely_failure(planner2, ast.buildingmap);

# ╔═╡ 4f29753c-f0e3-42be-baab-a06da23a0255
md"""
# Overall Risk Measure
Using the risk metrics (i.e., mean cost, VaR, CVaR, and worst case cost), we can compute a polar plot to visualize the overall risk.

We can adjust the weight of each metrics using the vector $\mathbf{w}$.
"""

# ╔═╡ e71c6ef2-09b2-40e6-ae42-08810de2a01a
𝐰 = [1, 1, 1, 1];

# ╔═╡ 416b0fc0-d644-48a9-ad06-a6dfb7accc5c
plotting.plot_risk_metrics([metrics, metrics2], ["IDM", "Princeton"]; weights=𝐰)

# ╔═╡ c2cfad18-9ee5-412c-be31-89b9ab4b0aea
md"""
Compute the area under each curve (higher = more risky).
"""

# ╔═╡ 241fc15d-07a8-486e-9d16-4b2e95941f1f
plotting.risk_area([metrics, metrics2]; weights=𝐰)

# ╔═╡ 75b14899-2165-4b04-a1aa-6298e40380c8
md"""
Now using _all_ metrics (i.e., risk metrics _and_ failure metrics from AST), we can recompute the polar plots and compute each AV's risk measure as the area under the curve.
"""

# ╔═╡ 09c827a6-7782-4449-9953-df585514b72d
𝐰′ = [1, 1, 1, 1, 1, 1, 1];

# ╔═╡ f9c101cc-6861-4ac3-95f9-a96a6fa757f7
plotting.plot_overall_metrics([planner,planner2], ["IDM","Princeton"]; weights=𝐰′,α=α)

# ╔═╡ ed25fe07-36f3-4264-99ba-6181eb4adca9
md"""
Re-calculate the risk area using _both_ risk metrics and failure metrics.
"""

# ╔═╡ 70c38db1-229c-4bd0-89b1-6621d97f05f3
plotting.overall_area([planner, planner2]; weights=𝐰′, α=α)

# ╔═╡ d86498f0-4162-440a-8c15-9973bb508116
md"""
> **Notice** how the overall risk is different when including the _failure metrics_ combined with the _risk metrics_. The weights $\mathbf{w}'$ can be adjusted to balance how much each metric affects the overall risk.
"""

# ╔═╡ a58e787e-6082-4971-b445-c9fc18898ae0
md"""
# Distribution of failure likelihood
"""

# ╔═╡ 8c5b2c05-6ef4-4f5c-b964-d795ed0a3369
md"Display likelihood distribution? $(@bind llviz CheckBox())"

# ╔═╡ 7da84a62-979b-4b25-963c-9d6f461bcefa
llviz && using PyPlot, Seaborn, Revise, POMDPStressTesting

# ╔═╡ 21705423-986b-4c58-abc2-a697d7f4ded6
if llviz
	distribution_figures(planner.mdp.metrics; label="IDM")
	distribution_figures!(planner2.mdp.metrics; color="purple", label="Princeton")
end

# ╔═╡ 83525435-68c1-436e-8cd0-5e8fc0930eae
md"""
# Fitting and predictions
"""

# ╔═╡ c0bd451e-7667-4194-ad36-be05ecf16259
md"""
## Quadratic discriminant analysis (QDA)

To predict a failure given a rate $r$ and distance $d$ as $x=[r,d]$, we can use quadratic discriminant analysis (QDA):$^{[1]}$

$$\delta_k(x) = -\frac{1}{2}\mu_k\Sigma_k^{-1}\mu_k + x^\top\Sigma_k^{-1}\mu_k - \frac{1}{2}x^\top\Sigma_k^{-1}x - \frac{1}{2}\log|\Sigma_k|$$

and when $\delta_1(x) > \delta_2(x)$, then we classify as _failure_ (i.e., class 1).

We can also set $\delta_1(x) = \delta_2(x)$ and solve for $\delta_1(x)$ to get:

$$\delta(x) = (x - \mu_1)^\top\Sigma_1^{-1}(x - \mu_1) + \log(|\Sigma_1|) - (x - \mu_2)^\top\Sigma_2^{-1}(x - \mu_2) - \log(|\Sigma_2|)$$

and when $\delta(x) < 0$, then we classify as _failure_ (i.e., class 1).
"""

# ╔═╡ b6abf576-dfb1-4924-a171-09a9d0326851
 plotting2 = ingredients("../RiskSimulator.jl/src/visualization/plotting.jl")

# ╔═╡ d005ecc7-36dc-4b27-933e-c6b55152be13
 plotting3 = ingredients("../RiskSimulator.jl/src/visualization/plotting.jl")

# ╔═╡ 63c817ac-58b9-4304-87d1-4c7065af2393
qda_fig_linear = plotting2.plot_multivariate_distance_and_rate(𝒟;
	gda=true,
	gdatype=:qda,
	svm=true,
	boundary=:binary,
	subplots=false); savefig("qda_linear_boundary.pdf", bbox_inches = "tight",
    pad_inches = 0.1); qda_fig_linear

# ╔═╡ f772cb78-6939-4919-b4e0-7749029e9932
md"""
### QDA for classification
"""

# ╔═╡ 52f940e7-ba00-4bcb-8f6c-7643ce9b4646
qda_fig, predict_qda = plotting2.plot_multivariate_distance_and_rate(𝒟;
	gda=true,
	gdatype=:qda,
	svm=true,
	boundary=:gradient,
	return_predict=true,
	subplots=false); savefig("qda.pdf", bbox_inches = "tight",
    pad_inches = 0.1); qda_fig

# ╔═╡ 0f4b36fc-79bb-46e7-89b2-6dcd6a5c58be
classify(x, pred) = pred(x) < 0 ? :failure : :non_failure;

# ╔═╡ fd43367b-a33c-4c09-99e1-ecb7a8f1b293
show_prediction(x, pred) = println(x, "\t=> ", pred(x), "\t=> ", classify(x, pred));

# ╔═╡ 308441b8-b485-433f-b233-91de7da3b7e7
with_terminal() do
	show_prediction([0.2, 4], predict_qda)
	show_prediction([0.2, 5], predict_qda)
end

# ╔═╡ d20eba0f-6720-49ec-8f8f-e3f37c555f79
md"""
## Linear discriminant analysis (LDA)

If we (incorrectly) assume the covariances are the same (i.e., $\Sigma_1 = \Sigma_2 = \Sigma$), then we get linear discriminant analysis (LDA):

$$\delta_k(x) = x^\top\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k\Sigma^{-1}\mu_k + \log(\pi_k)$$

and when $\delta_1(x) > \delta_2(x)$, then we classify as _failure_ (i.e., class 1).

Similarly, we can also set $\delta_1(x) = \delta_2(x)$ and solve for $\delta_1(x)$ to get:

$$\delta(x) = (x - \mu_1)^\top\Sigma^{-1}(x - \mu_1) - (x - \mu_2)^\top\Sigma^{-1}(x - \mu_2)$$

and if $\delta(x) < 0$, then we classify as _failure_.
"""

# ╔═╡ 06241562-1ab8-4a6f-92ce-722a607a6d6d
lda_fig_linear, predict_lda_linear = plotting3.plot_multivariate_distance_and_rate(𝒟;
	gda=true,
	gdatype=:lda,
	svm=true,
	boundary=:binary,
	return_predict=true,
	subplots=true); savefig("lda_linear_boundary.pdf", bbox_inches = "tight",
    pad_inches = 0.1); lda_fig_linear

# ╔═╡ d20b6d40-0d60-4208-9aa1-76a7493418c5
predict_lda_linear([0.1, 4])

# ╔═╡ 88a3ef1f-a1fb-4965-9fc9-0656573897f1
lda_fig, predict_lda = plotting2.plot_multivariate_distance_and_rate(𝒟;
	gda=true,
	gdatype=:lda,
	svm=true,
	boundary=:gradient,
	return_predict=true,
	subplots=false); savefig("lda.pdf", bbox_inches = "tight",
    pad_inches = 0.1); lda_fig

# ╔═╡ 296c3373-2cab-49d6-a173-94e4fec021a8
predict_lda([0.1, 4])

# ╔═╡ 3ff338ac-cabb-4cc1-a326-282562901318
with_terminal() do
	show_prediction([0.2, 4], predict_lda)
	show_prediction([0.2, 5], predict_lda)
end

# ╔═╡ 11d024b2-2a12-4ebe-9c79-96e19028cdf1
begin
	Z_fail = ast.cost_data(𝒟)
    Z_nonfail = ast.cost_data(𝒟; nonfailures=true)
    d_fail = ast.distance_data(𝒟)
    d_nonfail = ast.distance_data(𝒟; nonfailures=true)
    
    pos_data = [Z_fail d_fail]'
    neg_data = [Z_nonfail d_nonfail]'
end

# ╔═╡ c18d944a-56b1-45bd-b974-ad08ad44df9c
pos_data'

# ╔═╡ 98aebd62-ba14-481b-9c28-97b9abf4bc4f
σ(z) = 1 / (1 + exp(-z))

# ╔═╡ d56c10a8-4feb-4621-a005-0206cfe0f14a
-predict_qda([0.5,3]) |> tanh # σ

# ╔═╡ 2b6a3941-5d03-4b3a-a8e8-1b3ebbc62f20
predict_qda([0.5, 3])

# ╔═╡ e5318d56-6289-4451-ac03-f5119223990a
svm_boundary = (x,w,b) -> (-w[1] * x .+ b)/w[2] # line of the decision boundary

# ╔═╡ 4942de35-2a00-4419-b959-b2caa90b361b
w, b = plotting2.compute_svm(pos_data, neg_data)

# ╔═╡ 2051d694-9864-4341-ace4-5ad0184c441a
svm_boundary(0, w, b)

# ╔═╡ 1ca3e292-ce0a-4cef-a08b-6d6628bad2c3
svm_classify(x) = sign((-w'*x + b)/w[2]) > 0 ? 1 : 0

# ╔═╡ ce934e7b-b915-4730-92e4-24f31bd1a745
svm_classify([0,4.08])

# ╔═╡ a95f7059-f862-4284-bfb2-17a753136e8b
sum([svm_classify(x) for x in eachcol(pos_data)]) / size(pos_data)[2]

# ╔═╡ 02a4fa64-c9c1-4b1d-9ce3-0128317b360a
sum([1-svm_classify(x) for x in eachcol(neg_data)]) / size(neg_data)[2]

# ╔═╡ 32fefe1e-7ec0-4756-afa8-70f2d0ae9b25
sum([predict_qda(x) < 0 ? 1 : 0 for x in eachcol(pos_data)]) / size(pos_data)[2]

# ╔═╡ 6903aa6c-26ca-4c6f-8dcd-45e9bb7a97b8
sum([1-predict_qda(x) < 0 ? 1 : 0 for x in eachcol(neg_data)]) / size(neg_data)[2]

# ╔═╡ 8009fcbf-892f-445b-a12d-3d2470c71a68
md"""
## Efficient Validation using Predictive Risk
"""

# ╔═╡ b7c80bcd-664a-49e5-a9f4-c84164a213be
begin
	planner3 = ast.setup_ast(net, net_params; sut=system, seed=SEED);
	planner3.mdp.predict = x -> -predict_qda(x) # NOTE negation.
	ast.search!(planner3);
	failure_metrics3 = ast.failure_metrics(planner3);
	ast.POMDPStressTesting.latex_metrics(failure_metrics3)
end

# ╔═╡ 67188719-ebb0-4dc1-8bcd-36ab2f6f1870
# 0.18 (normal), 0.51 (LDA), 0.65 (QDA)

# ╔═╡ 312fec2e-94c3-4beb-8e29-9a6a417541c3
md"""
# Multivariate Gamma Fits
"""

# ╔═╡ a6bc299e-deb9-4bb7-8661-52657377988a
figure(); scatter(pos_data[1,:], pos_data[2,:]); gcf()

# ╔═╡ 0fa42aa3-88b8-4786-a619-8bea71f6be8d
# Wishart(2, 

# ╔═╡ 632a6f89-1528-4142-820e-adeb9d41e48e
G = pos_data * pos_data'

# ╔═╡ 60d7695c-63cc-4757-807a-a1e294a86360
W = Wishart(2, G)

# ╔═╡ 601932f0-a0b1-4fdc-801f-e01b08087e43
pdf(W, [0.00001 0; 0 0.00000001])

# ╔═╡ 09272854-da18-417c-ab4c-a3db469f4c92
fit_mle(W, pos_data)

# ╔═╡ c3be7332-7451-4fba-a459-835b36f420f6
Γ₁ = fit_mle(Gamma, pos_data[1,:])

# ╔═╡ dc8afbce-17dd-470b-8ab3-5023605537d6
Γ₂ = fit_mle(Gamma, pos_data[2,:])

# ╔═╡ 721a32bd-1a1a-4f1f-bf1f-ffc09fcde6e2
pos_data

# ╔═╡ ce68a804-8cd1-4932-8b76-67b5b863ef75
pdfΓ(𝚪, 𝐱) = pdf(𝚪[1], 𝐱[1]) * pdf(𝚪[2], 𝐱[2])

# ╔═╡ 6dc2d711-4286-429f-97df-c360d7846e2b
pdfΓ([Γ₁, Γ₂], [0,4])

# ╔═╡ 1a936087-22f7-427a-829b-8f328e71046d
begin
	figure()
	scatter(pos_data[1,:], pos_data[2,:])
	current_xlim = xlim()
	current_ylim = ylim()
    fX = range(-2, stop=current_xlim[2], length=1000)
    fY = range(-2, stop=current_ylim[2], length=1000)
    fZ = [pdfΓ([Γ₁, Γ₂], [x,y]) for y in fY, x in fX] # Note x-y "for" ordering
    
    contour(fX, fY, fZ, alpha=0.75, cmap="plasma")
	xlim(current_xlim)
	ylim(current_ylim)
	gcf()
end

# ╔═╡ acbfdf1f-dcd8-4d2c-bcfc-ad221a808134
MF = mixfit([pos_data[1,:]; pos_data[2,:]], 2)

# ╔═╡ cd71d1f6-059a-4b65-ba40-57c02386998b
MF.μ

# ╔═╡ 046e4943-61c2-4258-b876-8b898819cd5b
md"""
---
[1]: T. Hastie, R. Tibshirani, and J. Friedman, _The Elements of Statistical Learning_, Springer, 2001.
"""

# ╔═╡ Cell order:
# ╠═e1cc5eae-c7dd-4e14-ac53-c76dc83cf831
# ╟─345067fa-8d53-4847-944d-e373f1c0dd01
# ╟─696414a6-f2ac-437f-a0a0-c8e13d35377e
# ╟─7e0a2028-5b33-4050-9b3e-ef7f29c2de3d
# ╠═929c42ed-11c3-48f9-ae8a-74cbff0f8586
# ╟─0aabf8be-93fa-4022-afb9-d381dd14ba5d
# ╟─12adfc80-775b-4105-8fff-52d4d5e4d274
# ╠═d89e1d69-9339-4c68-a1db-c87bb463ee12
# ╟─a58bb08d-1507-41d4-b9d5-620e52700a67
# ╟─89c59d7d-d0d6-425f-8cdb-b4b10cc0a211
# ╠═727b5568-89ec-4c98-b6b3-08aa03442777
# ╟─d54a2257-8ff2-4ce5-b80e-ba2434e14d2f
# ╠═f3459a95-fd43-495a-8c5b-362dc4f35536
# ╟─5308a9c1-82fb-424b-bb0b-26d2fe64b04a
# ╠═18d03acb-5dcf-474f-a8cb-2c7f96ad99af
# ╟─f599cb73-cd85-4f6e-96f6-c63e6802a03c
# ╠═65e3ea70-493f-4d48-a056-daec44993e01
# ╠═0f7dd3f7-e595-4d4f-9770-387b9f58ad32
# ╟─6f302f7c-e2fd-4267-a8e6-57ea493c2699
# ╠═746c7128-c854-4cdd-82a8-0566287b1c35
# ╟─0a40b9d0-835a-4986-85b3-26a2ee96aeea
# ╟─02ff1404-1914-44da-a5ea-de59e854f702
# ╠═7560d798-3e9d-4800-9f47-73c9b03eed8f
# ╠═a43e2484-ff74-4d7b-a3e7-fc13cb9558a1
# ╟─9045b21a-fc55-414e-a490-5f568a22efb0
# ╟─9641d717-229e-48cc-a105-455ea250ddaf
# ╠═2538080d-ea56-46ae-b4eb-ecf4a2807b6b
# ╠═9c250ac1-ddf9-46ee-9d53-a43ff55ca765
# ╠═9c54880f-a81d-456f-aad4-b89bc429f2c8
# ╠═347023a3-4f9a-4f1d-8def-1c88d955301e
# ╟─cc812055-735b-4fb6-9300-d5c14a3531a0
# ╠═de9ffff5-93f5-4cd7-a402-4d6b81b8fee1
# ╟─06691cee-80cf-4863-a52f-4ad6aeff8deb
# ╟─b0f742ab-1144-4a77-bea7-1ad28bcfba42
# ╟─4993fbcb-8ce6-4b2b-9001-c774825b0811
# ╠═791b9070-c99e-4df1-9d56-ffe01aaed609
# ╟─4e4801f5-7125-4d1b-bb14-650254ac68a9
# ╠═68d0160f-91c2-4a99-afd5-c5548b668c4a
# ╟─40eca7fb-cbe4-4908-a522-d6747d46efe1
# ╠═2252129b-407d-4df3-ad6d-bd7fff7d29ce
# ╠═a82c4d5e-8beb-4e91-ae2e-07972d1a8882
# ╠═0f93a146-1a9e-466b-b44a-e6a702d076cf
# ╟─01c76e6f-caca-4042-a70f-2516c6f66f36
# ╠═8d77bc0f-f78e-4b80-901b-69a8d0770e2f
# ╠═888cb1a8-f18f-460e-ae79-93505c45045e
# ╟─dd0b95ed-7955-4f0e-bd0c-8ed62bcce9cf
# ╠═c3d482ba-7129-40eb-8dbc-d46b1bee04d7
# ╠═fffdded8-69a1-4361-90a8-1a9de0256965
# ╠═f30bd6a9-7fdf-448c-b28b-662b5afe8e5a
# ╠═5046d560-2deb-4702-8272-bc9b27ef8462
# ╟─c1ded51d-dacd-4908-90a7-99a20283ea19
# ╠═ce5ee9cf-f614-4b48-b1e1-58791f7256ad
# ╟─4f29753c-f0e3-42be-baab-a06da23a0255
# ╠═e71c6ef2-09b2-40e6-ae42-08810de2a01a
# ╠═416b0fc0-d644-48a9-ad06-a6dfb7accc5c
# ╟─c2cfad18-9ee5-412c-be31-89b9ab4b0aea
# ╠═241fc15d-07a8-486e-9d16-4b2e95941f1f
# ╟─75b14899-2165-4b04-a1aa-6298e40380c8
# ╠═09c827a6-7782-4449-9953-df585514b72d
# ╠═f9c101cc-6861-4ac3-95f9-a96a6fa757f7
# ╟─ed25fe07-36f3-4264-99ba-6181eb4adca9
# ╠═70c38db1-229c-4bd0-89b1-6621d97f05f3
# ╟─d86498f0-4162-440a-8c15-9973bb508116
# ╟─a58e787e-6082-4971-b445-c9fc18898ae0
# ╟─8c5b2c05-6ef4-4f5c-b964-d795ed0a3369
# ╠═7da84a62-979b-4b25-963c-9d6f461bcefa
# ╠═21705423-986b-4c58-abc2-a697d7f4ded6
# ╟─83525435-68c1-436e-8cd0-5e8fc0930eae
# ╟─c0bd451e-7667-4194-ad36-be05ecf16259
# ╠═b6abf576-dfb1-4924-a171-09a9d0326851
# ╠═d005ecc7-36dc-4b27-933e-c6b55152be13
# ╠═63c817ac-58b9-4304-87d1-4c7065af2393
# ╟─f772cb78-6939-4919-b4e0-7749029e9932
# ╠═52f940e7-ba00-4bcb-8f6c-7643ce9b4646
# ╠═0f4b36fc-79bb-46e7-89b2-6dcd6a5c58be
# ╠═fd43367b-a33c-4c09-99e1-ecb7a8f1b293
# ╠═308441b8-b485-433f-b233-91de7da3b7e7
# ╟─d20eba0f-6720-49ec-8f8f-e3f37c555f79
# ╠═06241562-1ab8-4a6f-92ce-722a607a6d6d
# ╠═d20b6d40-0d60-4208-9aa1-76a7493418c5
# ╠═88a3ef1f-a1fb-4965-9fc9-0656573897f1
# ╠═296c3373-2cab-49d6-a173-94e4fec021a8
# ╠═3ff338ac-cabb-4cc1-a326-282562901318
# ╠═11d024b2-2a12-4ebe-9c79-96e19028cdf1
# ╠═c18d944a-56b1-45bd-b974-ad08ad44df9c
# ╠═98aebd62-ba14-481b-9c28-97b9abf4bc4f
# ╠═d56c10a8-4feb-4621-a005-0206cfe0f14a
# ╠═2b6a3941-5d03-4b3a-a8e8-1b3ebbc62f20
# ╠═e5318d56-6289-4451-ac03-f5119223990a
# ╠═4942de35-2a00-4419-b959-b2caa90b361b
# ╠═2051d694-9864-4341-ace4-5ad0184c441a
# ╠═1ca3e292-ce0a-4cef-a08b-6d6628bad2c3
# ╠═ce934e7b-b915-4730-92e4-24f31bd1a745
# ╠═a95f7059-f862-4284-bfb2-17a753136e8b
# ╠═02a4fa64-c9c1-4b1d-9ce3-0128317b360a
# ╠═32fefe1e-7ec0-4756-afa8-70f2d0ae9b25
# ╠═6903aa6c-26ca-4c6f-8dcd-45e9bb7a97b8
# ╟─8009fcbf-892f-445b-a12d-3d2470c71a68
# ╠═b7c80bcd-664a-49e5-a9f4-c84164a213be
# ╠═67188719-ebb0-4dc1-8bcd-36ab2f6f1870
# ╟─312fec2e-94c3-4beb-8e29-9a6a417541c3
# ╠═a6bc299e-deb9-4bb7-8661-52657377988a
# ╠═0fa42aa3-88b8-4786-a619-8bea71f6be8d
# ╠═632a6f89-1528-4142-820e-adeb9d41e48e
# ╠═60d7695c-63cc-4757-807a-a1e294a86360
# ╠═601932f0-a0b1-4fdc-801f-e01b08087e43
# ╠═09272854-da18-417c-ab4c-a3db469f4c92
# ╠═c3be7332-7451-4fba-a459-835b36f420f6
# ╠═dc8afbce-17dd-470b-8ab3-5023605537d6
# ╠═721a32bd-1a1a-4f1f-bf1f-ffc09fcde6e2
# ╠═ce68a804-8cd1-4932-8b76-67b5b863ef75
# ╠═6dc2d711-4286-429f-97df-c360d7846e2b
# ╠═1a936087-22f7-427a-829b-8f328e71046d
# ╠═ce186b02-7e6a-4f46-8d5b-548a3689eebe
# ╠═acbfdf1f-dcd8-4d2c-bcfc-ad221a808134
# ╠═cd71d1f6-059a-4b65-ba40-57c02386998b
# ╟─046e4943-61c2-4258-b876-8b898819cd5b
