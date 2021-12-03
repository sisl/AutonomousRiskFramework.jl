using Plots
using FileIO
using RiskSimulator
using Dates
using D3Trees

softmax(x) = exp.(x) ./ sum(exp.(x))
weight(logwt) = exp.(logwt)

N = 100000;
visualize_tree = false

# SCENARIO
# path = raw"data\new_baseline_risks_21700_ALL.jld2";
# path = raw"data\mcts_IS_10000_0.5_ALL.jld2";

# SIMPLE
path = raw"data\simple_baseline_risks_100000_ALL.jld2";
# path = raw"data\simple_mcts_IS_10000_0.5_ALL.jld2";

name = path[6:end-5];

baseline = false
if occursin("baseline", name)
    baseline = true
end

data = load(path);
if !baseline
    # states = data["states:"][1:N];
    logprob = data["IS_weights:"][1:N];
end
risks = data["risks:"][1:N];

#######################################
# Visualize tree if available
########################################
if !baseline && visualize_tree
    tree = data["tree:"];
    t = D3Tree(tree, init_expand=1);
    inchrome(t)
end


#######################################
# Find valid risk and IS weight values
########################################
if !baseline
    logprob =  [logprob[i] for i=1:length(risks) if (risks[i] >= 0)];
    weights = weight(logprob);
end
risks =  [risk for risk in risks if (risk >= 0)];

#######################################
# Compute and plot metrics
########################################
# states = [states[i] for i=1:length(states) if risks[i]<0];
# states = states[risks .< 0 ];
# if !baseline
#     logprob =  [logprob[i]-exp(risks[i]*0.5)+1 for i=1:length(risks)];
#     weights = softmax(logprob);
# end

# plot([state.init_sut[1] for state in states])
viridis_green = "#238a8dff"
if !baseline
    histogram(risks,
        weights=weights,
        color=viridis_green,
        label=nothing,
        alpha=0.5,
        reuse=false,
        framestyle=:box,
        xlabel="cost",
        ylabel="density",
        bins=100,
        normalize=:pdf, size=(600, 300)
        )
else
    histogram(risks,
        color=viridis_green,
        label=nothing,
        alpha=0.5,
        reuse=false,
        framestyle=:box,
        xlabel="cost",
        ylabel="density",
        bins=100,
        normalize=:pdf, size=(600, 300)
        )
end
plot!(ylim=(0, 15))

if !baseline
    metrics = RiskMetrics(risks, 0.2, weights)
else
    metrics = RiskMetrics(risks, 0.2)
end
plot_risk(metrics, mean_y=0.75, var_y=0.52, cvar_y=0.31, α_y=0.45)
plot!(ylim=(0, 1))
filename =raw"figures\\"*name*"_"*string(Dates.format(now(),"yyyymmddHMS"))*"_$(N)_risks.png"
savefig(filename)

#######################################
# Store convergence data
########################################

n_metrics = Dict(
    "mean" => [],
    "var" => [],
    "cvar" => [],
    "worst" => []
);

for n=1:5:length(risks)
    if !baseline
        metrics = RiskMetrics(risks[1:n], 0.2, weight(logprob[1:n]));
    else
        metrics = RiskMetrics(risks[1:n], 0.2);
    end
    push!(n_metrics["mean"], metrics.mean)
    push!(n_metrics["var"], metrics.var)
    push!(n_metrics["cvar"], metrics.cvar)
    push!(n_metrics["worst"], metrics.worst)
end

plot(n_metrics["mean"])
plot!(ylim=(0, 0.3))

save(raw"data\\"*name*"_Nmetrics.jld2", n_metrics)

#######################################
# Convergence analysis
########################################
# max_N = 1200

# # SCENARIO
# baseline_path = raw"data\new_baseline_risks_21700_ALL_Nmetrics.jld2"
# mcts_path = raw"data\mcts_IS_10000_0.5_ALL_Nmetrics.jld2"
# limits = Dict(
#     :mean => 0.12355874452566123,
#     :var => 0.13887798383139607,
#     :cvar => 0.5994454542009163,
#     :worst => 1.6632798007402125
# )

# SIMPLE
baseline_path = raw"data\simple_baseline_risks_10000_ALL_Nmetrics.jld2"
mcts_path = raw"data\simple_mcts_IS_10000_0.5_ALL_Nmetrics.jld2"
limits = Dict(
    :mean => 0.5994164999999999,
    :var => 0.7,
    :cvar => 0.7647930370075081,
    :worst => 1.0
)

metrics_baseline = load(baseline_path);
metrics_mcts = load(mcts_path);

function plot_trajectory(metric_name, metrics_baseline, metrics_mcts; max_N=nothing, savename=nothing)
    @assert(Symbol(metric_name) in keys(limits))
    if max_N === nothing
        max_N = length(metrics_mcts[metric_name])
    end
    p = plot(ylim=(limits[Symbol(metric_name)]-0.2, limits[Symbol(metric_name)]+0.2))
    plot!(1:5:max_N*5, ones(max_N)*limits[Symbol(metric_name)], label="MC limit (20000 samples)", linestyle=:dash)
    plot!(1:5:max_N*5, metrics_baseline[metric_name][1:max_N], label="Monte Carlo")
    plot!(1:5:max_N*5, metrics_mcts[metric_name][1:max_N], label="MCTS-IS")
    if !(savename === nothing)
        savefig(raw"figures\\"*savename*".png")
    end
    display(p)
end

# Metrics
plot_trajectory("mean", metrics_baseline, metrics_mcts)
plot_trajectory("var", metrics_baseline, metrics_mcts)
plot_trajectory("cvar", metrics_baseline, metrics_mcts)
plot_trajectory("worst", metrics_baseline, metrics_mcts)

