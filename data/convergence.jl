using FileIO
using ImportanceWeightedRiskMetrics
using Dates
using ProgressMeter

softmax(x) = exp.(x) ./ sum(exp.(x))
weight(logwt) = exp.(logwt)

N_mcts = 100000;
N_baseline = 100000;
alpha = 1e-4;

# SCENARIO
# path = raw"data\new_baseline_risks_21700_ALL.jld2";
# path = raw"data\mcts_IS_10000_0.5_ALL.jld2";

# # SIMPLE
# baseline_path = raw"data\simple_baseline_risks_100000_ALL.jld2";
# mcts_path = raw"data\simple_mcts_IS_10000_0.5_ALL.jld2";

# # INVERTED PENDU
# baseline_path = raw"data\inverted_pendulum_baseline_10000.jld2";
# mcts_path = raw"data\inverted_pendulum_mcts_IS_10000.jld2";

# GRIDWORLD
baseline_path = "data/gridworld_baseline_100000.jld2";
mcts_path = "data/gridworld_mcts_IS_100000.jld2";

baseline_name = baseline_path[6:end-5];
mcts_name = mcts_path[6:end-5];

baseline_data = load(baseline_path);
mcts_data = load(mcts_path);

N_mcts = min(N_mcts, length(mcts_data["risks:"]))
N_baseline = min(N_baseline, length(baseline_data["risks:"]))

logprob = mcts_data["IS_weights:"][1:N_mcts];
mcts_risks = mcts_data["risks:"][1:N_mcts];
baseline_risks = baseline_data["risks:"][1:N_baseline];

logprob =  [logprob[i] for i=1:length(mcts_risks)];
weights = weight(logprob);
mcts_risks =  [risk for risk in mcts_risks];
baseline_risks =  [risk for risk in baseline_risks];



n_mcts_metrics = Dict(
    "mean" => [],
    "var" => [],
    "cvar" => [],
    "worst" => []
);

n_mcts_nowt_metrics = Dict(
    "mean" => [],
    "var" => [],
    "cvar" => [],
    "worst" => []
);

n_baseline_metrics = Dict(
    "mean" => [],
    "var" => [],
    "cvar" => [],
    "worst" => []
);

@showprogress for n=1:5:length(mcts_risks)
    t_mcts_metrics = IWRiskMetrics(mcts_risks[1:n], weight(logprob[1:n]), alpha);
    t_mcts_metrics_nowt = IWRiskMetrics(mcts_risks[1:n], weight(zeros(n)), alpha);
    t_baseline_metrics = IWRiskMetrics(baseline_risks[1:n], weight(zeros(n)), alpha);

    push!(n_mcts_metrics["mean"], t_mcts_metrics.mean)
    push!(n_mcts_metrics["var"], t_mcts_metrics.var)
    push!(n_mcts_metrics["cvar"], t_mcts_metrics.cvar)
    push!(n_mcts_metrics["worst"], t_mcts_metrics.worst)

    push!(n_mcts_nowt_metrics["mean"], t_mcts_metrics_nowt.mean)
    push!(n_mcts_nowt_metrics["var"], t_mcts_metrics_nowt.var)
    push!(n_mcts_nowt_metrics["cvar"], t_mcts_metrics_nowt.cvar)
    push!(n_mcts_nowt_metrics["worst"], t_mcts_metrics_nowt.worst)

    push!(n_baseline_metrics["mean"], t_baseline_metrics.mean)
    push!(n_baseline_metrics["var"], t_baseline_metrics.var)
    push!(n_baseline_metrics["cvar"], t_baseline_metrics.cvar)
    push!(n_baseline_metrics["worst"], t_baseline_metrics.worst)
end

save("data/"*mcts_name*"_alpha_"*string(alpha)*"_Nmetrics.jld2", n_mcts_metrics)
save("data/"*baseline_name*"_alpha_"*string(alpha)*"_Nmetrics.jld2", n_baseline_metrics)
save("data/"*mcts_name*"_alpha_"*string(alpha)*"_unwt_Nmetrics.jld2", n_mcts_nowt_metrics)
