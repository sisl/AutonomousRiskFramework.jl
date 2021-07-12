using FileIO
using Dates
using POMDPStressTesting

include("visualization.jl")
include("risk_metrics.jl")

path = raw"variables\record_2021_07_11_183731.jld2"
name = path[end-21:end-5]

tmp = load(path)
fail_metrics = POMDPStressTesting.print_metrics(tmp["metrics"]) 
is_dist_opt = tmp["is_dist_opt"]

α = 0.2 # Risk tolerance.
𝒟 = tmp["dataset"];
filter!(x->x[1][end]≠0,𝒟);

p_closure = plot_closure_rate_distribution(𝒟; reuse=false)
savefig(p_closure, raw"plots\closure_rate_distribution_"*name*".png")

p_distance = plot_miss_distance_distribution(𝒟; reuse=false)
savefig(p_distance, raw"plots\miss_distance_distribution_"*name*".png")

# Plot cost distribution.
metrics = risk_assessment(𝒟, α)
p_risk = plot_risk(metrics; mean_y=2.33, var_y=4.25, cvar_y=2.1, α_y=6.2)
savefig(p_risk, raw"plots\risk_"*name*".png")

# Polar plot of risk and failure metrics
w = ones(7);
p_metrics = plot_polar_risk([𝒟], [tmp["metrics"]], ["Carla BaseAgent"]; weights=w, α=α)
savefig(raw"plots\polar_risk_"*name*".png")

