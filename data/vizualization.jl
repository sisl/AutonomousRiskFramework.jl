using Plots
using FileIO
using RiskSimulator
default(show=false, reuse=true)
# data = load(raw"data\\mctsrisks_100_STOPPING.jld2");
data = load("/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/ScenarioSelection.jl/examples/old_mcts_random_IS_risks_1000_ALL.jld2");
states = data["states:"];
risks = data["risks:"];
logprob = data["logprob:"];
risks =  [risk for risk in risks if (risk >= 0)];

# states = [states[i] for i=1:length(states) if risks[i]<0];
# states = states[risks .< 0 ];

# plot([state.init_sut[1] for state in states])
# histogram(risks)
metrics = RiskMetrics(risks, 0.2)
plot_risk(metrics, mean_y=2.5, var_y=2, cvar_y=1, α_y=1.7)
plot!(ylim=(0, 5))

savefig("plot_mcts_1000.png")