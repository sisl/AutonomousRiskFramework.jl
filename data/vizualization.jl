using Plots
using FileIO
using RiskSimulator

# data = load(raw"data\\mctsrisks_100_STOPPING.jld2");
data = load(raw"data\\mcts_random_risks_1000_ALL.jld2");
states = data["states:"];
risks = data["risks:"];
risks =  [risk for risk in risks if (risk >= 0)];

# states = [states[i] for i=1:length(states) if risks[i]<0];
# states = states[risks .< 0 ];

# plot([state.init_sut[1] for state in states])
# histogram(risks)
metrics = RiskMetrics(risks, 0.2)
plot_risk(metrics, mean_y=2.5, var_y=2, cvar_y=1, α_y=1.7)
plot!(ylim=(0, 5))