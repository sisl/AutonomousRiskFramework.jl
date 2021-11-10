using Plots
using FileIO

# data = load(raw"data\\mctsrisks_100_STOPPING.jld2");
data = load(raw"data\\risks_1000_STOPPING.jld2");
states = data["states:"];
risks = data["risks:"][1:130];
risks =  [risk for risk in risks if (risk !=-10)];
# plot([state.init_sut[1] for state in states])
# histogram(risks)
metrics = RiskMetrics(risks, 0.2)
plot_risk(metrics, mean_y=2.5, var_y=2, cvar_y=1, α_y=1.7)