# Example end-to-end risk assessment.
using Revise
using RiskSimulator
using Random

global SEED = 1000
Random.seed!(SEED)

# Phase 1: Observation model training.
# ————————————————————————————————————————————————
net, net_params = training_phase()


# Phase 2: Failure search.
# ————————————————————————————————————————————————
# First, select an AV policy as the system under test.
system = IntelligentDriverModel(v_des=12.0)

# Setup the AST planner.
planner = setup_ast(net, net_params; sut=system, seed=SEED)

# Run most likely failure search.
action_trace = search!(planner)


# Risk Assessment.
# ————————————————————————————————————————————————
fail_metrics = failure_metrics(planner)
@show fail_metrics

α = 0.2 # Risk tolerance.
𝒟 = planner.mdp.dataset
plot_closure_rate_distribution(𝒟; reuse=false)

# Plot cost distribution.
metrics = risk_assessment(𝒟, α)
@show metrics
risk_plot(metrics; mean_y=0.33, var_y=0.25, cvar_y=0.1, α_y=0.2)


# Playback most likely failure.
# ————————————————————————————————————————————————
false && visualize_most_likely_failure(planner, buildingmap)
