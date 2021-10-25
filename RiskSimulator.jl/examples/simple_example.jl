# using RiskSimulator
# system = IntelligentDriverModel()
# scenario = get_scenario(MERGING)
# planner = setup_ast(sut=system, scenario=scenario)
# search!(planner)
# fail_metrics = failure_metrics(planner)
# α = 0.2 # risk tolerance
# risk_metrics = risk_assessment(planner, α)
# risk = overall_area(planner, α=α)


using RiskSimulator

# After MCTS has chosen an action
# Convert action to a scenario (type, initial conditions, disturbance distributions)
system = IntelligentDriverModel()
scenario = scenario_hw_merging_no_obs(s1=6.0, s2=9.0)
planner = setup_ast(sut=system, scenario=scenario, nnobs=false)

# Find risk in the selected scenario
search!(planner)
fail_metrics = failure_metrics(planner)
α = 0.2 # risk tolerance
risk_metrics = risk_assessment(planner, α)
risk = overall_area(planner, α=α)