@everywhere begin
    using POMDPs
    using POMDPGym
    using RiskSimulator
    using PyCall
    pyimport("adv_carla")

    include("ast_td3_solver.jl")

    SCENARIO_CLASS_MAPPING = Dict(
        # "Scenario1" => "ControlLoss",
        "Scenario2" => "FollowLeadingVehicle",
        "Scenario3" => "DynamicObjectCrossing",
        "Scenario4" => "VehicleTurningRoute",
        "Scenario5" => "OtherLeadingVehicle",
        # "Scenario6" => "ManeuverOppositeDirection", # NOTE: See "test_scenario6_error.{json/xml}"
        # "Scenario7" => "SignalJunctionCrossingRoute",
        # "Scenario8" => "SignalJunctionCrossingRoute",
        # "Scenario9" => "SignalJunctionCrossingRoute",
        # "Scenario10" => "NoSignalJunctionCrossingRoute",
    )

    function eval_carla_task_core(seed, α, scenario_type, weather)

        sensors = [
            Dict(
                "id" => "GPS",
                "lat" => Dict("mean" => 0, "std" => 0.0001, "upper" => 10, "lower" => -10),
                "lon" => Dict("mean" => 0, "std" => 0.0001, "upper" => 10, "lower" => -10),
                "alt" => Dict("mean" => 0, "std" => 0.00000001, "upper" => 0.0000001, "lower" => 0),
            ),
        ]

        @info "$scenario_type: $(SCENARIO_CLASS_MAPPING[scenario_type])"
        display(weather)

        gym_args = (sensors=sensors, seed=seed, scenario_type=scenario_type, weather=weather, no_rendering=false)
        carla_mdp = GymPOMDP(Symbol("adv-carla"); gym_args...)

        # TODO: Replace with A. Corso TD3 (costs and weights)
        # prior_weights = POLICY_WEIGHTS[s] # IF EXISTS

        costs = run_td3_solver(carla_mdp, sensors) # NOTE: Pass in `prior_weights`
        @show costs
        risk_metrics = RiskMetrics(costs, α)
        cvar = risk_metrics.cvar

        return cvar
    end

end # @everywhere