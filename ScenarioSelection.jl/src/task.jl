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

    function eval_carla_task_core(seed, α, scenario_type, weather; monte_carlo_run=false)

        sensors = [
            Dict(
                "id" => "GPS",
                "lat" => Dict("mean" => 0, "std" => 0.0001, "upper" => 0.000000001, "lower" => -0.000000001),
                "lon" => Dict("mean" => 0, "std" => 0.0001, "upper" => 0.000000001, "lower" => -0.000000001),
                "alt" => Dict("mean" => 0, "std" => 0.00000001, "upper" => 0.0000001, "lower" => 0),
            ),
            Dict(
                "id" => "rgb",
                "dynamic_noise_std" => Dict("mean" => 0, "std" => 0.001, "upper" => 1, "lower" => 0),
                "exposure_compensation" => Dict("mean" => 0, "std" => 0.5, "upper" => 1, "lower" => -1),
            ),
        ]

        @info "$scenario_type: $(SCENARIO_CLASS_MAPPING[scenario_type])"
        @info "Seed: $seed"
        display(weather)

        use_neat = true
        if use_neat
            agent = joinpath(@__DIR__, "../../CARLAIntegration/neat/leaderboard/team_code/neat_agent.py")
            gym_args = (sensors=sensors, seed=seed, scenario_type=scenario_type, weather=weather, no_rendering=false, agent=agent)
        else
            deleteat!(sensors, 2)
            gym_args = (sensors=sensors, seed=seed, scenario_type=scenario_type, weather=weather, no_rendering=false)
        end
        carla_mdp = GymPOMDP(Symbol("adv-carla"); gym_args...)

        # TODO: Replace with A. Corso TD3 (costs and weights)
        # prior_weights = POLICY_WEIGHTS[s] # IF EXISTS

        if monte_carlo_run
            env = carla_mdp.env
            action_dim = sum([length(filter(!=("id"), keys(sensor))) for sensor in sensors])
            # local info
            dataset = missing
            info = Dict("cost"=>0)
            action = zeros(action_dim) # TODO: Skip all the adversarial sensors to begin with (instead of just zeroing out actions)
            while !env.done
                reward, obs, info = POMDPGym.step!(env, action)
            end
            close(env)
            cost = info["cost"]
            return cost, dataset
        else
            costs, dataset = run_td3_solver(carla_mdp, sensors) # NOTE: Pass in `prior_weights`
            @show costs
            @show dataset
            risk_metrics = RiskMetrics(costs, α)
            cvar = risk_metrics.cvar

            return cvar, dataset
        end
    end

end # @everywhere