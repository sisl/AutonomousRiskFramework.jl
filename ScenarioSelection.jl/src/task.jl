@everywhere begin
    using POMDPs
    using POMDPGym
    using RiskSimulator
    using PyCall
    pyimport("adv_carla")

    include("solvers.jl")

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

    function eval_carla_task_core(run_solver, seed, scenario_type, weather;
                                  monte_carlo_run=false, use_neat=true, apply_gnss_noise=false,
                                  sensor_config_gnss=nothing, sensor_config_camera=nothing)
        sensors = []

        @info "$scenario_type: $(SCENARIO_CLASS_MAPPING[scenario_type])"
        @info "Seed: $seed"
        display(weather)

        if apply_gnss_noise
            push!(sensors, sensor_config_gnss)
        end

        if use_neat
            push!(sensors, sensor_config_camera)
            agent = joinpath(@__DIR__, "../../CARLAIntegration/neat/leaderboard/team_code/neat_agent.py")
        else
            agent = nothing
        end
        gym_args = (sensors=sensors, seed=seed, scenario_type=scenario_type, weather=weather, no_rendering=false, agent=agent)
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
            cost, dataset = run_solver(carla_mdp, sensors) # NOTE: Pass in `prior_weights`
            @show cost
            @show dataset
            return cost, dataset
        end
    end

end # @everywhere