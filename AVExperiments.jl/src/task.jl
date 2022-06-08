using POMDPs
using POMDPGym
using PyCall
pyimport("adv_carla")

# include("solvers.jl")
# include("agents.jl")

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

function eval_carla_task_core(run_solver, seed, scenario_type, other_actor_type, weather;
                              leaf_noise=true, agent=NEAT, apply_gnss_noise=false,
                              sensor_config_gnss=nothing, sensor_config_camera=nothing,
                              no_rendering=false)
    sensors = []

    @info "$scenario_type: $(SCENARIO_CLASS_MAPPING[scenario_type])"
    @info other_actor_type
    @info "Seed: $seed"
    display(weather)

    if apply_gnss_noise
        push!(sensors, sensor_config_gnss)
    end

    if agent == NEAT
        push!(sensors, sensor_config_camera)
        agent_path = joinpath(@__DIR__, "../../CARLAIntegration/neat/leaderboard/team_code/neat_agent.py")
        agent_config = joinpath(@__DIR__, "../../CARLAIntegration/neat/model_ckpt/neat")
    elseif agent == WorldOnRails
        push!(sensors, sensor_config_camera)
        agent_path = joinpath(@__DIR__, "../../CARLAIntegration/WorldOnRails/autoagents/image_agent.py")
        agent_config = joinpath(@__DIR__, "../../CARLAIntegration/WorldOnRails/config.yaml")
    elseif agent == GNSS
        agent_path = nothing
        agent_config = nothing
    end

    gym_args = (sensors=sensors, seed=seed, scenario_type=scenario_type, other_actor_type=other_actor_type, weather=weather, no_rendering=no_rendering, agent=agent_path, agent_config=agent_config)
    carla_mdp = GymPOMDP(Symbol("adv-carla"); gym_args...)

    if leaf_noise
        cost, dataset = run_solver(carla_mdp, sensors)
    else
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
    end
    @show cost
    @show dataset
    return cost, dataset
end
