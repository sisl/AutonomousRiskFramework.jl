include("ablation.jl")

function run_end_to_end_study(systems)
    seeds = 1:5
    scenario_keys = [CROSSING, T_HEAD_ON, T_LEFT, STOPPING, MERGING, CROSSWALK]
    # scenario_keys = [CROSSING]
    which_solver = :mcts
    state_proxy = :rate
    adjust_noise = true
    include_rate_reward = true
    use_nnobs = true
    use_learned_rollout = true

    count = 0
    for scenario_key in scenario_keys
        for system in systems
            count += 1
            @show count, seeds, scenario_key, typeof(system), state_proxy, which_solver, use_learned_rollout
            run_sut(system, scenario_key, seeds, include_rate_reward, use_nnobs, state_proxy, which_solver, adjust_noise, use_learned_rollout)
        end
    end
end


run_end_to_end_study([SUT(), SUT2(), SUT3()])
