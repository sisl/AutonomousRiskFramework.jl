include("ablation.jl")

function run_ablation_rollout(system)
    seeds = 1:5
    scenario_key = STOPPING # TODO?
    scenario = get_scenario(scenario_key)
    which_solver = :mcts
    state_proxy = :none
    adjust_noise = true
    include_rate_reward = false

    count = 0
    # @sync @distributed
    for use_nnobs in [false, true] # parallelize
        for state_proxy in [:distance, :rate, :actual]
                for use_learned_rollout in [false, true]
                count += 1
                @show count, seeds, scenario_key, typeof(system), state_proxy, which_solver, use_learned_rollout, use_nnobs
                run_sut(system, scenario, seeds, include_rate_reward, use_nnobs, state_proxy, which_solver, adjust_noise, use_learned_rollout)
            end
        end
    end
end


run_ablation_rollout(system)