include("ablation.jl")

function run_ablation_rollout(system)
    seeds = 1:5
    scenario_key = STOPPING
    which_solver = :mcts
    state_proxy = :none
    adjust_noise = true
    include_rate_reward = false
    use_nnobs = true

    count = 0
    # @sync @distributed
    for use_learned_rollout in [false, true]
        for state_proxy in [:actual, :distance, :rate]
            if (state_proxy == :actual && use_learned_rollout == false) || use_learned_rollout  # only need to run ONE random rollout (i.e., does not use proxy)
                count += 1
                @show count, seeds, scenario_key, typeof(system), state_proxy, which_solver, use_learned_rollout
                run_sut(system, scenario_key, seeds, include_rate_reward, use_nnobs, state_proxy, which_solver, adjust_noise, use_learned_rollout)
            end
        end
    end
end


run_ablation_rollout(SUT())