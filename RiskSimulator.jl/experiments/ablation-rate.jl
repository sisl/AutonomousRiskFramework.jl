include("ablation.jl")

function run_ablation_rate(system)
    seeds = 1:5
    scenario_key = CROSSWALK
    which_solver = :mcts
    state_proxy = :none
    adjust_noise = true
    use_learned_rollout = false
    use_nnobs = true

    count = 0
    # @sync @distributed 
    for include_rate_reward in [false, true] # parallelize
        count += 1
        @show count, seeds, scenario_key, typeof(system), include_rate_reward
        run_sut(system, scenario_key, seeds, include_rate_reward, use_nnobs, state_proxy, which_solver, adjust_noise, use_learned_rollout)
    end
end


run_ablation_rate(SUT())