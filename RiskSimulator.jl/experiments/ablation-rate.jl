include("ablation.jl")

function run_ablation_rate(system)
    seeds = 1:5
    scenario_key = STOPPING # CROSSWALK?
    scenario = get_scenario(scenario_key)
    which_solver = :mcts
    state_proxy = :none
    adjust_noise = true
    use_learned_rollout = false

    count = 0
    # @sync @distributed 
    for include_rate_reward in [false, true] # parallelize
        for use_nnobs in [false, true] # parallelize
            count += 1
            @show count, seeds, scenario_key, typeof(system), which_solver, include_rate_reward, use_nnobs
            run_sut(system, scenario, seeds, include_rate_reward, use_nnobs, state_proxy, which_solver, adjust_noise, use_learned_rollout)
        end
    end
end


run_ablation_rate(system)