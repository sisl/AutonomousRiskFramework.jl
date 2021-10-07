include("ablation.jl")

function run_ablation_solvers(system)
    seeds = 1:5
    scenario_keys = [CROSSING, T_HEAD_ON, T_LEFT, STOPPING, MERGING, CROSSWALK]
    state_proxy = :none
    adjust_noise = true
    include_rate_reward = true
    use_nnobs = true
    use_learned_rollout = false

    count = 0
    # @sync @distributed
    for scenario_key in scenario_keys
        for which_solver in [:random, :mcts]
            count += 1
            @show count, seeds, scenario_key, typeof(system), which_solver
            @time run_sut(system, scenario_key, seeds, include_rate_reward, use_nnobs, state_proxy, which_solver, adjust_noise, use_learned_rollout)
        end
    end
end

run_ablation_solvers(SUT())
