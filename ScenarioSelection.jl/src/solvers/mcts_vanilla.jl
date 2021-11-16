function mcts_vanilla(mdp)
    solver = MCTS.DPWSolver(;   estimate_value=rollout, # required.
                            exploration_constant=1.0,
                            n_iterations=1000,
                            enable_state_pw=false, # required.
                            show_progress=true,
                            tree_in_info=true);

    planner = solve(solver, mdp);
    return planner
end

function MCTS.node_tag(s::DecisionState) 
    if s.done
        return "done"
    else
        return "[$(s.type),$(s.init_sut),$(s.init_adv)]"
    end
end

MCTS.node_tag(a::Union{Int64, Float64, Nothing}) = "[$a]"
