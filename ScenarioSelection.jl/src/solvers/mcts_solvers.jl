function mcts_dpw(mdp; N=10, c=1.0)
    solver = MCTS.DPWSolver(;   estimate_value=rollout, # required.
                            exploration_constant=c,
                            n_iterations=N,
                            enable_state_pw=false, # required.
                            show_progress=true,
                            tree_in_info=true);

    planner = solve(solver, mdp);
    return planner
end

function mcts_isdpw(mdp; N=10, c=1.0)
    solver = ISDPWSolver(;  depth=100, 
                            estimate_value=rollout, # required.
                            exploration_constant=c,
                            n_iterations=N,
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

function MCTS.node_tag(s::SimpleState) 
    if s.done
        return "done"
    else
        return "$(s.levels)"
    end
end

MCTS.node_tag(a::Union{Int64, Float64, Nothing}) = "[$a]"
