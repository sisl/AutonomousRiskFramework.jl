# Takes input array [a, b, c ...]
# Constructs a tree-like MDP with actions 

struct TreeState 
    values::Vector{Any} # multi-level state
    done::Bool
    w::Float64  # Importance sampling weight
end

# initial state constructor
TreeState() = TreeState([], false, 0.0)

# The simple mdp type
mutable struct TreeMDP <: MDP{TreeState, Any}
    rmdp::Any
    discount_factor::Float64 # disocunt factor
    costs::Vector
    IS_weights::Vector
    levels::Int
    max_actions::Int
    N_disturbances::Int
    disturbances::Vector
    probs::Vector
end

function get_actions(mdp::TreeMDP, state::TreeState)
    level = mdp.levels - length(state.values)
    if level > 0
        mult_factor = mdp.max_actions^(level-1)
        bias_factor = reduce(+, [(state.values[i]-1)*mdp.max_actions^(mdp.levels-i) for i=1:length(state.values)], init=0.0)
        last_action = min(mdp.max_actions, ceil((mdp.N_disturbances-bias_factor)/mult_factor))
        return Categorical(Int(last_action))
    else
        return Categorical(1)
    end
end

function construct_tree_rmdp(rmdp, disturbances, probs; max_actions=5)
    N_disturbances = length(disturbances)
    
    levels = ceil(log(N_disturbances)/log(max_actions))

    return TreeMDP(rmdp, 1.0, [], [], levels, max_actions, N_disturbances, disturbances, probs)
end

function eval_reward(mdp::TreeMDP, state::TreeState) 
    idx = Int(reduce(+, [(state.values[i]-1)*mdp.max_actions^(mdp.levels-i) for i=1:mdp.levels], init=0.0)) + 1
    disturbance = mdp.disturbances[idx]
    prob = mdp.probs[idx]
    r = maximum(collect(simulate(HistoryRecorder(), mdp.rmdp, FunctionPolicy((s) -> disturbance))[:r]))
    return r, prob
end

function POMDPs.reward(mdp::TreeMDP, state::TreeState, action)
    if !state.done
        r = 0
    else
        r, prob = eval_reward(mdp, state)
        # print("\nState: ", state, " Reward: ", r)
        push!(mdp.costs, r)
        push!(mdp.IS_weights, state.w - log(prob))
    end
    return r
end

function POMDPs.initialstate(mdp::TreeMDP) # rng unused.
    return TreeState()
end

function POMDPs.gen(m::TreeMDP, s::TreeState, a, rng)
    # transition model
    if length(s.values) < m.levels
        sp = TreeState([s.values..., first(a)], false, last(a))
    elseif length(s.values) == m.levels
        sp = TreeState(s.values, true, last(a))
    else
        print("\nUnexpected state: ", s)
    end
    r = POMDPs.reward(m, sp, a)
    return (sp=sp, r=r)
end


function POMDPs.isterminal(mdp::TreeMDP, s::TreeState)
    return s.done
end

POMDPs.discount(mdp::TreeMDP) = mdp.discount_factor

function POMDPs.actions(mdp::TreeMDP, s::TreeState)
    return get_actions(mdp, s)
end

# function POMDPs.action(policy::RandomPolicy, s::TreeState)
#     return rand(get_actions(s))
# end

function rollout(mdp::TreeMDP, s::TreeState, w::Float64, d::Int64)
    if d == 0 || isterminal(mdp, s)
        return 0.0
    else
        p_action = POMDPs.actions(mdp, s)
        a = rand(p_action)
        
        (sp, r) = @gen(:sp, :r)(mdp, s, [a, w], Random.GLOBAL_RNG)
        q_value = r + discount(mdp)*rollout(mdp, sp, w, d-1)

        return q_value
    end
end