struct SimpleState 
    levels::Vector{Any} # multi-level state
    done::Bool
    w::Float64  # Importance sampling weight
end

# initial state constructor
SimpleState() = SimpleState([nothing], false, 0.0)

# The simple mdp type
mutable struct SimpleSearch <: MDP{SimpleState, Any}
    discount_factor::Float64 # disocunt factor
    cvars::Vector
    IS_weights::Vector
    levels::Int
    n_actions::Int
end

function transform_rv(x, i::Int; n_actions=10)
    if i < 3
        return (x>n_actions-2 ? 7 : 0)
    else
        return -2*log(x/n_actions)
    end
end

function eval_simple_reward(state::SimpleState; n_actions=10) 
    rewards = [transform_rv(state.levels[i], i; n_actions=n_actions) for i in 1:length(state.levels)]
    reward = (sum(rewards)/(length(rewards)))/7
    # print("\n", state, reward)
    return reward
end

function POMDPs.reward(mdp::SimpleSearch, state::SimpleState, action)
    if !state.done
        r = 0
    else
        r = eval_simple_reward(state; n_actions=mdp.n_actions)
        print("\nState: ", state, " Reward: ", r)
        push!(mdp.cvars, r)
        push!(mdp.IS_weights, state.w)
        # r = sum(state.init_cond)
    end
    return r
end

function POMDPs.initialstate(mdp::SimpleSearch) # rng unused.
    return SimpleState()
end

function POMDPs.gen(m::SimpleSearch, s::SimpleState, a, rng)
    # transition model
    if s.levels[1] === nothing
        sp = SimpleState([first(a)], false, last(a))
    elseif length(s.levels) < m.levels
        sp = SimpleState([s.levels..., first(a)], false, last(a))
    elseif length(s.levels) == m.levels
        sp = SimpleState(s.levels, true, last(a))
    else
        print("\nUnexpected state: ", s)
    end
    r = POMDPs.reward(m, sp, a)
    return (sp=sp, r=r)
end


function POMDPs.isterminal(mdp::SimpleSearch, s::SimpleState)
    return s.done
end

POMDPs.discount(mdp::SimpleSearch) = mdp.discount_factor

function get_actions(s::SimpleState; n_s=10)
    return Distributions.Categorical(n_s)
end

function POMDPs.actions(mdp::SimpleSearch, s::SimpleState)
    return get_actions(s; n_s=mdp.n_actions)
end

# function POMDPs.action(policy::RandomPolicy, s::SimpleState)
#     return rand(get_actions(s))
# end

function rollout(mdp::SimpleSearch, s::SimpleState, w::Float64, d::Int64)
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