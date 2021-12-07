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
end

function eval_simple_reward(state::SimpleState) 
    reward_mask = ones(10)*100
    reward_mask[1] = 6
    reward_mask[2] = 7
    reward_mask[4] = 8

    rewards = [((state.levels[i]>reward_mask[i]) ? state.levels[i]*2/20 : state.levels[i]*0.5/20) for i=1:length(state.levels)]
    reward = sum(rewards)/(length(rewards))
    # print("\n", state, reward)
    return reward
end

function POMDPs.reward(mdp::SimpleSearch, state::SimpleState, action)
    if !state.done
        r = 0
    else
        r = eval_simple_reward(state)
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
        sp = SimpleState([a], false, 0.0)
    elseif length(s.levels) < m.levels
        sp = SimpleState([s.levels..., a], false, 0.0)
    elseif length(s.levels) == m.levels
        sp = SimpleState(s.levels, true, a)
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

function get_actions(s::SimpleState)
    return Distributions.Categorical(10)
end

function POMDPs.actions(mdp::SimpleSearch, s::SimpleState)
    return get_actions(s)
end

function POMDPs.action(policy::RandomPolicy, s::SimpleState)
    return rand(get_actions(s))
end

function rollout(mdp::SimpleSearch, s::SimpleState, w::Float64, d::Int64)
    if d == 0 || isterminal(mdp, s)
        return 0.0
    else
        p_action = POMDPs.actions(mdp, s)
        if length(s.levels) == mdp.levels && s.done == false
            # print("\nWeight update: ", w)
            a = w   # Weight update action
        else
            a = rand(p_action)
        end
        
        (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + discount(mdp)*rollout(mdp, sp, w, d-1)

        return q_value
    end
end