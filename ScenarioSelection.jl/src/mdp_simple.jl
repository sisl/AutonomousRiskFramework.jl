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
end

function POMDPs.reward(mdp::SimpleSearch, state::SimpleState, action)
    if !state.done
        r = 0
    else
        r = sum(state.levels)/(length(state.levels)*5)
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
    elseif length(s.levels) < 5
        sp = SimpleState([s.levels..., a], false, 0.0)
    else
        sp = SimpleState(s.levels, true, a)
    end
    r = POMDPs.reward(m, sp, a)
    return (sp=sp, r=r)
end


function POMDPs.isterminal(mdp::SimpleSearch, s::SimpleState)
    return s.done
end

POMDPs.discount(mdp::SimpleSearch) = mdp.discount_factor

function POMDPs.actions(mdp::SimpleSearch, s::SimpleState)
    
    return Distributions.Categorical(5)   # TODO: Replace with a better placeholder
end

function POMDPs.action(policy::RandomPolicy, s::SimpleState)
    return rand(Distributions.Categorical(5))
end

function rollout(mdp::SimpleSearch, s::SimpleState, w::Float64, d::Int64)
    if d == 0 || isterminal(mdp, s)
        return 0.0
    else
        p_action = POMDPs.actions(mdp, s)
        a = rand(p_action)
        if length(s.levels) >= 5
            a = w
        end
        
        (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + discount(mdp)*rollout(mdp, sp, w, d-1)

        return q_value
    end
end