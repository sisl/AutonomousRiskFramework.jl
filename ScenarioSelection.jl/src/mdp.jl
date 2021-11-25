struct DecisionState 
    type::Any # scenario type
    init_sut::Vector{Any} # Initial conditions SUT
    init_adv::Vector{Any} # Initial conditions Adversary
    done::Bool
    w::Float64  # Importance sampling weight
end

# initial state constructor
DecisionState() = DecisionState(nothing,[nothing],[nothing], false, 1.0)

# Define the system to test
system = IntelligentDriverModel()    

# Evaluates a scenario using AST
# Returns: scalar risk if failures were discovered, 0 if not, -10.0 if an error occured during search

function eval_AST(s::DecisionState)
    try
        scenario = get_scenario(scenario_types[Int64(s.type)]; s_sut=Float64(s.init_sut[1]), s_adv=Float64(s.init_adv[1]), v_sut=Float64(s.init_sut[2]), v_adv=Float64(s.init_adv[2]))
        planner = setup_ast(sut=system, scenario=scenario, nnobs=false, seed=rand(1:100000))
        planner.solver.show_progress = false
        search!(planner)    
        α = 0.2 # risk tolerance
        cvar_wt = [0, 0, 1, 0, 0, 0, 0]  # only compute cvar
        risk = overall_area(planner,weights=cvar_wt, α=α)[1]
        if isnan(risk)
            return 0.0
        end
        return 10*risk
    catch err
        # TODO: Write to log file
        @warn err
        return 0.0
    end
end

# The scenario decision mdp type
mutable struct ScenarioSearch <: MDP{DecisionState, Any}
    discount_factor::Float64 # disocunt factor
    cvars::Vector
    logprob::Vector
end

function POMDPs.reward(mdp::ScenarioSearch, state::DecisionState, action)
    if state.type===nothing || state.init_sut[1]===nothing || state.init_adv[1]===nothing
        r = 0
    else
        r = eval_AST(state)
        push!(mdp.cvars, r)
        push!(mdp.logprob, state.w)
        # r = sum(state.init_cond)
    end
    return r
end

function POMDPs.initialstate(mdp::ScenarioSearch) # rng unused.
    return DecisionState()
end

# Base.convert(::Type{Int64}, x) = x
# convert(::Type{Union{Float64, Nothing}}, x) = x

function POMDPs.gen(m::ScenarioSearch, s::DecisionState, a, rng)
    # transition model
    if s.type === nothing
        sp = DecisionState(first(a), [nothing], [nothing], false, last(a))
    elseif s.init_sut[1] === nothing
        sp =  DecisionState(s.type, first(a), [nothing], false, s.w + last(a))
    elseif s.init_adv[1] === nothing
        sp =  DecisionState(s.type, s.init_sut, first(a), false, s.w + last(a))
    else
        sp = DecisionState(s.type, s.init_sut, s.init_adv, true, s.w)
    end
    r = POMDPs.reward(m, s, a)
    return (sp=sp, r=r)
end

function POMDPs.isterminal(mdp::ScenarioSearch, s::DecisionState)
    return s.done
end

POMDPs.discount(mdp::ScenarioSearch) = mdp.discount_factor

function POMDPs.actions(mdp::ScenarioSearch, s::DecisionState)
    if s.type===nothing
        return get_actions(nothing, nothing)
    elseif s.init_sut[1] === nothing
        return get_actions(:type, s.type)
    elseif s.init_adv[1] === nothing
        return get_actions(:sut, s.type)
    else
        return Distributions.Uniform(0, 1)   # TODO: Replace with a better placeholder
    end
end

function POMDPs.action(policy::RandomPolicy, s::DecisionState)
    if s.type===nothing
        return rand(get_actions(nothing, nothing))
    elseif s.init_sut[1] === nothing
        return rand(get_actions(:type, s.type))
    elseif s.init_adv[1] === nothing
        return rand(get_actions(:sut, s.type))
    else
        return nothing
    end
end

function rollout(mdp::ScenarioSearch, s::DecisionState, d::Int64)
    if d == 0 || isterminal(mdp, s)
        return 0.0
    else
        p_action = POMDPs.actions(mdp, s)
        a = rand(p_action)

        (sp, r) = @gen(:sp, :r)(mdp, s, [a, logpdf(p_action, a)], Random.GLOBAL_RNG)
        q_value = r + discount(mdp)*rollout(mdp, sp, d-1)

        return q_value
    end
end