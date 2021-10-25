using Random
using BayesNets
# first import the POMDPs.jl interface
using POMDPs

# POMDPModelTools has tools that help build the MDP definition
using POMDPModelTools
# POMDPPolicies provides functions to help define simple policies
using POMDPPolicies
# POMDPSimulators provide functions for running MDP simulations
using POMDPSimulators

using CrossEntropyMethod
using Distributions
using Parameters
using MCTS

#####################################################################################
# Bayes Net representation of the scenario decision making problem
#####################################################################################
bn = BayesNet(); 
push!(bn, StaticCPD(:a, Categorical(5)));

function get_actions(parent, value)
    if parent === nothing
		return Categorical(5)
    elseif parent == :a
        # @show parent, value
		return Normal(0, value)
	end
end

push!(bn, CategoricalCPD(:b, [:a], [5], [get_actions(:a, x) for x in 1:5]));

rand(bn)

# #####################################################################################
# # Cross Entropy from Bayes Net
# #####################################################################################
# # starting sampling distribution
# is_dist_0 = Dict{Symbol, Vector{Sampleable}}(:a => [Categorical(5)], :b => [Normal(0, 1)])

# function l(d, s)
#     a = s[:a][1]
#     b = s[:b][1]
#     -(abs(b)>3)
# end

# # Define the likelihood ratio weighting function
# function w(d, s)
#     a = s[:a][1]
#     b = s[:b][1]
#     exp(logpdf(bn, :a=>a, :b=>b) - logpdf(d, s))
# end



#####################################################################################
# MDP definition from Bayes Net
#####################################################################################
struct DecisionState 
    type::Any # scenario type
    pos::Any # Initial position
    done::Bool
end

# initial state constructor
DecisionState() = DecisionState(nothing,nothing, false)

# the scenario decision mdp type
mutable struct ScenarioSearch <: MDP{DecisionState, Union{Int64, Float64, Nothing}}
    discount_factor::Float64 # disocunt factor
end

mdp = ScenarioSearch(1)

function POMDPs.reward(mdp::ScenarioSearch, state::DecisionState, action)
    if state.type===nothing || state.pos===nothing
        r = 0
    else
        r = abs(state.pos)
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
        sp = DecisionState(a, nothing, false)
    elseif s.pos === nothing
        sp =  DecisionState(s.type, a, false)
    else
        sp = DecisionState(s.type, s.pos, true)
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
    elseif s.pos===nothing
        return get_actions(:a, s.type)
    else
        return Normal(0, 1)
    end
end

function POMDPs.action(policy::RandomPolicy, s::DecisionState)
    if s.type===nothing
        return rand(get_actions(nothing, nothing))
    elseif s.pos===nothing
        return rand(get_actions(:a, s.type))
    else
        return nothing
    end
end

function rollout(mdp::ScenarioSearch, s::DecisionState, d::Int64)
    if d == 0 || isterminal(mdp, s)
        return 0.0
    else
        a = rand(POMDPs.actions(mdp, s))

        (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + discount(mdp)*rollout(mdp, sp, d-1)

        return q_value
    end
end

solver = MCTS.DPWSolver(;   estimate_value=rollout, # required.
                            n_iterations=100,
                            enable_state_pw=false, # required.
                            show_progress=true,
                            tree_in_info=true)

planner = solve(solver, mdp)

a = action(planner, DecisionState())

function MCTS.node_tag(s::DecisionState) 
    if s.done
        return "done"
    else
        return "[$(s.type),$(s.pos)]"
    end
end

MCTS.node_tag(a::Union{Int64, Float64, Nothing}) = "[$a]"

using D3Trees

a, info = action_info(planner, DecisionState(), tree_in_info=true)
t = D3Tree(info[:tree], init_expand=1);
inchrome(t)

sim = RolloutSimulator()
simulate(sim, mdp, RandomPolicy(mdp))