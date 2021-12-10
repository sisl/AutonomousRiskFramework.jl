using Distributions
using Random
using BayesNets
# first import the POMDPs.jl interface
using POMDPs

using AdversarialDriving
# POMDPModelTools has tools that help build the MDP definition
using POMDPModelTools
# POMDPPolicies provides functions to help define simple policies
using POMDPPolicies
# POMDPSimulators provide functions for running MDP simulations
using POMDPSimulators

using AutomotiveSimulator
using AutomotiveVisualization
using POMDPStressTesting
using Reel

using CrossEntropyMethod
using Distributions
using Parameters
using MCTS
using RiskSimulator
using FileIO

using Printf
using Distributed
using ProgressMeter
using D3Trees

Random.seed!(1234)
#####################################################################################
# Bayes Net representation of the scenario decision making problem
#####################################################################################
scenario_types = [T_HEAD_ON, T_LEFT, STOPPING, CROSSING, MERGING, CROSSWALK];
# scenario_types = [STOPPING];


function render_gif(filename::String, mdp::ASTMDP, action_trace::Vector{ASTAction})
    scenes = playback(mdp, action_trace, sim->sim.state, verbose=false, return_trace=true)
    roadway = mdp.sim.roadway

    timestep = 0.5
    nticks = length(scenes)
    animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
        i = Int(floor(t/dt)) + 1
        AutomotiveVisualization.render([roadway, scenes[i]], canvas_height=240)
    end

    write(filename, animation)
end


function get_scenario_options(scenario_enum::SCENARIO)
    if scenario_enum == T_HEAD_ON
        return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    elseif scenario_enum == T_LEFT
        return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    elseif scenario_enum == STOPPING
        return Dict("s_sut" => [4.0, 20.0], "s_adv_del" => [4.0, 10.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0],
                    "s_max" => 30.0, "s_min" => 0.0)
    # elseif scenario_enum == STOPPING
    #     return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    elseif scenario_enum == CROSSING
        return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    elseif scenario_enum == MERGING
        return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    elseif scenario_enum == CROSSWALK
        return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    end
end


function get_actions(parent, value)
    if parent === nothing
        return Distributions.Categorical(length(scenario_types))
    elseif parent == :type
        # @show parent, value
        options = get_scenario_options(scenario_types[value])
        range_s_sut = options["s_sut"]
        range_v_sut = options["v_sut"]
        actions = [
            Distributions.Uniform(range_s_sut[1], range_s_sut[2]),
            Distributions.Uniform(range_v_sut[1], range_v_sut[2]),
        ]
        return product_distribution(actions)
    elseif parent == :sut
        # @show parent, value
        options = get_scenario_options(scenario_types[value])
        s_adv_key = getkey(options, "s_adv_del", "s_adv")
        range_s_adv = options[s_adv_key]
        range_v_adv = options["v_adv"]
        actions = [
            Distributions.Uniform(range_s_adv[1], range_s_adv[2]),
            Distributions.Uniform(range_v_adv[1], range_v_adv[2]),
            Distributions.Uniform(0, 1),
        ]
        return product_distribution(actions)
    end
end

"""
function create_bayesnet()
    bn = BayesNet(); 
    push!(bn, StaticCPD(:type, get_actions(nothing, nothing)));
    push!(bn, CategoricalCPD(:sut, [:type], [length(scenario_types)], [get_actions(:type, x) for x in 1:length(scenario_types)]));
    push!(bn, CategoricalCPD(:adv, [:type], [length(scenario_types)], [get_actions(:sut, x) for x in 1:length(scenario_types)]));
    # @show rand(bn)
    return bn
end

bn = create_bayesnet();
"""

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
# Scenario state and evaluation
#####################################################################################
struct DecisionState
    type::Any # scenario type
    init_sut::Vector{Any} # Initial conditions SUT
    init_adv::Vector{Any} # Initial conditions Adversary
    done::Bool
end

# initial state constructor
DecisionState() = DecisionState(nothing,[nothing],[nothing], false)

# Define the system to test
system = IntelligentDriverModel()

# Evaluates a scenario using AST
# Returns: scalar risk if failures were discovered, 0 if not, -10.0 if an error occured during search

error_logs = open("/Users/kykim/Desktop/Tmp/risks.txt", "w");
error_count = 0
error_states = []

failure_logs = open("/Users/kykim/Desktop/Tmp/failure-states.txt", "w");
failure_count = 0
failure_states = []
failure_mdps = []
failure_traces = []

function eval_AST(s::DecisionState)
    try
        scenario = get_scenario(scenario_types[s.type]; s_sut=Float64(s.init_sut[1]), s_adv=Float64(s.init_adv[1]), v_sut=Float64(s.init_sut[2]), v_adv=Float64(s.init_adv[2]))
        planner = setup_ast(sut=system, scenario=scenario, nnobs=false, seed=rand(1:100000))
        planner.solver.show_progress = false
        action_trace = search!(planner)
        α = 0.2 # risk tolerance
        cvar_wt = [0, 0, 1, 0, 0, 0, 0]  # only compute cvar
        risk = overall_area(planner, weights=cvar_wt, α=α)[1]
        if isnan(risk)
            return 0.0
        end
        if risk > 0.0
            global failure_count += 1
            push!(failure_states, s)
            push!(failure_mdps, planner.mdp)
            push!(failure_traces, action_trace)
            write(failure_logs, string(s, ":", risk, "\n"))
            # filename = joinpath("/Users/kykim/Desktop/Tmp/GIFs", string(index, ".gif"))
            # render_gif(filename, planner.mdp, action_trace);
        end
        return risk / 3.5
    catch err
        write(error_logs, string(s, "\n"))
        push!(error_states, s)
        @warn err
        global error_count += 1
        return 0.0
    end
end

#####################################################################################
# Baseline Evaluation
#####################################################################################

function random_baseline()
    tmp_sample = rand(bn)
    tmp_s = DecisionState(tmp_sample[:type], tmp_sample[:sut], tmp_sample[:adv], true)
    return (tmp_s, eval_AST(tmp_s))
end

if false
    results = []
    for i=1:1000
        push!(results, random_baseline())
    end

    states = [result[1] for result in results];
    risks = [result[2] for result in results];
    save(raw"data\\risks_1000_ALL-2.jld2", Dict("risks:" => risks, "states:" => states))
end

#####################################################################################
# MDP definition from Bayes Net
#####################################################################################

# The scenario decision mdp type
mutable struct ScenarioSearch <: MDP{DecisionState, Any}
    discount_factor::Float64 # disocunt factor
    cvars::Vector
end

mdp = ScenarioSearch(1, [])

function POMDPs.reward(mdp::ScenarioSearch, state::DecisionState, action)
    if state.type===nothing || state.init_sut[1]===nothing || state.init_adv[1]===nothing
        r = 0
    else
        r = eval_AST(state)
        push!(mdp.cvars, r)
    end
    return r
end


function POMDPs.initialstate(mdp::ScenarioSearch) # rng unused.
    return DecisionState()
end


function POMDPs.gen(m::ScenarioSearch, s::DecisionState, a, rng)
    # transition model
    if s.type === nothing
        sp = DecisionState(a, [nothing], [nothing], false)
    elseif s.init_sut[1] === nothing
        sp = DecisionState(s.type, a, [nothing], false)
    elseif s.init_adv[1] === nothing
        d = get_scenario_options(scenario_types[s.type])
        pos_max, pos_min = get(d, "s_max", Inf), get(d, "s_min", -Inf)
        s_pos = s.init_sut[1]
        adv_pos_del, adv_vel, adv_behind = a[1], a[2], a[3]
        if s.type == 3  # STOPPING
            adv_pos = adv_behind > 0.5 ? s_pos - adv_pos_del : s_pos + adv_pos_del
        else
            adv_pos = adv_pos_del
        end
        adv_pos = min(pos_max, max(pos_min, adv_pos))
        sp = DecisionState(s.type, s.init_sut, [adv_pos, adv_vel], false)
    else
        sp = DecisionState(s.type, s.init_sut, s.init_adv, true)
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
        a = rand(POMDPs.actions(mdp, s))

        (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + discount(mdp)*rollout(mdp, sp, d-1)

        return q_value
    end
end


function MCTS.node_tag(s::DecisionState) 
    if s.done
        return "done"
    else
        return "[$(s.type),$(s.init_sut),$(s.init_adv)]"
    end
end


MCTS.node_tag(a::Union{Int64, Float64, Nothing}) = "[$a]"


solver = MCTS.DPWSolver(;   estimate_value=rollout, # required.
                            exploration_constant=0.3,
                            n_iterations=1000,
                            enable_state_pw=false, # required.
                            show_progress=true,
                            tree_in_info=true)


planner = solve(solver, mdp)

a, info = action_info(planner, DecisionState(), tree_in_info=true)
# t = D3Tree(info[:tree], init_expand=1);
# inchrome(t)

println("Error count: ", error_count)
println("Failure count: ", failure_count)
close(failure_logs)
