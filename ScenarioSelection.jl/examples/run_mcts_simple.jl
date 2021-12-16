using ScenarioSelection
using FileIO
using Random
using ProgressMeter
using MCTS

N = 10
levels = 5
n_actions = 10
c = 0.5

mdp = SimpleSearch(1, [], [], levels, n_actions)

planner = mcts_isdpw(mdp; N, c)

a, w, info = action_info(planner, SimpleState(), tree_in_info=true)

save("/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/simple_mcts_IS_$(N)_$(c)_ALL.jld2", Dict("risks:" => planner.mdp.cvars, "states:" => [], "IS_weights:" => planner.mdp.IS_weights, "tree:" => info[:tree]))