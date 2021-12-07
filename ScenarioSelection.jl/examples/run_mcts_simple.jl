using ScenarioSelection
using FileIO
using Random
using ProgressMeter
using MCTS

N = 10000
c = 0.5

mdp = SimpleSearch(1, [], [], 5)

planner = mcts_isdpw(mdp; N, c)

a, info = action_info(planner, SimpleState(), tree_in_info=true)

save("/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/simple_mcts_IS_$(N)_$(c)_ALL.jld2", Dict("risks:" => planner.mdp.cvars, "states:" => [], "IS_weights:" => planner.mdp.IS_weights, "tree:" => info[:tree]))