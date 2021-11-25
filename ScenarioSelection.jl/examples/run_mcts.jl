using ScenarioSelection
using FileIO
using Random
using Distributed
using ProgressMeter
# using D3Trees
using MCTS

N = 10
c = 0.5

mdp = ScenarioSearch(1, [], [])

planner = mcts_isdpw(mdp; N, c)

a, info = action_info(planner, DecisionState(), tree_in_info=true)
# t = D3Tree(info[:tree], init_expand=1);
# inchrome(t)

save("/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/mcts_IS_Lrisks_$(N)_$(c)_ALL.jld2", Dict("risks:" => planner.mdp.cvars, "states:" => [], "logprob:" => planner.mdp.logprob))
