using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))      # comment if using base package
using ScenarioSelection
using FileIO
using Random
using Distributed
using ProgressMeter
using D3Trees
using MCTS

mdp = ScenarioSearch(1, [])

planner = mcts_vanilla(mdp)

a, info = action_info(planner, DecisionState(), tree_in_info=true; random=true)
t = D3Tree(info[:tree], init_expand=1);
inchrome(t)

# save(raw"data\\mctsrisks_100_ALL.jld2", Dict("risks:" => planner.mdp.cvars, "states:" => []))
