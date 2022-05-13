using POMDPPolicies
using ProgressMeter
using POMDPSimulators
include("mdp_carla.jl")

mdp = CARLAScenarioMDP(monte_carlo_run=true)
Random.seed!(mdp.seed) # Determinism
s0 = rand(initialstate(mdp))

policy = RandomPolicy(mdp)

N = 1000
results = []
@showprogress for i in 1:N
    global results
    res = POMDPSimulators.simulate(HistoryRecorder(), mdp, policy, s0)
    push!(results, res)
end

results