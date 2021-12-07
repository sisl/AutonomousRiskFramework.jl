using Pkg
using ScenarioSelection
using FileIO
using Random
using ProgressMeter

N = 100_000
levels = 4

results = []
@showprogress for i=1:N
    push!(results, simple_random_baseline(levels))
end
states = [result[1] for result in results]
risks = [result[2] for result in results]
save("/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/simple_baseline_risks_$(N)_ALL.jld2", Dict("risks:" => risks, "states:" => states))

