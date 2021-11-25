using Pkg
using ScenarioSelection
using FileIO
using Random
using ProgressMeter

N = 10000

bn = create_bayesnet()

results = []
@showprogress for i=1:N
    push!(results, random_baseline(bn))
end

states = [result[1] for result in results]
risks = [result[2] for result in results]
save("/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/baseline_risks_$(N)_ALL.jld2", Dict("risks:" => risks, "states:" => states))
