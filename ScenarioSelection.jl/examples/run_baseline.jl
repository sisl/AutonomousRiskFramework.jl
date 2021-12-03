using Pkg
using ScenarioSelection
using FileIO
using Random
using ProgressMeter

N = 100000

bn = create_bayesnet()

results = []
@showprogress for i=1:N
    push!(results, random_baseline(bn))
    if i%100 == 0
        states = [result[1] for result in results]
        risks = [result[2] for result in results]
        save("/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/new_baseline_risks_$(i)_ALL.jld2", Dict("risks:" => risks, "states:" => states))
    end
end

