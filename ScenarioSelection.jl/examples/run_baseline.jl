using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))      # comment if using base package
using ScenarioSelection
using FileIO
using Random
using Distributed
using ProgressMeter

bn = create_bayesnet()

results = []
@showprogress @distributed for i=1:10
    push!(results, random_baseline(bn))
end

states = [result[1] for result in results]
risks = [result[2] for result in results]
# save(raw"data\\risks_1000_ALL.jld2", Dict("risks:" => risks, "states:" => states))