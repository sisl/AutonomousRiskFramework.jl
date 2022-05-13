using Pkg

packages = [
    # [deps] ObservationModel.jl
    PackageSpec(url="https://github.com/sisl/AdversarialDriving.jl"),
    PackageSpec(url="https://github.com/sisl/Vec.jl"),
    PackageSpec(url=joinpath(@__DIR__, "ObservationModels.jl")),

    # [deps] STLCG.jl
    PackageSpec(url=joinpath(@__DIR__, "STLCG.jl")),

    # [deps] IntelligentDriving.jl
    PackageSpec(url="https://github.com/rdyro/SpAutoDiff.jl"),
    PackageSpec(url=joinpath(@__DIR__, "IntelligentDriving.jl")),

    # [deps] RiskSimulator.jl
    PackageSpec(url=joinpath(@__DIR__, "RiskSimulator.jl")),

    # [deps] ScenarioSelection.jl
    PackageSpec(url=joinpath(@__DIR__, "ScenarioSelection.jl")),
    PackageSpec(url="https://github.com/shubhg1996/ImportanceWeightedRiskMetrics.jl"),
    PackageSpec(url="https://github.com/shubhg1996/TreeImportanceSampling.jl"),

    # [deps] AutonomousRiskFramework.jl
    PackageSpec(url="https://github.com/ancorso/POMDPGym.jl"),
    PackageSpec(url="https://github.com/ancorso/Crux.jl"),
    PackageSpec(url=joinpath(@__DIR__)),
]

ci = haskey(ENV, "CI") && ENV["CI"] == "true"

if ci
    # remove "own" package when on CI
    pop!(packages)
end

# Run dev altogether
# This is important that it's run together so there
# are no "expected pacakge X to be registered" errors.
Pkg.develop(packages)

if ci
    # pytorch does not work with 3.9
    pkg"add Conda"
    using Conda
    Conda.add("python=3.7.5")
end

Pkg.add("MCTS")
Pkg.add("D3Trees")
Pkg.add("BSON")
Pkg.add("Distributions")
Pkg.add("Infiltrator")
Pkg.add("POMDPModelTools")
Pkg.add("POMDPPolicies")
Pkg.add("POMDPs")
Pkg.add("Parameters")
Pkg.add("PyCall")
Pkg.add("Flux")
