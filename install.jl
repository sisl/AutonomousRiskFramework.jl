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

    # [deps] AutonomousRiskFramework.jl
    PackageSpec(url=joinpath(@__DIR__)),
]

if haskey(ENV, "CI") && ENV["CI"] == "true"
    pop!(packages) # remove "own" package when on CI

    # pytorth does not work with 3.9
    pkg"add Conda"
    using Conda
    Conda.add("python=3.6.5")
end

Pkg.develop(packages)
