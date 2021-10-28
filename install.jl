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

Pkg.develop(packages)
