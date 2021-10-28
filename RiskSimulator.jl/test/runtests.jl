using Test
using RiskSimulator

@testset "Simple example" begin
    include("../examples/simple_example.jl")
    @test true
end

@testset "AV policies example" begin
    for system in [IntelligentDriverModel(), PrincetonDriver(), MPCDriver()]
        @info "Testing $(typeof(system))"
        system = IntelligentDriverModel()
        scenario = get_scenario(MERGING)
        planner = setup_ast(sut=system, scenario=scenario)
        search!(planner)
        @test true
    end
end
