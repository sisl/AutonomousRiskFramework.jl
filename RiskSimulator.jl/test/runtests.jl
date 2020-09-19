using Test
using RiskSimulator

@testset "Example" begin
    include("example.jl")
    @test sim isa Simulator
end