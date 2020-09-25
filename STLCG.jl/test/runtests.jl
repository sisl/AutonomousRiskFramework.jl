using Test
using STLCG

@testset "Example STL formulas (old method)" begin
    # Examples:
    signals = 0:10
    times = 0:0.5:5 # non-integer time.

    @test always(s->s < 11, signals) == true

    @test □([0, 5], s->s < 6, signals, times) == false
    @test □([0, 5], s->s < 6, signals) == true # assumes linear time

    @test ◊([0, 3], s->s == 6, signals, times) == true
end
