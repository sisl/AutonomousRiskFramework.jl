using LinearAlgebra, Printf
using Debugger, BenchmarkTools
#using PyPlot

include(joinpath(@__DIR__, "../src/IntelligentDriving.jl"))
#if !isdefined(Main, :IntelligentDriving)
#  include(joinpath(@__DIR__, "../src/IntelligentDriving.jl"))
#end

#while true
#  try
#    using IntelligentDriving
#    using SpAutoDiff
#
#    break
#  catch
#    using Pkg
#    Pkg.develop(; path=joinpath(@__DIR__, ".."))
#    Pkg.develop(; path=joinpath(@__DIR__, "../../SpAutoDiff.jl"))
#  end
#end

ID = IntelligentDriving
SAD = ID.SAD
