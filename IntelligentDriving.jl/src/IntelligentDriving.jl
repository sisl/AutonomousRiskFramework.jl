module IntelligentDriving

using Random, LinearAlgebra, Statistics, Printf
using Zygote, Optim, PyCall, Debugger, Documenter

using AutomotiveSimulator
using AdversarialDriving

#include(joinpath(@__DIR__, "../../SpAutoDiff.jl/src/SpAutoDiff.jl"))
#const SAD = SpAutoDiff
#include(joinpath(@__DIR__, "../../SpAutoDiff.jl/src/naked_SpAutoDiff.jl"))
while true
  try
    using SpAutoDiff
    break
  catch e
    display(e)
    using Pkg
    Pkg.develop(; path=joinpath(@__DIR__, "../../SpAutoDiff.jl"))
  end
end
SAD = SpAutoDiff

include("utils.jl")
include("scp.jl")
include("dynamics.jl")

include("mpc_driver_types.jl")
include("mpc_driver_utils.jl")
include("objective_fns.jl")
include("mpc_driver.jl")

stack = SAD.stack

#export stack, reduce_sum
#export jacobian_gen, hessian_gen
export scp_fns_gen
export rollout, linearized_dynamics
export unicycle_f, unicycle_fx, unicycle_fu

export MPCDriver

end
