using LinearAlgebra, Printf
using Zygote, BenchmarkTools

include(joinpath(@__DIR__, "../src/dyn.jl"))

xdim, udim, N = 4, 2, 200

f, fx, fu = randn(xdim, N), randn(xdim, xdim, N), randn(xdim, udim, N)
U = randn(udim, N)
x0 = randn(xdim)
X_prev, U_prev = randn(xdim, N), randn(udim, N)

X1 = rollout1(U, x0, f, fx, fu, X_prev, U_prev)
X2 = rollout2(U, x0, f, fx, fu, X_prev, U_prev)
X3 = rollout3(U, x0, f, fx, fu, X_prev, U_prev)
X4 = rollout4(U, x0, f, fx, fu, X_prev, U_prev)
rollouts = [rollout1, rollout2, rollout3, rollout4]
for (i, rollout_) in enumerate(rollouts)
  @printf("rollout%d ---------------------------------\n", i)
  try
    @btime X = $rollout_(U, x0, f, fx, fu, X_prev, U_prev)
    #g = Zygote.gradient(
    #  U -> rollout_(U, x0, f, fx, fu, X_prev, U_prev)[end, end],
    #  U,
    #)[1]
    #@btime g = Zygote.gradient(
    #  U -> $rollout_(U, x0, f, fx, fu, X_prev, U_prev)[end, end],
    #  U,
    #)[1]
  catch e
    println(e)
    @printf("rollout%d doesn't support gradients\n", i)
  end
end

return
