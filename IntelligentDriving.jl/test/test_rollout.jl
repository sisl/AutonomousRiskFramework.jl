include(joinpath(@__DIR__, "header.jl"))

xdim, udim, N = 4, 2, 200

f, fx, fu = randn(xdim, N), randn(xdim, xdim, N), randn(xdim, udim, N)
U = randn(udim, N)
x0 = randn(xdim)
X_prev, U_prev = randn(xdim, N), randn(udim, N)

X1 = ID.rollout1(U, x0, f, fx, fu, X_prev, U_prev)
X2 = ID.rollout2(U, x0, f, fx, fu, X_prev, U_prev)
X3 = ID.rollout3(U, x0, f, fx, fu, X_prev, U_prev)
X4 = ID.rollout4(U, x0, f, fx, fu, X_prev, U_prev)
rollouts = [ID.rollout1, ID.rollout2, ID.rollout3, ID.rollout4]
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
