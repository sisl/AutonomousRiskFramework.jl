using LinearAlgebra, Printf
using Debugger, BenchmarkTools

include(joinpath(@__DIR__, "../src/IntelligentDriving.jl"))
#using Pkg
#Pkg.develop(; path=joinpath(@__DIR__, "../src/IntelligentDriving.jl"))

ID = IntelligentDriving

function quad_prod(x, Q)
  ret = ID.reduce_sum(
    x .* ID.reduce_sum(Q .* reshape(x, 1, size(x)...); dims = 2);
    dims = 1,
  )
  return size(ret) == () ? ret[] : ret
end

function obj_fn(X, U, params...)
  Q, R, X_ref, U_ref = params
  xdim, N = size(X_ref)
  X = X[:, (end - N + 1):end]
  Jx = sum(quad_prod(X - X_ref, Q))
  Ju = sum(quad_prod(U - U_ref, R))
  J = Jx + Ju
  return J
end

function test()
  xdim, udim, N = 4, 2, 20
  x0 = 0.0 * ones(4)
  X_prev, U_prev = repeat(x0, 1, N), zeros(udim, N)
  U = 1e-3 * randn(udim, N)
  P = repeat(reshape([0.1, 1.0, 1.0], :, 1), 1, N)

  Q = repeat(diagm(0 => [1e0, 1e0, 1e-3, 1e-3]), 1, 1, N)
  R = repeat(diagm(0 => 1e-2 * ones(udim)), 1, 1, N)
  X_ref, U_ref = 2 * ones(xdim, N), zeros(udim, N)
  params = Q, R, X_ref, U_ref

  f_fn, g_fn!, h_fn!, args_ref = ID.scp_fns_gen(obj_fn)

  function test_optimization(X_prev, U_prev)
    for i = 1:10
      X, U = X_prev, U_prev
      reg = [1e-1, 1e-1]
      f = ID.stack(
        map(i -> ID.unicycle_f(X[:, i], U[:, i], P[:, i]), 1:N);
        dims = 2,
      )
      fx = ID.stack(
        map(i -> ID.unicycle_fx(X[:, i], U[:, i], P[:, i]), 1:N);
        dims = 3,
      )
      fu = ID.stack(
        map(i -> ID.unicycle_fu(X[:, i], U[:, i], P[:, i]), 1:N);
        dims = 3,
      )
      Ft, ft = ID.linearized_dynamics(x0, f, fx, fu, X_prev, U_prev)
      args_ref[] = (X_prev, U_prev, reg, Ft, ft, Q, R, X_ref, U_ref)
      results =
        ID.Optim.optimize(f_fn, g_fn!, h_fn!, reshape(U, :), ID.Optim.Newton())
      @assert results.ls_success
      U = reshape(results.minimizer, udim, N)
      X = ID.rollout(U, x0, f, fx, fu, X_prev, U_prev)

      X_prev, U_prev = X[:, 1:(end - 1)], U
    end
  end

  @btime $test_optimization($X_prev, $U_prev)

  return
end

test()
