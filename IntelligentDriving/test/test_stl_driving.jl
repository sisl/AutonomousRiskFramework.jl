include(joinpath(@__DIR__, "header.jl"))
using PyPlot
using STLCG

# always less than
function alt(pos, val)
  signal = pos[1, :]
  phi = Always(; subformula = LessThan(:r, val), interval = nothing)
  scale = -1
  rho = STLCG.ρ(phi, reshape(reverse(signal), 1, :, 1); scale = scale)[1]
  return rho
end

function quad_prod(x, Q)
  ret = ID.reduce_sum(
    x .* ID.reduce_sum(Q .* reshape(x, 1, size(x)...); dims = 2);
    dims = 1,
  )
  return size(ret) == () ? ret[] : ret
end

function softmax2(x::AbstractArray{T, 1}) where T
  x_max = maximum(x)
  z = x .- x_max
  expz = exp.(z)
  return expz ./ sum(expz)
end

function obj_fn(X, U, params...)
  Q, R, X_ref, U_ref = params
  xdim, N = size(X_ref)
  X = X[:, (end - N + 1):end]
  Jx = sum(quad_prod(X - X_ref, Q))
  Ju = sum(quad_prod(U - U_ref, R))
  #J = Jx + Ju - 1e2 * min(alt(X, 1.0), 0.0)
  z = [alt(X, 1.0), 0.0]
  J = Jx + Ju - 1e2 * (z .* softmax2(-1e1 * z))[1]
  return J
end

function test()
  xdim, udim, N = 4, 2, 20
  x0 = 0.0 * ones(4)
  X_prev, U_prev = repeat(x0, 1, N), zeros(udim, N)
  U = 1e-3 * randn(udim, N)
  P = repeat(reshape([0.1, 1.0, 1.0], :, 1), 1, N)

  Q = repeat(diagm(0 => [1e0, 1e0, 1e-3, 1e-3]), 1, 1, N)
  R = repeat(diagm(0 => 1e-1 * ones(udim)), 1, 1, N)
  X_ref, U_ref = 4 * ones(xdim, N), zeros(udim, N)
  params = Q, R, X_ref, U_ref

  f_fn, g_fn!, h_fn!, args_ref = ID.scp_fns_gen(obj_fn)

  function test_optimization(X_prev, U_prev)
    for i = 1:10
      X, U = X_prev, U_prev
      reg = 1e-1 .* [1.0, 1.0]
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
      results = ID.Optim.optimize(
        f_fn,
        g_fn!,
        h_fn!,
        reshape(U, :),
        ID.Optim.Newton(),
        ID.Optim.Options(;
          g_tol = 1e-3,
          allow_f_increases = true,
          time_limit = 3.0,
          show_trace=true,
          show_every=1,
        ),
      )
      #@assert results.ls_success
      println(results.ls_success)
      #display(results)
      U = reshape(results.minimizer, udim, N)
      X = ID.rollout(U, x0, f, fx, fu, X_prev, U_prev)

      err =
        mean(map(i -> norm(X[:, i] - X_prev[:, i]), 1:N)) +
        mean(map(i -> norm(U[:, i] - U_prev[:, i]), 1:N))
      (err < 1e-4) && (break)
      println(err)
      X_prev, U_prev = X[:, 1:(end - 1)], U
    end
    return X, U
  end

  #@btime $test_optimization($X_prev, $U_prev)
  X, U = test_optimization(X_prev, U_prev)

  return X, U
end

for r in range(0.0; stop = 2.0, length = 10)
  println((r, alt(r * ones(3, 10), 1.0)))
end

X, U = test()
display(alt(X, 1.0))

figure()
for r = 1:size(X, 1)
  plot(X[r, :]; label = @sprintf("x%d", r))
end
legend()

figure()
for r = 1:size(U, 1)
  plot(U[r, :]; label = @sprintf("u%d", r))
end
legend()
