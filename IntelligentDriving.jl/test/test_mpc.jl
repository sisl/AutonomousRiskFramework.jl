using PyPlot

@time include(joinpath(@__DIR__, "header.jl"))

quad_prod(x, Q_diag) = dot(reshape(Q_diag, :), reshape(x, :) .^ 2)

function obj_fn(X, U, params...)
  Q_diag, R_diag, X_ref, U_ref = params
  xdim, N = size(X_ref)
  X = X[:, (end - N + 1):end]
  Jx = quad_prod(X - X_ref, Q_diag)
  Ju = quad_prod(U - U_ref, R_diag)
  #J_stl =
  #  SAD.until(U[1, :] .- 5.0, U[2, :] .- 1.0, 2; scale = 1e0, outer_max = true)
  #J_stl = SAD.always(-U[1, :]; scale = 1e0)
  return Jx + Ju
end

function test()
  max_it = 10000

  xdim, udim, N = 4, 2, 20
  x0 = [0.0, 0.0, 0.0, pi / 2]
  X_prev, U_prev = repeat(x0, 1, N), zeros(udim, N)
  P_dyn = repeat(reshape([0.1, 1.0, 1.0], :, 1), 1, N)

  Q_diag = repeat([1e0, 1e0, 1e-3, 1e-3], 1, N)
  R_diag = repeat(1e-1 * ones(udim), 1, N)
  X_ref, U_ref = repeat([2, 2, 0.0, 0.0], 1, N), zeros(udim, N)
  params = Q_diag, R_diag, X_ref, U_ref

  f_fn, g_fn!, h_fn!, args_ref = ID.scp_fns_gen(obj_fn; check = false, use_SAD = true)

  X, U = [X_prev x0], U_prev + 1e-2 * randn(size(U_prev))
  reg = [1e0, 1e0]

  X, U, _ = ID.solve_mpc(
    "unicycle",
    obj_fn,
    x0,
    P_dyn,
    params...;
    max_it = max_it,
    reg = reg,
    debug=true,
    use_rollout=true,
    debug_plot=true,
  )
  return
end

test()
