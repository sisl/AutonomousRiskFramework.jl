using PyPlot

function scp_fns_gen(obj_fn::Function; use_SAD::Bool = true, check::Bool = true)
  args_ref = Ref{Tuple}()

  function f_fn_(U, X_prev, U_prev, reg, Ft, ft, params...)
    (udim, N), xdim = size(U_prev), size(X_prev, 1)
    U, X = reshape(U, udim, N), reshape(Ft * reshape(U, :) + ft, xdim, N)
    J = obj_fn(X, U, params...)
    dX, dU = X[:, 1:(end - 1)] - X_prev[:, 2:end], U - U_prev
    J_reg_x = 0.5 * reg[1] * sum(reshape(dX, :) .^ 2)
    J_reg_u = 0.5 * reg[2] * sum(reshape(dU, :) .^ 2)
    return J + J_reg_u + J_reg_x
  end
  f_fn(arg1) = f_fn_(arg1, args_ref[]...)

  # generate dense versions
  g_fn_ = SAD.jacobian_gen(f_fn_; argnums = 1)
  h_fn_ = SAD.hessian_gen(f_fn_; argnums = 1)
  g_fn_dense!(ret, arg1) = (ret[:] = g_fn_(arg1, args_ref[]...)[:]; return)
  h_fn_dense!(ret, arg1) = (ret[:, :] = h_fn_(arg1, args_ref[]...); return)

  # generate sparse versions
  function g_fn_sad!(ret, arg1)
    arg1_ = SAD.Tensor(arg1)
    ret_ = f_fn_(arg1_, args_ref[]...)
    J = SAD.compute_jacobian(ret_, arg1_)
    ret[:] = J[:]
    if check
      J_dense = reshape(g_fn_(arg1, args_ref[]...), size(J)...)
      err = norm(J_dense - J) / max(norm(J_dense), 1e-3)
      @warn @sprintf("J_error = %9.4e", err)
      #@assert norm(J_dense) < 1e-9 || err <= 1e-5
    end
    return
  end
  function h_fn_sad!(ret, arg1)
    arg1_ = SAD.Tensor(arg1)
    ret_ = f_fn_(arg1_, args_ref[]...)
    J, H = SAD.compute_hessian(ret_, arg1_)
    ret[:, :] = H
    if check
      H_dense = reshape(h_fn_(arg1, args_ref[]...), size(H)...)
      err = norm(H_dense - H) / max(norm(H_dense), 1e-3)
      @warn @sprintf("H_error = %9.4e", err)
      #@assert norm(H_dense) < 1e-9 || err <= 1e-5
    end
  end

  # return the appropriate versions
  if !use_SAD
    return f_fn, g_fn_dense!, h_fn_dense!, args_ref
  else
    return f_fn, g_fn_sad!, h_fn_sad!, args_ref
  end
end

function solve_mpc(
  dynamics::Union{String,Tuple{Function,Function,Function,Int,Int}},
  obj_fn::Function,
  x0::Vector{T},
  P_dyn::Union{Vector{T},Matrix{T}}, # parameters of the dynamics e.g. (dt, ...)
  params...; # objective function paramters (X_ref, U_ref, weight_i, ...)
  U::Union{Nothing,AbstractMatrix{T}} = nothing,
  X::Union{Nothing,AbstractMatrix{T}} = nothing,
  N::Int = -1, # number of timesteps in the plan
  max_it::Int = 10, # maximum number of SCP iterations
  reg::Union{Tuple{T,T},Vector{T}} = (1e-1, 1e-1), # SCP regularization
  use_rollout::Bool=false, # whether to use rollout or the linear approx.
  cache::Dict{String,Any} = Dict{String,Any}(), # cache, avoids fn regeneration
  debug::Bool = false, # whether to print debugging info
  debug_plot::Bool = false, # whether to plot the trajectories (for debugging)
) where {T}
  # extract the dynamics ##################################################
  if isa(dynamics, String)
    dyn_f, dyn_fx, dyn_fu = DYNAMICS_FN_MAP[dynamics]
    xdim, udim = DYNAMICS_DIM_MAP[dynamics]
  else
    dyn_f, dyn_fx, dyn_fu, xdim, udim = dynamics
  end

  # initialize (X, U, P) ##################################################
  (size(P_dyn, 2) > 1) && (N = size(P_dyn, 2))
  (U != nothing) && (N = size(U, 2))
  P_dyn = size(P_dyn, 2) == 1 ? repeat(P_dyn, 1, N) : P_dyn
  X = X == nothing ? repeat(x0, 1, N + 1) :
    (size(X, 2) == N + 1 ? X : [X, X[:, end - 1]])
  U = U == nothing ? 1e-3 * randn(udim, N) : U

  # generate or extract the derivative functions ##########################
  if all(haskey(cache, k) for k in ["f_fn", "g_fn!", "h_fn!", "args_ref"])
    f_fn, g_fn!, h_fn!, args_ref =
      (cache[k] for k in ["f_fn", "g_fn!", "h_fn!", "args_ref"])
  else
    f_fn, g_fn!, h_fn!, args_ref =
      scp_fns_gen(obj_fn; check = false, use_SAD = true)
    cache["f_fn"] = f_fn
    cache["g_fn!"] = g_fn!
    cache["h_fn!"] = h_fn!
    cache["args_ref"] = args_ref
  end

  for it in 1:max_it
    X_prev, U_prev = X[:, 1:(end - 1)], U
    f = batch_dynamics(dyn_f, X_prev, U_prev, P_dyn)
    fx = batch_dynamics(dyn_fx, X_prev, U_prev, P_dyn)
    fu = batch_dynamics(dyn_fu, X_prev, U_prev, P_dyn)
    Ft, ft = linearized_dynamics(x0, f, fx, fu, X_prev, U_prev)
    args_ref[] = (X_prev, U_prev, reg, Ft, ft, params...)


    results =
      Optim.optimize(f_fn, g_fn!, h_fn!, reshape(U_prev, :), Optim.Newton())
    U = reshape(results.minimizer, size(U)...)
    if use_rollout
      X = rollout(U, x0, f, fx, fu, X_prev, U_prev)
    else
      X = [x0 reshape(Ft * reshape(U, :) + ft, xdim, N)]
    end

    imprv_U = mean(sqrt.(sum((U - U_prev) .^ 2; dims = 1)))
    imprv_X = mean(sqrt.(sum((X[:, 1:(end - 1)] - X_prev) .^ 2; dims = 1)))
    imprv = imprv_X + imprv_U
    (imprv < 1e-3) && (break)
    (debug) && (@printf("SCP improvement = %9.4e\n", imprv))

    if debug_plot
      figure(32498)
      clf()
      for r in 1:size(U, 1)
        plot(U[r, :])
      end

      figure(435543)
      clf()
      for r in 1:size(X, 1)
        plot(X[r, :])
      end
    end
  end
  return [x0 X], U, cache
end
