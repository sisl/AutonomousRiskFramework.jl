##^# imports ###################################################################
include("diff.jl")
##$#############################################################################
##^# scp generate functions ####################################################
function scp_fns_gen(obj_fn::Function)
  args_ref = Ref{Tuple}()

  function f_fn(U, X_prev, U_prev, reg, Ft, ft, params...)
    udim, N = size(U_prev)
    xdim = size(X_prev, 1)

    U = reshape(U, udim, N)
    X = reshape(Ft * reshape(U, :) + ft, xdim, N)
    J = obj_fn(X, U, params...)
    reg_x, reg_u = reg
    dX, dU = X[:, 1:(end - 1)] - X_prev[:, 2:end], U - U_prev
    J_reg_x = 0.5 * reg_x * dot(reshape(dX, :), reshape(dX, :))
    J_reg_u = 0.5 * reg_u * dot(reshape(dU, :), reshape(dU, :))

    return J + J_reg_x + J_reg_u
  end
  g_fn, h_fn = jacobian_gen(f_fn; argnums = 1), hessian_gen(f_fn; argnums = 1)

  function f_fn_(arg1)
    return f_fn(arg1, args_ref[]...)
  end
  function g_fn!(ret, arg1)
    ret[:] = g_fn(arg1, args_ref[]...)[:]
    return
  end
  function h_fn!(ret, arg1)
    ret[:, :] = h_fn(arg1, args_ref[]...)
  end

  return f_fn_, g_fn!, h_fn!, args_ref
end
##$#############################################################################
