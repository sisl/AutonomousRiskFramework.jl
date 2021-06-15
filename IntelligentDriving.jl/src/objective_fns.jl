# examples of objective functions #########################################
# standard MPC reference trajectory following
function track_reference_obj_fn(X, U, params...)
  Q_diag, R_diag, X_ref, U_ref = params
  return quad_prod(X - X_ref, Q_diag) + quad_prod(U - U_ref, R_diag)
end

# standard MPC reference trajectory following and avoidance of others
function track_reference_avoid_others_obj_fn(X, U, params...)
  Q_diag, R_diag, X_ref, U_ref, other_trajs = params

  # avoid other agents in the scene
  dists_s = [(X[1, :] - other_traj[1, :]) .^ 2 for other_traj in other_trajs]
  dists_t = [(X[3, :] - other_traj[3, :]) .^ 2 for other_traj in other_trajs]
  dist_penalty = sum(
    sum(exp.(-dist_s ./ 4.0 - dist_t))
    for (dist_s, dist_t) in zip(dists_s, dists_t)
  )
  return 1e3 * dist_penalty +
         quad_prod(X - X_ref, Q_diag) +
         quad_prod(U - U_ref, R_diag)
end

