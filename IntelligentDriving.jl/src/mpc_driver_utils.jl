# low level utility functions ##################################################
quad_prod(x, Q_diag) = dot(reshape(Q_diag, :), reshape(x, :) .^ 2)

function vehicle_state(ent::Entity)
  pf, vf = posf(ent), velf(ent)
  return [pf.s, vf.s, pf.t, vf.t]
end

function simulate_other_agents(vehs, ego, roadway, N, dt)
  x0 = vehicle_state(ego)
  relative_pos = [
    get_frenet_relative_position(other_veh, ego, roadway) for other_veh in vehs
  ]
  other_states = [
    [x0[1] + pos.Δs, velf(veh.state).s, x0[3] + pos.t, velf(veh.state).t]
    for (pos, veh) in zip(relative_pos, vehs)
  ]
  other_trajs =
    [fixed_speed_predict_trajectory(state, N, dt) for state in other_states]
  return other_trajs
end

function fixed_speed_predict_trajectory(x, N, dt)
  x1, x2, x3, x4 = x
  x1 = x[1] .+ (dt * x[2] * (0:(N - 1)))
  x2 = repeat([x2], N)
  x3 = x[3] .+ (dt * x[3] * (0:(N - 1)))
  x4 = repeat([x4], N)
  return vcat(x1', x2', x3', x4')
end

function solve_mpc!(driver::MPCDriver, x0::Vector{Float64}, params...)
  @assert driver.mpc_problem != nothing
  mpc_problem = driver.mpc_problem
  if driver.dynamics == "lon_lane"
    P_dyn = repeat([driver.dt, 1.0, 1e20], 1, driver.N)
  else
    P_dyn = repeat([driver.dt, 1.0, 1.0, 1e20], 1, driver.N)
  end
  X_prev, U_prev = mpc_problem.X, mpc_problem.U
  X, U, cache = solve_mpc(
    driver.dynamics,
    mpc_problem.obj_fn,
    x0,
    P_dyn,
    params...;
    X = X_prev,
    U = U_prev,
    N = driver.N,
    reg = (1e-6, 1e-6), # dynamics are linear, this is numerical regularization
    max_it = 1,
    use_rollout = false,
    debug_plot = false,
  )
  mpc_problem.cache = cache
  mpc_problem.X = X
  mpc_problem.U = U
  return X, U
end
