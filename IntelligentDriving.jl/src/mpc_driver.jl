# AutomotiveSimulator functions ################################################
function AutomotiveSimulator.set_desired_speed!(
  model::MPCDriver,
  v_des::Float64,
)
  model.v_ref = v_des
  return model
end

function AutomotiveSimulator.observe!(
  driver::MPCDriver,
  scene::Scene{Entity{S,D,I}},
  roadway::Roadway,
  egoid::I,
) where {S,D,I}
  vehicle_index = findfirst(egoid, scene)
  ego = scene[vehicle_index]
  x0 = vehicle_state(ego)
  (driver.v_ref == nothing) && (driver.v_ref = x0[2]) # v0 as reference
  vehs = [veh for veh in scene if veh.id != egoid]

  other_trajs = simulate_other_agents(vehs, ego, roadway, driver.N, driver.dt)

  # assemble objective function parameters:
  #
  # a reference trajectory is such a standard MPC formulation that the driver
  # should probably assemble these values and pass them to an objective function
  #
  # this is just an examaple
  X_ref = fixed_speed_predict_trajectory(
    [x0[1], driver.v_ref, 0, 0],
    driver.N,
    driver.dt,
  )
  U_ref = zeros(2, driver.N)
  Q_diag = repeat([1e-5, 3e-1, 3e-1, 1e-1], 1, driver.N)
  R_diag = repeat(1e-1 * ones(2), 1, driver.N)

  if driver.dynamics == "lon_lane"
    x0, X_ref, U_ref = x0[1:2], X_ref[1:2, :], U_ref[1:1, :]
    Q_diag, R_diag = Q_diag[1:2, :], R_diag[1:1, :]
  end
  params = Q_diag, R_diag, X_ref, U_ref, other_trajs

  # compute the MPC plan and extract control
  X, U = solve_mpc!(driver, x0, params...)
  u0 = U[:, 1]
  driver.a_lon = u0[1]
  driver.a_lat = length(u0) == 2 ? u0[2] : 0.0 # allow longitudinal dynamics only

  # append history of states for potential system ID
  (length(driver.u_hist) >= 10) && (deleteat!(driver.u_hist, 1))
  (length(driver.x_hist) >= 10) && (deleteat!(driver.x_hist, 1))
  push!(driver.u_hist, u0)
  push!(driver.x_hist, x0)

  driver
end

function AutomotiveSimulator.track_longitudinal!(
  driver::MPCDriver,
  v_ego::Float64,
  v_oth::Float64,
  headway::Float64,
)
  x0 = [0.0, v_ego, 0.0, 0.0] # assume we're in the center of the lane
  x_other = [x0[1] + headway, v_oth, 0.0, 0.0]
  other_trajs = [fixed_speed_predict_trajectory(x_other, driver.N, driver.dt)]
  X_ref = fixed_speed_predict_trajectory(
    [x0[1], driver.v_ref, 0, 0],
    driver.N,
    driver.dt,
  )
  U_ref = zeros(2, driver.N)
  Q_diag = repeat([1e-5, 3e-1, 1e0, 1e0], 1, driver.N)
  R_diag = repeat(1e-2 * ones(2), 1, driver.N)

  if driver.dynamics == "lon_lane"
    x0, X_ref, U_ref = x0[1:2], X_ref[1:2, :], U_ref[1:1, :]
    Q_diag, R_diag = Q_diag[1:2, :], R_diag[1:1, :]
  end
  params = Q_diag, R_diag, X_ref, U_ref, other_trajs

  # compute the MPC plan and extract control
  X, U = solve_mpc!(driver, x0, params...)
  u0 = U[:, 1]
  driver.a_lon = u0[1]
  #driver.a_lat = 0.0 # do not affect lateral velocity
  
  return driver
end
# overriding functions required by AutomotiveSimulator #########################
Base.rand(rng::AbstractRNG, m::MPCDriver) = LatLonAccel(m.a_lat, m.a_lon)

#Distributions.pdf(driver::MPCDriver, a::LatLonAccel) =
#  pdf(driver.mlat, a.a_lat) * pdf(driver.mlon, a.a_lon)
#
#Distributions.logpdf(driver::MPCDriver, a::LatLonAccel) =
#  logpdf(driver.mlat, a.a_lat) * logpdf(driver.mlon, a.a_lon)


# Computes the time it takes to cover a given distance, assuming the current acceleration of the provided idm
function AdversarialDriving.time_to_cross_distance_const_acc(veh::Entity, mpc::MPCDriver, ds::Float64)
    v = vel(veh)
    d = v^2 + 2*mpc.a_lon*ds
    d < 0 && return 0 # We have already passed the point we are trying to get to
    if isnothing(mpc.v_ref)
        vf = sqrt(d)
    else
        vf = min(mpc.v_ref, sqrt(d))
    end
    2*ds/(vf + v)
end

function Base.rand(rng::AbstractRNG, model, mpc::MPCDriver)
    na = model.next_action
    BlinkerVehicleControl(mpc.a_lon, na.da, na.toggle_goal, na.toggle_blinker, na.noise)
end
