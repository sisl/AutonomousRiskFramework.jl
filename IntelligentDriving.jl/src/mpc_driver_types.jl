# mpc problem definition #######################################################
mutable struct MPCProblem
  obj_fn::Union{Function,Nothing}
  cache::Dict{String,Any}
  X::Any
  U::Any
end
MPCProblem() = MPCProblem(nothing, Dict{String,Any}(), nothing, nothing)

MPCProblem(obj_fn::Function) =
  MPCProblem(obj_fn, Dict{String,Any}(), nothing, nothing)

# driver type definition #######################################################
"""
	MPCDriver
Driver that combines longitudinal driver and lateral driver into one model.

# Constructors

`MPCDriver(...)`
The keywords argument are the fields described below.

# Fields
- `a_lat::Float64 = lateral acceleration
- `a_lon::Float64 = longitudinal acceleration
"""
mutable struct MPCDriver <: DriverModel{LatLonAccel}
  obj_fn::Union{Nothing,Function}
  dynamics::String
  a_lat::Float64
  a_lon::Float64
  v_ref::Union{Nothing,Float64}
  N::Int
  dt::Float64
  mpc_problem::Union{Nothing,MPCProblem}
  x_hist::Vector{Vector{Float64}}
  u_hist::Vector{Vector{Float64}}
end
function MPCDriver()
  obj_fn = track_reference_avoid_others_obj_fn
  driver = MPCDriver(obj_fn, "lane", 0, 0, nothing, 10, 0.1, nothing, [], [])
  driver.mpc_problem = MPCProblem(obj_fn)
  return driver
end
function MPCDriver(obj_fn, dynamics)
  driver = MPCDriver(obj_fn, dynamics, 0, 0, nothing, 10, 0.1, nothing, [], [])
  driver.mpc_problem = MPCProblem(obj_fn)
  return driver
end
function MPCDriver(obj_fn, dynamics, N, dt)
  driver = MPCDriver(obj_fn, dynamics, 0, 0, nothing, N, dt, nothing, [], [])
  driver.mpc_problem = MPCProblem(obj_fn)
  return driver
end

function setproperty(driver::MPCDriver, s::Symbol, fn::Function)
  (s == :obj_fn) && (setfield(driver, :mpc_problem, MPCProblem(obj_fn)))
  setfield(driver, s, fn)
  return getfield(driver, s)
end

