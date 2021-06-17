using Distributions, Debugger
using Cairo, Gtk
using AutomotiveSimulator, AutomotiveVisualization
using IntelligentDriving
ID = IntelligentDriving

include("draw_utils.jl")

roadway = gen_stadium_roadway(3)

w = DEFAULT_LANE_WIDTH
scene = Scene([
  Entity(
    VehicleState(VecSE2(30.0, -w, 0.0), roadway, 10.0),
    VehicleDef(),
    :alice,
  ),
  Entity(
    VehicleState(VecSE2(40.0, 0.0, 0.0), roadway, 20.0),
    VehicleDef(),
    :bob,
  ),
  Entity(
    VehicleState(VecSE2(25.0, -w, 0.0), roadway, 15.0),
    VehicleDef(),
    :charlie,
  ),
])
car_colors = get_pastel_car_colors(scene)

renderables = vcat(
  [roadway],
  [FancyCar(car = veh, color = car_colors[veh.id]) for veh in scene],
)
snapshot = render(renderables)
#show_cairo(snapshot)

################################################################################
timestep = 0.1
nticks = 500

models = Dict{Symbol,DriverModel}(
  :alice => LatLonSeparableDriver( # produces LatLonAccels
    ProportionalLaneTracker(), # lateral model
    IntelligentDriverModel(), # longitudinal model
  ),
  #:bob => Tim2DDriver(mlane = MOBIL()),
  :bob => LatLonSeparableDriver( # produces LatLonAccels
    ProportionalLaneTracker(), # lateral model
    IntelligentDriverModel(), # longitudinal model
  ),
  :charlie => ID.MPCDriver(),
)

set_desired_speed!(models[:alice], 12.0)
set_desired_speed!(models[:bob], 12.0)
set_desired_speed!(models[:charlie], 15.0)

# test track_longitudinal! #####################################################
#ego, oth = models[:charlie], models[:alice]
#ego_ent = filter(x -> x.id == :charlie, collect(scene))[1]
#oth_ent = filter(x -> x.id == :alice, collect(scene))[1]
#v_ego, v_oth = velf(ego_ent).s, velf(oth_ent).s
#
#track_longitudinal!(
#  ego,
#  v_ego,
#  v_oth,
#  get_frenet_relative_position(oth_ent, ego_ent, roadway).Δs,
#)
#println()
#display(models[:charlie].a_lon)
#display(models[:charlie].a_lat)

scenes = simulate(scene, roadway, models, nticks, timestep)
#frames = render_frames(scenes, roadway)
show_cairo(render_snapshots(scenes, roadway))
return
