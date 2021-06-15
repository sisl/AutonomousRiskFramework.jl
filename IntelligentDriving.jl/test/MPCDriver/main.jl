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
    VehicleState(VecSE2(10.0, -w, 0.0), roadway, 15.0),
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

scenes = simulate(scene, roadway, models, nticks, timestep)
#frames = render_frames(scenes, roadway)
show_cairo(render_snapshots(scenes, roadway))
return
