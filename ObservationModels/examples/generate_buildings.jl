using ObservationModels
using AutomotiveSimulator
using AutomotiveVisualization
using AdversarialDriving

##############################################################################
"""
Generate Buildings
"""
function gen_my_buildings(nbuildings::Int, top::Float64, bottom::Float64)
    buildingmap = BuildingMap()
    id = 1

    # Top buildings
    for i = 1:nbuildings
            origin::VecSE2{Float64} = VecSE2(25.0 + (i-1)*13, top + 1.0, 0.0)
            push!(buildingmap.buildings, Building(id, 10.0, 3.0, origin))
            id += 1
    end

    # Bottom buildings
    for i = 1:nbuildings
        origin::VecSE2{Float64} = VecSE2(25.0 + (i-1)*13, bottom - 1.0, 0.0)
        push!(buildingmap.buildings, Building(id, 10.0, 3.0, origin))
        id += 1
    end

    return buildingmap
end

buildingmap = gen_my_buildings(7, 9.0, -3.0) # num buildings, top and bottom points (meters)

##############################################################################
"""
Generate Roadway and scene
"""
## Geometry parameters
roadway_length = 100.

roadway = gen_straight_roadway(3, roadway_length) # lanes and length (meters)

init_noise = Noise(pos = (0, 0), vel = 0)

scene = Scene([
	Entity(BlinkerState(VehicleState(VecSE2(10.0, 6.0, 0), roadway, 10.0), false, [], init_noise), VehicleDef(), 1),
	Entity(BlinkerState(VehicleState(VecSE2(30.0, 3.0, 0), roadway, 0.0), false, [], init_noise), VehicleDef(), 2)
]);

# Render the scene
render([roadway, buildingmap, scene])