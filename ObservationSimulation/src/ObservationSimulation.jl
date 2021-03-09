module ObservationSimulation

export roadway

using AutomotiveSimulator, AutomotiveVisualization

roadway = gen_straight_roadway(3, 200.0) # lanes and length (meters)

end # module
