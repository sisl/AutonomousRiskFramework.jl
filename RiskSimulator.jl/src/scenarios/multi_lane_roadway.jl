"""
Create multi-lane roadway.
- w: lane width
- l: lane length
- n: number of lanes
"""
function multi_lane_roadway(; w=DEFAULT_LANE_WIDTH, l=50.0, n=3, origin=VecSE2(0.0,0,0))
    roadway = gen_straight_roadway(n, l, origin=origin) # lanes and length (meters)
    return roadway
end


function hw_behind(; roadway::Roadway, noise::Noise=Noise(), id::Int64=1, s::Float64=2.0, v::Float64=15.0, goals::Vector=Int[], blinker::Bool=false)
    vehicle = BlinkerVehicle(id=id, roadway=roadway, lane=1, s=s, v=v, goals=goals, blinker=blinker, noise=noise)
    return (rng::AbstractRNG=Random.GLOBAL_RNG) -> vehicle
end


function hw_stopping(; roadway::Roadway, noise::Noise=Noise(), id::Int64=1, s::Float64=20.0, v::Float64=2.0, goals::Vector=Int[], blinker::Bool=false)
    vehicle = BlinkerVehicle(id=id, roadway=roadway, lane=1, s=s, v=v, goals=goals, blinker=blinker, noise=noise)
    return (rng::AbstractRNG=Random.GLOBAL_RNG) -> vehicle
end


# TODO: (kwargs...) for `t_intersection`
function scenario_hw_stopping(; init_noise_1=Noise(), init_noise_2=Noise(), ignore_idm=false)
    roadway = multi_lane_roadway()

    # TODO: ignore_idm = !params.ignore_sensors
    sut_model = UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=true, ignore_idm=ignore_idm)
    sut = BlinkerVehicleAgent(hw_behind(id=1, noise=init_noise_1, roadway=roadway), sut_model);

    adversary_model = UrbanIDM(idm=IntelligentDriverModel(v_des=0.0), noisy_observations=false)
    adversary = BlinkerVehicleAgent(hw_stopping(id=2, noise=init_noise_2, roadway=roadway), adversary_model)

    return Scenario(roadway, sut, adversary)
end