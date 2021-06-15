"""
Return pedestrain crosswalk roadway from AdversarialDriving.
"""
function crosswalk_roadway()
    return [AdversarialDriving.ped_roadway, AdversarialDriving.crosswalk]
end


function cw_left_to_right(; roadway::Roadway, noise::Noise=Noise(), id::Int64=1, s::Float64=0.0, v::Float64=12.0, goals::Vector=[1], blinker::Bool=false)
    vehicle = BlinkerVehicle(id=id, roadway=roadway, lane=1, s=s, v=v, goals=goals, blinker=blinker, noise=noise)
    return (rng::AbstractRNG=Random.GLOBAL_RNG) -> vehicle
end


function cw_pedestrian_walking_up(; roadway::Roadway, noise::Noise=Noise(), id::Int64=1, s::Float64=8.0, v::Float64=2.0)
    pedestrain = NoisyPedestrian(id=id, roadway=roadway, lane=2, s=s, v=v, noise=noise)
    return (rng::AbstractRNG=Random.GLOBAL_RNG) -> pedestrain
end

# TODO: (kwargs...) for `crosswalk_roadway`
# TODO: UrbanIDM???
function scenario_pedestrian_crosswalk(; init_noise_1=Noise(), init_noise_2=Noise())
    roadway, crosswalk = crosswalk_roadway()
    # TODO: Store `crosswalk` for rendering

    # sut_model = UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=true, ignore_idm=!params.ignore_sensors)
    sut_model = TIDM(get_XIDM_template())
    sut_model.noisy_observations = true # TODO. Flip who is SUT/adversary?
    sut = BlinkerVehicleAgent(cw_left_to_right(id=1, noise=init_noise_1, roadway=roadway), sut_model);

    # adversary_model = UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=false)
    adversary_model = AdversarialPedestrian() # important.
    adversary = NoisyPedestrianAgent(cw_pedestrian_walking_up(id=2, noise=init_noise_2, roadway=roadway), adversary_model)

    return Scenario(roadway, sut, adversary, Disturbance[BlinkerVehicleControl(), PedestrianControl()])
end
