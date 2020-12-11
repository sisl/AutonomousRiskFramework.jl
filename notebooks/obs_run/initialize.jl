init_noise = Noise(pos = (0, 0), vel = 0)
cont_noise = Noise(pos = (-7, 0), vel = 0)

sut_agent = BlinkerVehicleAgent(get_ped_vehicle(id=1, s=5.0, v=15.0),
    TIDM(ped_TIDM_template, noisy_observations=true));

get_pedestrian_noisy(;id::Int64, s::Float64, v::Float64, noise::Noise) = (rng::AbstractRNG = Random.GLOBAL_RNG) -> NoisyPedestrian(roadway = AdversarialDriving.ped_roadway, lane = 2, s=s, v=v, id=id, noise=noise)

adv_ped = NoisyPedestrianAgent(get_pedestrian_noisy(id=2, s=7.0, v=2.0, noise=init_noise), AdversarialPedestrian());

ad_mdp = AdversarialDrivingMDP(sut_agent, [adv_ped], ped_roadway, 0.1);

ped_state, veh_state = initialstate(ad_mdp)

noisy_action = Disturbance[PedestrianControl(a=VecE2(0, 0), da=VecE2(0, 0), noise=cont_noise)]

# Behavior with noise
hist_noise = POMDPSimulators.simulate(HistoryRecorder(), ad_mdp, FunctionPolicy((s) -> noisy_action));

map(x -> (x.entities[1].state.veh_state.posG[2], 
         AdversarialDriving.noisy_entity(x.entities[1], ad_mdp.roadway).state.veh_state.posG[2], 
         noise(x.entities[1]).pos[1]) , 
         POMDPSimulators.stepthrough(ad_mdp, FunctionPolicy((s) -> noisy_action), "s", max_steps=20))

ad_scenes_noise = state_hist(hist_noise);

# win = Blink.Window()

# # t = 10
# man = @manipulate for t=1:length(ad_scenes_noise)
#     AutomotiveVisualization.render([ad_mdp.roadway, crosswalk, ad_scenes_noise[t]])
# end;
# body!(win, man)