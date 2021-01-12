@with_kw struct AutoRiskParams
    endtime::Real = 30 # Simulate end time
end;

@with_kw mutable struct AutoRiskSim <: GrayBox.Simulation
    t::Real = 0 # Current time
    params::AutoRiskParams = AutoRiskParams() # Parameters
    dist_params::DistParams = DistParams() # Parameters for observation distribution estimation

    # System under test, ego vehicle
    sut = BlinkerVehicleAgent(get_ped_vehicle(id=1, s=5.0, v=15.0),
                              TIDM(ped_TIDM_template, noisy_observations=true))

    # Noisy adversary, pedestrian
    adversary = NoisyPedestrianAgent(get_pedestrian_noisy(id=2, s=7.0, v=2.0, noise=init_noise),
                                     AdversarialPedestrian())

    # Adversarial Markov decision process
    problem::MDP = AdversarialDrivingMDP(sut, [adversary], ped_roadway, 0.1)
    state::Scene = rand(initialstate(problem))
    prev_distance::Real = -Inf # Used when agent goes out of frame

    # Noise distributions and disturbances
    # xposition_noise_ped::Distribution = INormal_Uniform(0, 6) # Gaussian noise (notice larger σ)
    # yposition_noise_ped::Distribution = INormal_Uniform(0, 6) # Gaussian noise
    xposition_noise_ped::Distribution = Normal(0, 10) # Gaussian noise (notice larger σ)
    yposition_noise_ped::Distribution = Normal(0, 10) # Gaussian noise
    velocity_noise_ped::Distribution = Normal(0, 1) # Gaussian noise

    xposition_noise_sut::Distribution = Normal(0, 10) # Gaussian noise (notice larger σ)
    yposition_noise_sut::Distribution = Normal(0, 1) # Gaussian noise
    velocity_noise_sut::Distribution = Normal(0, 1) # Gaussian noise
    
    # GPS range noise
    range_sigma = 5.0
    obs_noise::Dict{Symbol, Vector{Sampleable}} = Dict(
        :ranges => [Normal(0, range_sigma), 
                    Normal(0, range_sigma), 
                    Normal(0, range_sigma), 
                    Normal(0, range_sigma), 
                    Normal(0, range_sigma)]
        ) # Array of Gaussian
    
    disturbances = Disturbance[BlinkerVehicleControl(), PedestrianControl()] # Initial 0-noise disturbance
end;

# Step the scene forward by one timestep and return the next state
function AdversarialDriving.step_scene(mdp::AdversarialDriving.AdversarialDrivingMDP, s::Scene, actions::Vector{Disturbance}, rng::AbstractRNG = Random.GLOBAL_RNG)
    entities = []

    # Add noise in SUT
    update_adversary!(sut(mdp), actions[1], s)

    # Loop through the adversaries and apply the instantaneous aspects of their disturbance
    for (adversary, action) in zip(adversaries(mdp), actions[2:end])
        update_adversary!(adversary, action, s)
    end

    # Loop through the vehicles in the scene, apply action and add to next scene
    for (i, veh) in enumerate(s)
        m = model(mdp, veh.id)
        observe!(m, s, mdp.roadway, veh.id)
        a = rand(rng, m)
        bv = Entity(propagate(veh, a, mdp.roadway, mdp.dt), veh.def, veh.id)
        !end_of_road(bv, mdp.roadway, mdp.end_of_road) && push!(entities, bv)
    end
    isempty(entities) ? Scene(typeof(sut(mdp).get_initial_entity())) : Scene([entities...])
end

function GrayBox.environment(sim::AutoRiskSim)
    
    # RECIPE:
    # Get current state of each entity
    # Calculate the distribution of each measurement (need to do sampling)
    # Draw samples from the measurements and pass to the localization/state estimation function
    # The function returns noisy states of ALL entities 
    # Fit distribution over noise in estimated entity states
    
    # if sim.t>0
    #     adv_ent = sim.state[AdversarialDriving.id(sim.adversary)]

    #     (xposition_noise, yposition_noise, velocity_noise) = ObservationModels.gps_distribution_estimate(adv_ent, [sim.range_noise..., sim.velocity_noise], sim.dist_params)

    # else
    #     xposition_noise = Normal(0., 1.)
    #     yposition_noise = Normal(0., 1.)
    #     velocity_noise = Normal(0., 1.)
    # end
    # prev_noise = noise(sim.state[AdversarialDriving.id(sim.adversary)])

    return GrayBox.Environment(
                            # :vel_ped => sim.velocity_noise_ped,
                            :xpos_ped => sim.xposition_noise_ped,
                            :ypos_ped => sim.yposition_noise_ped,
                            # :vel_sut => sim.velocity_noise_sut,
                            :xpos_sut => sim.xposition_noise_sut,
                            :ypos_sut => sim.yposition_noise_sut
                        )
end

function GrayBox.transition!(sim::AutoRiskSim, sample::GrayBox.EnvironmentSample)
    sim.t += sim.problem.dt # Keep track of time

#     println(sample)
    
    # replace current noise with new sampled noise
#     range_noise = [sample[:range_1].value, sample[:range_2].value, sample[:range_3].value, sample[:range_4].value, sample[:range_5].value]
#     range_noise = [sample[:range_1].value, 0.0, sample[:range_3].value, 0.0, 0.0]
#     noise = Noise(pos = (0.0, 0.0), vel = sample[:vel].value, gps_range = [0.0, 0.0, 0.0, 0.0, 0.0])
#     noise = Noise(pos = (0.0, 0.0), vel = 0.0, gps_range = range_noise)
    # noise_ped = Noise(pos = (sample[:xpos_ped].value, sample[:ypos_ped].value), vel = sample[:vel_ped].value)
    # noise_sut = Noise(pos = (sample[:xpos_sut].value, sample[:ypos_sut].value), vel = sample[:vel_sut].value)
    noise_ped = Noise(pos = (sample[:xpos_ped].value, sample[:ypos_ped].value), vel = 0.0)
    noise_sut = Noise(pos = (sample[:xpos_sut].value, sample[:ypos_sut].value), vel = 0.0)
#     noise = Noise(pos = (sample[:xpos].value, sample[:ypos].value), vel = sample[:vel].value, gps_range = [0.0, 0.0, 0.0, 0.0, 0.0])
#     noise = Noise(pos = (sample[:xpos].value, sample[:ypos].value), vel = 0.0, gps_range = [0.0, 0.0, 0.0, 0.0, 0.0])
    sim.disturbances[1] = BlinkerVehicleControl(noise=noise_sut)
    sim.disturbances[2] = PedestrianControl(noise=noise_ped)

    # step agents: given MDP, current state, and current action (i.e. disturbances)
    (sim.state, r) = @gen(:sp, :r)(sim.problem, sim.state, sim.disturbances)

    # return log-likelihood of actions, summation handled by `logpdf()`
    return logpdf(sample)::Real
end

function BlackBox.initialize!(sim::AutoRiskSim)
    sim.t = 0
    sim.problem = AdversarialDrivingMDP(sim.sut, [sim.adversary], ped_roadway, 0.1)
    sim.state = rand(initialstate(sim.problem))
    sim.disturbances = Disturbance[BlinkerVehicleControl(), PedestrianControl()] # noise-less
    sim.prev_distance = -Inf
end

simx = AutoRiskSim()
BlackBox.initialize!(simx);

out_of_frame(sim) = length(sim.state.entities) < 2 # either agent went out of frame

function BlackBox.distance(sim::AutoRiskSim)
    if out_of_frame(sim)
        return sim.prev_distance
    else
        pedestrian, vehicle = sim.state.entities
        pos1 = posg(pedestrian)
        pos2 = posg(vehicle)
        return hypot(pos1.x - pos2.x, pos1.y - pos2.y)
    end
end

begin
    envsample = rand(GrayBox.environment(simx))
    GrayBox.transition!(simx, envsample)
    BlackBox.distance(simx)
end

function BlackBox.isevent(sim::AutoRiskSim)
    if out_of_frame(sim)
        return false
    else
        pedestrian, vehicle = sim.state.entities
        return collision_checker(pedestrian, vehicle)
    end
end

function BlackBox.isterminal(sim::AutoRiskSim)
    return isterminal(sim.problem, sim.state) ||
           out_of_frame(sim) ||
           BlackBox.isevent(sim) ||
           sim.t ≥ sim.params.endtime
end

function BlackBox.evaluate!(sim::AutoRiskSim, sample::GrayBox.EnvironmentSample)
    logprob::Real = GrayBox.transition!(sim, sample) # Step simulation
    d::Real       = BlackBox.distance(sim)           # Calculate miss distance
    event::Bool   = BlackBox.isevent(sim)            # Check event indication
    sim.prev_distance = d                            # Store previous distance
    return (logprob::Real, d::Real, event::Bool)
end

begin
    envsample2 = rand(GrayBox.environment(simx))
    BlackBox.evaluate!(simx, envsample2) # (log-likelihood, distance, isevent)
end