function obs_cem_losses(d, sample, r_dist_net; mdp::ASTMDP, initstate::ASTState)
    sim = mdp.sim
    env = GrayBox.environment(sim)

    true_s = []
    s = initstate

    BlackBox.initialize!(sim)
    AST.go_to_state(mdp, s)

    sample_length = length(last(first(sample))) # get length of sample vector ("second" element in pair using "first" key)
    temp = 0.0

    # Collect true state sequence for given sequence of noisy states
    for i in 1:sample_length
        push!(true_s, mdp.sim.state)
        env_sample = GrayBox.EnvironmentSample()
        for k in keys(sample)
            value = sample[k][i]
            logprob = 0.0
            env_sample[k] = GrayBox.Sample(value, logprob)
        end
        a = ASTSampleAction(env_sample)
        s = @gen(:sp)(mdp, s, a)
        
        if BlackBox.isterminal(sim)
            break
        end
    end

    logprobs = set_logprob(sim, r_dist_net, true_s, sample)

    BlackBox.initialize!(sim)
    s = initstate
    AST.go_to_state(mdp, s)
    R = 0 # accumulated reward

    for i in 1:sample_length
        env_sample = GrayBox.EnvironmentSample()
        for k in keys(sample)
            value = sample[k][i]
            # logprob = logpdf(env[k], value) # log-probability from true distribution
            logprob = logprobs[i][k]
            env_sample[k] = GrayBox.Sample(value, logprob)
        end
        a = ASTSampleAction(env_sample)
        (s, r) = @gen(:sp, :r)(mdp, s, a)
        R += r
        
        if BlackBox.isterminal(sim)
            break
        end
    end
    # throw("Hi")
    # R += temp
    # R += traj_logprob(sample, true_s, sim)

    # @show -R
    return -R # negative (loss)
end

function POMDPs.action(planner::CEMPlanner, s; rng=Random.GLOBAL_RNG)
    mdp::ASTMDP = planner.mdp
    if actiontype(mdp) != ASTSampleAction
        error("MDP action type must be ASTSampleAction to use CEM.")
    end

    env::GrayBox.Environment = GrayBox.environment(mdp.sim)

    # Importance sampling distributions, fill one per time step.
    is_dist_0 = convert(Dict{Symbol, Vector{Sampleable}}, env, planner.solver.episode_length)

    # Calculate reward distributions
    r_dist_net = ObservationModels.simple_distribution_fit(mdp.sim, mdp.sim.obs_noise[:ranges])

    # Run cross-entropy method using importance sampling
    loss = (d, sample)->obs_cem_losses(d, sample, r_dist_net; mdp=mdp, initstate=s)
    is_dist_opt = cross_entropy_method(loss,
                                       is_dist_0;
                                       max_iter=planner.solver.n_iterations,
                                       N=planner.solver.num_samples,
                                       min_elite_samples=planner.solver.min_elite_samples,
                                       max_elite_samples=planner.solver.max_elite_samples,
                                       elite_thresh=planner.solver.elite_thresh,
                                       weight_fn=planner.solver.weight_fn,
                                       add_entropy=planner.solver.add_entropy,
                                       verbose=planner.solver.verbose,
                                       show_progress=planner.solver.show_progress,
                                       rng=rng)

    # Save the importance sampling distributions
    planner.is_dist = is_dist_opt

    # Pass back action trace if recording is on (i.e. top_k)
    if mdp.params.top_k > 0
        return get_top_path(mdp)
    else
        return planner.is_dist # pass back the importance sampling distributions
    end
end

function setup_ast(seed=0)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = AutoRiskSim()

    # AST MDP formulation object
    mdp::ASTMDP = ASTMDP{ASTSampleAction}(sim)
    mdp.params.debug = true # record metrics
    mdp.params.top_k = 10   # record top k best trajectories
    mdp.params.seed = seed  # set RNG seed for determinism
    
#     function null_priority(mdp, s, snode) # snode is the state node of type DPWStateNode
# #         @show snode.tree.a_lookup
# #         Replace with null action calculated from noise-free measurements
#         null_action = ASTSampleAction(
#                                     GrayBox.EnvironmentSample(
#                                                 :vel => GrayBox.Sample(0., logpdf(Normal(0., 1), 0.)),
#                                                 :xpos => GrayBox.Sample(0., logpdf(Normal(0., 5.), 0.)),
#                                                 :ypos => GrayBox.Sample(0., logpdf(Normal(0., 1.), 0.))
#                                                 )
#                                     )
#         n_children = length(snode.tree.children[snode.index])
#         if n_children > 0
#             new_action = rand(actions(mdp, s))  # add a random action
#         else
#             new_action = null_action
#         end
# #         @show new_action.sample
#         return new_action
#     end
#     function prior_Q(mdp, s, a)
#         l1_norm = 0
#         for (key, val) in a.sample
#             l1_norm = l1_norm + abs(val.value)
#         end
#         return 100.
#     end


#     # Hyperparameters for MCTS-PW as the solver
#     solver = MCTSPWSolver(n_iterations=1000,        # number of algorithm iterations
#                           exploration_constant=1.0, # UCT exploration
#                           k_action=1.0,             # action widening
#                           alpha_action=0.5,         # action widening
#                           depth=sim.params.endtime, # tree depth
# #                           init_Q=100.,
# #                           init_N=1,
#                           next_action=null_priority
# #                           estimate_value=cem_rollout
#                          )
    solver = POMDPStressTesting.CEMSolver(n_iterations=100,
                       num_samples=300,
                       elite_thresh=100.,
                       min_elite_samples=20,
                       max_elite_samples=200,
                       episode_length=sim.params.endtime)
    
# return actions(mdp, initialstate(mdp))
    
# Get online planner (no work done, yet)
    planner = solve(solver, mdp)

    return planner
end;

planner = setup_ast();

action_trace = search!(planner);