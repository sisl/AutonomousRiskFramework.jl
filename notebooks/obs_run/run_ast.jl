function obs_cem_losses(d, samples, r_dist_net; mdp::ASTMDP, initstate::ASTState)
    sim = mdp.sim
    env = GrayBox.environment(sim)

    N = length(samples)
    sample_length = length(last(first(samples[1]))) # get length of sample vector ("second" element in pair using "first" key)

    true_sequences = SharedArray{VecSE2{Float64}}(N, 2, sample_length)
    
    # Collect true state sequence using sequences of noisy states as actions
    if nprocs() > 1
        @sync @distributed for i_samp in 1:N
            sample = samples[i_samp]
            
            # reset
            BlackBox.initialize!(sim)
            s = initstate
            AST.go_to_state(mdp, s)

            for i in 1:sample_length
                scene = mdp.sim.state
                true_sequences[i_samp, 1, i] = posg(scene[AdversarialDriving.id(mdp.sim.sut)])
                true_sequences[i_samp, 2, i] = posg(scene[AdversarialDriving.id(mdp.sim.adversary)])
                env_sample = GrayBox.EnvironmentSample()
                for k in keys(sample)
                    value = sample[k][i]
                    env_sample[k] = GrayBox.Sample(value, 0.0)
                end
                a = ASTSampleAction(env_sample)
                s = @gen(:sp)(mdp, s, a)

                if BlackBox.isterminal(mdp.sim)
                    break
                end
            end
        end
    else
        for i_samp in 1:N
            sample = samples[i_samp]
            
            # reset
            BlackBox.initialize!(sim)
            s = initstate
            AST.go_to_state(mdp, s)

            for i in 1:sample_length
                scene = mdp.sim.state
                true_sequences[i_samp, 1, i] = posg(scene[AdversarialDriving.id(mdp.sim.sut)])
                true_sequences[i_samp, 2, i] = posg(scene[AdversarialDriving.id(mdp.sim.adversary)])
                env_sample = GrayBox.EnvironmentSample()
                for k in keys(sample)
                    value = sample[k][i]
                    env_sample[k] = GrayBox.Sample(value, 0.0)
                end
                a = ASTSampleAction(env_sample)
                s = @gen(:sp)(mdp, s, a)

                if BlackBox.isterminal(mdp.sim)
                    break
                end
            end
        end
    end

    true_sequences = convert(Array, true_sequences)

    logprobs_all = ObservationModels.calc_logprobs(sim, r_dist_net, true_sequences, samples)
    
    # Compute the costs for all samples
    costs = SharedArray{Float64}(N)

    if nprocs() > 1
        @sync @distributed for i_samp in 1:N
            sample = samples[i_samp]
            logprobs = logprobs_all[i_samp]
            BlackBox.initialize!(sim)
            s = initstate
            AST.go_to_state(mdp, s)
            R = 0 # accumulated reward

            for i in 1:sample_length
                env_sample = GrayBox.EnvironmentSample()
                for k in keys(sample)
                    value = sample[k][i]
                    # logprob = logpdf(env[k], value) # log-probability from true distribution
                    logprob = logprobs[k][i]
                    env_sample[k] = GrayBox.Sample(value, logprob)
                end
                a = ASTSampleAction(env_sample)
                (s, r) = @gen(:sp, :r)(mdp, s, a)
                R += r
                
                if BlackBox.isterminal(mdp.sim)
                    break
                end
            end
            costs[i_samp] = -R # negative (loss)
        end
    else
        for i_samp in 1:N
            sample = samples[i_samp]
            logprobs = logprobs_all[i_samp]
            BlackBox.initialize!(sim)
            s = initstate
            AST.go_to_state(mdp, s)
            R = 0 # accumulated reward

            for i in 1:sample_length
                env_sample = GrayBox.EnvironmentSample()
                for k in keys(sample)
                    value = sample[k][i]
                    # logprob = logpdf(env[k], value) # log-probability from true distribution
                    logprob = logprobs[k][i]
                    env_sample[k] = GrayBox.Sample(value, logprob)
                end
                a = ASTSampleAction(env_sample)
                (s, r) = @gen(:sp, :r)(mdp, s, a)
                R += r
                
                if BlackBox.isterminal(mdp.sim)
                    break
                end
            end
            costs[i_samp] = -R # negative (loss)
        end
    end

    costs = convert(Array, costs)
    # @show costs
    # throw("Hi")
    return costs 
end

function POMDPs.action(planner::CEMPlanner, s; rng=Random.GLOBAL_RNG)
    mdp::ASTMDP = planner.mdp
    if actiontype(mdp) != ASTSampleAction
        error("MDP action type must be ASTSampleAction to use CEM.")
    end

    env::GrayBox.Environment = GrayBox.environment(mdp.sim)

    # Importance sampling distributions, fill one per time step.
    is_dist_0 = convert(Dict{Symbol, Vector{Sampleable}}, env, planner.solver.episode_length)

    # # Calculate reward distributions
    # r_dist_net = ObservationModels.simple_distribution_fit(mdp.sim, mdp.sim.obs_noise[:ranges])

    # Run cross-entropy method using importance sampling
    loss = (d, samples)->obs_cem_losses(d, samples, r_dist_net; mdp=mdp, initstate=s)
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
                                       batched=true,
                                       rng=rng)

    # Save the importance sampling distributions
    planner.is_dist = is_dist_opt

    # Pass back action trace if recording is on (i.e. top_k)
    if mdp.params.top_k > 0
        # return get_top_path(mdp)
        return get_dist_top_path(mdp, planner.is_dist, s)
        # return rand(planner.is_dist)
    else
        return planner.is_dist # pass back the importance sampling distributions
    end
end

# Find the most likely failure sequence from the sampling distribution
function get_dist_top_path(mdp, dist, initstate)
    sim = mdp.sim
    env = GrayBox.environment(sim)

    sample = rand(dist) #just a placeholder for a sample

    sample_length = length(last(first(sample))) # get length of sample vector ("second" element in pair using "first" key)

     # reset
     BlackBox.initialize!(sim)
     s = initstate
     AST.go_to_state(mdp, s)

    path = ASTAction[]
    for i in 1:sample_length
        env_sample = GrayBox.EnvironmentSample()
        for k in keys(sample)
            value = mode(dist[k][i])    # Distribution mode 
            # Replace with recalculation of the logprob
            logprob = logpdf(env[k], value) # log-probability from starting distribution
            # logprob = logprobs[k][i]
            env_sample[k] = GrayBox.Sample(value, logprob)
        end
        a = ASTSampleAction(env_sample)
        s = @gen(:sp)(mdp, s, a)

        push!(path, a)
        if BlackBox.isterminal(mdp.sim)
            break
        end
    end
    path
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
    solver = POMDPStressTesting.CEMSolver(n_iterations=50,
                       num_samples=500,
                       elite_thresh=3000.,
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