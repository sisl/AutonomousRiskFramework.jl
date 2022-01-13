POMDPs.solve(solver::ISDPWSolver, mdp::Union{POMDP,MDP}) = ISDPWPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::ISDPWPlanner)
    p.tree = nothing
end

"""
Utility function for numerically stable softmax
Adapted from: https://nextjournal.com/jbowles/the-mathematical-ideal-and-softmax-in-julia
"""
_exp(x) = exp.(x .- maximum(x))
_exp(x, θ::AbstractFloat) = exp.((x .- maximum(x)) * θ)
_sftmax(e, d::Integer) = (e ./ sum(e, dims = d))

function softmax(X, dim::Integer)
    _sftmax(_exp(X), dim)
end

function softmax(X, dim::Integer, θ::Float64)
    _sftmax(_exp(X, θ), dim)
end

softmax(X) = softmax(X, 1)

"""
quantile computation
"""
function parabolic_update(i, q, n, d)
    return q[i] + d/(n[i+1]-n[i-1])*((n[i]-n[i-1]+d)*(q[i+1]-q[i])/(n[i+1]-n[i]) + (n[i+1]-n[i]-d)*(q[i]-q[i-1])/(n[i]-n[i-1]))
end

function linear_update(i, q, n, d)
    return q[i] + d/(n[i+d]-n[i])*(q[i+d]-q[i])
end

function online_update_quantile(costs, IS_weights, q, n, n_prime, α, cost_i)
    if length(costs) < 5
        return [], [], []
    end
    if length(q) == 0
        q = [costs[i] for i=1:5]
        n = [i for i=1:5]
        n_prime = [1, 1 + 2α, 1+4α, 3+2α, 5]
        cost_i = 5
    end
    for j=cost_i+1:length(costs)
        x = costs[j]
        wt = exp(IS_weights[j])
        if x < q[1]
            q[1] = x
            k = 1
        elseif q[1] ≤ x < q[2]
            k = 1
        elseif q[2] ≤ x < q[3]
            k = 2
        elseif q[3] ≤ x < q[4]
            k = 3
        elseif q[4] ≤ x < q[5]
            k = 4
        else
            q[5] = x
            k = 4
        end
        for i=k:5
            n[i] = n[i] + wt
        end
        dn = [0, α/2, α, (1+α)/2, 1]
        n_prime = n_prime .+ wt*dn
        for i=2:4
            d = n_prime - n
            if ((d ≥ 1) && (n[i+1] - n[i] > 1))||((d ≤ -1) && (n[i-1] - n[i] < -1))
                d = sign(d)
                q_prime = parabolic_update(i, q, n, d)
                if q[i-1] < q_prime < q[i+1]
                    q[i] = q_prime
                else
                    q[i] = linear_update(i, q, n, d)
                end
                n[i] = n[i] + d
            end
        end
        cost_i = j
    end
    return q, n, n_prime, cost_i
end

"""
Calculate next action
"""
function select_action(nodes, values, q, p_prob)
    prob = adaptive_probs(values, q, p_prob)
    sanode_idx = sample(1:length(nodes), Weights(prob))
    sanode = nodes[sanode_idx]
    q_logprob = log(prob[sanode_idx])
    return sanode, q_logprob
end

function adaptive_probs(values, q, p_prob)
    if length(values)==1
        return 1.0
    end

    tail_prob = [(values[i] ≥ q ? values[i]*p_prob[i] : 0) for i=1:length(values)] .+ 1e-5
    notail_prob = [(values[i] < q ? p_prob[i] : 0) for i=1:length(values)] .+ 1e-5
    tail_prob /= sum(tail_prob)
    notail_prob /= sum(notail_prob)

    # Mixture weighting
    prob = 0.3*notail_prob .+ 0.7*tail_prob
    # @show tail_prob, notail_prob
    return prob
end

"""
Calculate IS weights
"""
function compute_IS_weight(q_logprob, a, distribution)
    if distribution === nothing
        w = -q_logprob
    else
        w = logpdf(distribution, a) - q_logprob
    end
    @show a, q_logprob, w
    return w
end

"""
Construct an ISDPW tree and choose an action.
"""
POMDPs.action(p::ISDPWPlanner, s) = first(action_info(p, s))

MCTS.estimate_value(f::Function, mdp::Union{POMDP,MDP}, state, w::Float64, depth::Int) = f(mdp, state, w, depth)

"""
Construct an ISDPW tree and choose the best action. Also output some information.
"""
function POMDPModelTools.action_info(p::ISDPWPlanner, s; tree_in_info=false, w=0.0)
    local a::actiontype(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """)
        end

        S = statetype(p.mdp)
        A = actiontype(p.mdp)
        if p.solver.keep_tree
            if p.tree === nothing
                tree = MCTS.DPWTree{S,A}(p.solver.n_iterations)
                p.tree = tree
            else
                tree = p.tree
            end
            if haskey(tree.s_lookup, s)
                snode = tree.s_lookup[s]
            else
                snode = MCTS.insert_state_node!(tree, s, true)
            end
        else
            tree = MCTS.DPWTree{S,A}(p.solver.n_iterations)
            p.tree = tree
            snode = MCTS.insert_state_node!(tree, s, p.solver.check_repeat_state)
        end

        p.solver.show_progress ? progress = Progress(p.solver.n_iterations) : nothing
        nquery = 0
        start_us = MCTS.CPUtime_us()
        for i = 1:p.solver.n_iterations
            nquery += 1
            simulate(p, snode, w, p.solver.depth) # (not 100% sure we need to make a copy of the state here)
            p.solver.show_progress ? next!(progress) : nothing
            if MCTS.CPUtime_us() - start_us >= p.solver.max_time * 1e6
                p.solver.show_progress ? finish!(progress) : nothing
                break
            end
        end
        p.reset_callback(p.mdp, s) # Optional: leave the MDP in the current state.
        info[:search_time_us] = MCTS.CPUtime_us() - start_us
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        sanode = best_sanode(tree, snode)
        a = tree.a_labels[sanode] # choose action randomly based on approximate value
    catch ex
        a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, info
end


"""
Return the reward for one iteration of MCTSDPW.
"""
function simulate(dpw::ISDPWPlanner, snode::Int, w::Float64, d::Int)
    S = statetype(dpw.mdp)
    A = actiontype(dpw.mdp)
    sol = dpw.solver
    tree = dpw.tree
    s = tree.s_labels[snode]
    dpw.reset_callback(dpw.mdp, s) # Optional: used to reset/reinitialize MDP to a given state.
    if isterminal(dpw.mdp, s)
        return 0.0
    elseif d == 0
        return estimate_value(dpw.solved_estimate, dpw.mdp, s, w, d)
    end

    # action progressive widening
    if dpw.solver.enable_action_pw
        if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
            a = next_action(dpw.next_action, dpw.mdp, s, MCTS.DPWStateNode(tree, snode)) # action generation step
            if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                n0 = init_N(sol.init_N, dpw.mdp, s, a)
                MCTS.insert_action_node!(tree, snode, a, n0,
                                    init_Q(sol.init_Q, dpw.mdp, s, a),
                                    sol.check_repeat_action
                                   )
                tree.total_n[snode] += n0
            end
        end
    elseif isempty(tree.children[snode])
        for a in support(actions(dpw.mdp, s))
            n0 = init_N(sol.init_N, dpw.mdp, s, a)
            MCTS.insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, dpw.mdp, s, a),
                                false)
            tree.total_n[snode] += n0
        end
    end

    all_UCB = []
    ltn = log(tree.total_n[snode])
    for child in tree.children[snode]
        n = tree.n[child]
        q = tree.q[child]
        c = sol.exploration_constant # for clarity
        if (ltn <= 0 && n == 0) || c == 0.0
            UCB = q
        else
            UCB = q + c*sqrt(ltn/n)
        end
        @assert !isnan(UCB) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
        @assert !isequal(UCB, -Inf)

        push!(all_UCB, UCB)
    end
    # print("Softmax weights: ", softmax(all_UCB))
    # dpw.quantile_q, dpw.quantile_n, dpw.quantile_n_prime, dpw.cost_i = online_update_quantile(dpw.mdp.costs, dpw.mdp.IS_weights, dpw.quantile_q, dpw.quantile_n, dpw.quantile_n_prime, dpw.quantile_α, dpw.cost_i)
    sanode, q_logprob = select_action(tree.children[snode], all_UCB, max(all_UCB...)/1.2, [pdf(actions(dpw.mdp, s), a) for a in support(actions(dpw.mdp, s))])
    a = tree.a_labels[sanode] # choose action randomly based on approximate value
    w_node = compute_IS_weight(q_logprob, a, actions(dpw.mdp, s))
    w = w + w_node

     # state progressive widening
    new_node = false
    if (dpw.solver.enable_state_pw && tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state) || tree.n_a_children[sanode] == 0
        sp, r = @gen(:sp, :r)(dpw.mdp, s, [a, w], dpw.rng)

        if sol.check_repeat_state && haskey(tree.s_lookup, sp)
            spnode = tree.s_lookup[sp]
        else
            spnode = MCTS.insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
            new_node = true
        end

        push!(tree.transitions[sanode], (spnode, r))

        if !sol.check_repeat_state
            tree.n_a_children[sanode] += 1
        elseif !((sanode,spnode) in tree.unique_transitions)
            push!(tree.unique_transitions, (sanode,spnode))
            tree.n_a_children[sanode] += 1
        end
    else
        spnode, r = rand(dpw.rng, tree.transitions[sanode])
    end

    if new_node
        # preload_actions!(dpw, tree, sp, spnode)
        q = r + discount(dpw.mdp)*estimate_value(dpw.solved_estimate, dpw.mdp, sp, w, d-1)
    else
        q = r + discount(dpw.mdp)*simulate(dpw, spnode, w, d-1)
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1

    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    return q
end

"""
Return the best action.
Some publications say to choose action that has been visited the most
e.g., Continuous Upper Confidence Trees by Couëtoux et al.
"""
function best_sanode(tree::MCTS.DPWTree, snode::Int)
    best_Q = -Inf
    sanode = 0
    for child in tree.children[snode]
        if tree.q[child] > best_Q
            best_Q = tree.q[child]
            sanode = child
        end
    end
    return sanode
end
