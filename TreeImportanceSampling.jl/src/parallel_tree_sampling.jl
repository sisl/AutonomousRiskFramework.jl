# Notes:
# - Essentially ignoring check_repeat_state and check_repeat_action which are by default true so not sure if they matter much
#
# TODOs:
# - Change timer

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
Calculate next action
"""
function select_action(nodes, values; c=5.0)
    prob = softmax(c*values)
    sanode_idx = sample(1:length(nodes), Weights(prob))
    sanode = nodes[sanode_idx]
    q_logprob = log(prob[sanode_idx])
    return sanode, q_logprob
end

"""
Calculate IS weights
"""
function compute_IS_weight(q_logprob, a, distribution)
    if distribution == nothing
        w = -q_logprob
    else
        w = logpdf(distribution, a) - q_logprob
    end
    return w
end

"""
Construct an ISDPW tree and choose an action.
"""
POMDPs.action(p::ISDPWPlanner, s) = first(action_info(p, s))

estimate_value(f::Function, mdp::Union{POMDP,MDP}, state, w::Float64, depth::Int) = f(mdp, state, w, depth)

"""
Construct an ISDPW tree and choose the best action. Also output some information.
"""
function POMDPModelTools.action_info(p::ISDPWPlanner, s; tree_in_info=false, w=0.0, use_prior=true)
    local a::actiontype(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """)
        end

        tree = p.tree
        if !p.solver.keep_tree || tree == nothing
            tree = PDPWTree{statetype(p.mdp),actiontype(p.mdp)}()
            p.tree = tree
        end
        snode = insert_state_node!(tree, s)

        timer = p.solver.timer
        start_s = timer()
        timeout_s = start_s + planner.solver.max_time
        n_iterations = p.solver.n_iterations
        p.solver.show_progress ? progress = Progress(n_iterations) : nothing
        sim_channel = Channel{Task}(min(1000, n_iterations)) do channel
            for n in 1:n_iterations
                put!(channel, Threads.@spawn simulate(planner, snode, w, p.solver.depth, timeout_s; use_prior))
            end
        end

        for sim_task in sim_channel
            if timer() > timeout_us
                p.solver.show_progress ? finish!(progress) : nothing
                break
            end

            try
                fetch(sim_task)  # Throws a TaskFailedException if failed.
                nquery += 1
                p.solver.show_progress ? next!(progress) : nothing
            catch err
                throw(err.task.exception)  # Throw the underlying exception.
            end
        end

        p.reset_callback(p.mdp, s) # Optional: leave the MDP in the current state.
        info[:search_time_us] = (timer() - start_s) * 1e6
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        sanode, q_logprob = sample_sanode(tree, snode)
        a = sanode.a_label
        w_node = compute_IS_weight(q_logprob, a, use_prior ? actions(dpw.mdp, s) : nothing)
        w = w + w_node
    catch ex
        a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, w, info
end


"""
Return the reward for one iteration of MCTSDPW.
"""
function simulate(dpw::ISDPWPlanner, snode::DPWStateNode, w::Float64, d::Int, timeout_s::Float64=0.0; use_prior=true)
    sol = dpw.solver
    timer = sol.timer
    tree = dpw.tree
    s = snode.s_label
    dpw.reset_callback(dpw.mdp, s) # Optional: used to reset/reinitialize MDP to a given state.
    if isterminal(dpw.mdp, s)
        return 0.0
    elseif d == 0 || (timeout_s > 0.0 && timer() > timeout_s)
        return estimate_value(dpw.solved_estimate, dpw.mdp, s, w, d)
    end

    # Action progressive widening.
    if sol.enable_action_pw
        lock(snode.s_lock) do
            !(n_children(snode) <= sol.k_action * total_n(snode)^sol.alpha_action) && return
            a = next_action(dpw.next_action, dpw.mdp, s, snode) # action generation step
            insert_action_node!(tree, snode, a,
                                init_N(sol.init_N, dpw.mdp, s, a),
                                init_Q(sol.init_Q, dpw.mdp, s, a))
        end
    else
        lock(snode.s_lock) do;
            !isempty(children(snode)) && return
            for a in actions(dpw.mdp, s)
                insert_action_node!(tree, snode, a,
                                    init_N(sol.init_N, dpw.mdp, s, a),
                                    init_Q(sol.init_Q, dpw.mdp, s, a))
            end
        end
    end

    sanode, q_logprob = lock(snode.s_lock) do; sample_sanode_UCB(tree, snode, sol.exploration_constant); end
    a = sanode.a_label
    w_node = compute_IS_weight(q_logprob, a, use_prior ? actions(dpw.mdp, s) : nothing) 
    w = w + w_node

    # State progressive widening.
    new_node = false
    transitioned = lock(sanode.a_lock) do
        !((sol.enable_state_pw && n_a_children(sanode) <= sol.k_state * n(sanode)^sol.alpha_state) || n_a_children(sanode) == 0) && return false

        sp, r = @gen(:sp, :r)(dpw.mdp, s, [a, w], dpw.rng)

        spnode = lock(tree.state_nodes_lock) do; insert_state_node!(tree, sp); end
        spnode_label = spnode.s_label
        new_node = (n_children(spnode) == 0)
        push!(sanode.transitions, (spnode_label, r))

        if !(spnode_label in sanode.unique_transitions)
            push!(sanode.unique_transitions, spnode_label)
            sanode.n_a_children += 1
        end
        return true
    end
    if !transitioned
        spnode_label, r = lock(sanode.a_lock) do; rand(dpw.rng, sanode.transitions); end
        spnode = lock(tree.state_nodes_lock) do; tree.state_nodes[spnodee_label]; end
    end

    if new_node
        q = r + discount(dpw.mdp) * estimate_value(dpw.solved_estimate, dpw.mdp, sp, w, d - 1)
    else
        q = r + discount(dpw.mdp) * simulate(dpw, spnode, w, d - 1)
    end

    function backpropagate(snode::DPWStateNode, sanode::DPWActionNode, q::Float64)
        snode.total_n += 1
        sanode.n += 1
        sanode.q += (q - sanode.q) / sanode.n
        delete!(snode.a_selected, sanode.a_label)
    end
    lock(snode.s_lock) do; backpropagate(snode, sanode, q); end

    return q
end


function sample_sanode(tree::PDPWTree, snode::DPWStateNode)
    all_Q = [q(child) for child in children(snode)]
    sanode, q_logprob = select_action(tree.children[snode], all_Q)
    return sanode, q_logprob
end


function sample_sanode_UCB(tree::PDPWTree, snode::DPWStateNode, c::Float64, virtual_loss::Float64=0.0)
    all_UCB = []
    ltn = log(total_n(snode))
    for child in children(snode)
        n = n(child)
        q = q(child)
        if (ltn <= 0 && n == 0) || c == 0.0
            UCB = q
        else
            UCB = q + c * sqrt(ltn / n)
        end

        vloss = (child.a_label in snode.a_selected ? virtual_loss : 0.0)
        UCB -= vloss

        @assert !isnan(UCB) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
        @assert !isequal(UCB, -Inf)
        
        push!(all_UCB, UCB)
    end
    sanode, q_logprob = select_action(children(snode), all_UCB)
    push!(snode.a_selected, sanode.a_label)
    return sanode, q_logprob
end
