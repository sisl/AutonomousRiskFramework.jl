using Crux
using Flux
import POMDPPolicies:FunctionPolicy


function td3_solver(mdp, sensors)
    S = state_space(mdp)
    amin = reduce(vcat, [[Float32(sensor[k]["lower"]) for k in filter(!=("id"), collect(keys(sensor)))] for sensor in sensors])
    amax = reduce(vcat, [[Float32(sensor[k]["upper"]) for k in filter(!=("id"), collect(keys(sensor)))] for sensor in sensors])
    rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

    state_dim = first(S.dims)
    action_dim = length(amin)
    QSA() = ContinuousNetwork(Chain(Dense(state_dim+action_dim, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
    TD3_A() = ContinuousNetwork(Chain(Dense(state_dim, 64, relu), Dense(64, 64, relu), Dense(64, action_dim), x -> x .* 0.0000001f0), action_dim)

    on_policy = ActorCritic(TD3_A(), DoubleNetwork(QSA(), QSA()))
    off_policy = (S=S,
                  ΔN=30, # 300, 1000
                  N=200, # NOTE: was 30_000 (then 3000, 100)
                  buffer_size=Int(5e5),
                  buffer_init=1,
                  c_opt=(batch_size=100, optimizer=ADAM(1e-3)),
                  a_opt=(batch_size=100, optimizer=ADAM(1e-3)),
                  log=(fns=[],),
                  π_explore=GaussianNoiseExplorationPolicy(0.00000001f0))

    return TD3(; π=on_policy, off_policy...)
end


function run_td3_solver(mdp, sensors)
    @info "Running TD3 to collect costs..."
    mdp_info = InfoCollector(mdp, extract_info)
    π_train = solve(td3_solver(mdp_info, sensors), mdp_info)
    costs = convert(Vector{Real}, map(d->d["cost"], mdp_info.dataset))
    rewards = convert(Vector{Real}, map(d->d["reward"], mdp_info.dataset))
    close(mdp.env) # Important!

    α = 0.2
    risk_metrics = RiskMetrics(costs, α)
    cvar = risk_metrics.cvar

    return cvar, mdp_info.dataset
end
