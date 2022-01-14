using Crux
using Flux
import POMDPPolicies:FunctionPolicy


function td3_solver(mdp, sensors)
    S = state_space(mdp)
    amin = [Float32(sensors[1][k]["lower"]) for k in ["lat", "lon", "alt"]]
    amax = [Float32(sensors[1][k]["upper"]) for k in ["lat", "lon", "alt"]]
    rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

    state_dim = first(S.dims)
    action_dim = length(amin)
    QSA() = ContinuousNetwork(Chain(Dense(state_dim+action_dim, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
    TD3_A() = ContinuousNetwork(Chain(Dense(state_dim, 64, relu), Dense(64, 64, relu), Dense(64, action_dim), x -> x .* 0.0000001), action_dim)

    on_policy = ActorCritic(TD3_A(), DoubleNetwork(QSA(), QSA()))
    off_policy = (S=S,
                  ΔN=300,
                  N=1000, # NOTE: was 30_000 (then 3000, 100)
                  buffer_size=Int(5e5),
                  buffer_init=1,
                  c_opt=(batch_size=100, optimizer=ADAM(1e-3)),
                  a_opt=(batch_size=100, optimizer=ADAM(1e-3)),
                  log=(fns=[],),
                  π_explore=GaussianNoiseExplorationPolicy(0.00000001f0))

    return TD3(; π=on_policy, off_policy...)
end


# TODO: pass (mdp, s) check isterminal(mdp, s)
function extract_info(info)
    if info["done"] == true
        # TODO: info["cost"]
        data_point = info["rate"]
    else
        data_point = missing
    end
    return data_point
end


function run_ast_solver(mdp, sensors)
    @info "Running AST to collect costs..."
    mdp_info = InfoCollector(mdp, extract_info)
    @time π_train = solve(td3_solver(mdp_info, sensors), mdp_info)
    costs = convert(Vector{Real}, mdp_info.dataset)

    return costs
end
