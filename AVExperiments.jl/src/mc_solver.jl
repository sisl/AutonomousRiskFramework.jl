using Infiltrator
using POMDPSimulators
using Distributions
import POMDPPolicies:FunctionPolicy

function mc_simulate(mdp, sensors)
    amin = reduce(vcat, [[Float32(sensor[k]["lower"]) for k in filter(!=("id"), collect(keys(sensor)))] for sensor in sensors])
    amax = reduce(vcat, [[Float32(sensor[k]["upper"]) for k in filter(!=("id"), collect(keys(sensor)))] for sensor in sensors])
    
    σ_scale = 1
    μ = reduce(vcat, [[Float32(sensor[k]["mean"]) for k in filter(!=("id"), collect(keys(sensor)))] for sensor in sensors])
    σ = σ_scale * reduce(vcat, [[Float32(sensor[k]["std"]) for k in filter(!=("id"), collect(keys(sensor)))] for sensor in sensors])

    # TODO: Conditional exposure policy (or Normal from μ and σ)
    rand_policy = FunctionPolicy(s -> Float32.(rand.(Distributions.Uniform.(amin, amax))))

    r = simulate(RolloutSimulator(), mdp, rand_policy)
    cost = mdp.dataset[end]["cost"]
    return cost

end


function run_mc_solver(mdp, sensors)
    @info "Running Monte Carlo (MC) noise at leaf to collect a cost..."
    mdp_info = InfoCollector(mdp, extract_info)
    cost = mc_simulate(mdp_info, sensors)
    close(mdp.env) # Important!
    return cost, mdp_info.dataset
end
