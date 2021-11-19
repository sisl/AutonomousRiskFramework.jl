using Revise
using POMDPs
using POMDPGym
using PyCall
pyimport("adv_carla")

sensors = [
    Dict(
        "id" => "GPS",
        "lat" => Dict("mean" => 0, "std" => 0.0001, "upper" => 10, "lower" => -10),
        "lon" => Dict("mean" => 0, "std" => 0.0001, "upper" => 10, "lower" => -10),
        "alt" => Dict("mean" => 0, "std" => 0.00000001, "upper" => 0.0000001, "lower" => 0),
    ),
]

mdp = GymPOMDP(Symbol("adv-carla"), sensors=sensors, seed=0xC0FFEE, scenario_type="Scenario2")
env = mdp.env
# initial_obs = reset!(env) # rest already called during the GymPOMDP constructor
σ = 12 # noise variance
while !env.done
    action = σ*rand(3)
    reward, obs, info = step!(env, action)
    render(env)
end
close(env)
println("Total reward of $(env.total_reward)")
