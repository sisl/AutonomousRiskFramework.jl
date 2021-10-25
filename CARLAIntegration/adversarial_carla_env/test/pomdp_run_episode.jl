using POMDPs
using POMDPGym
using PyCall
pyimport("adv_carla")

mdp = GymPOMDP(Symbol("adv-carla"))
s = rand(initialstate(mdp))
total_reward = 0
while !isterminal(mdp, s)
    global s, total_reward
    action = 100 .* rand(2)
    s, o, r = gen(mdp, s, action) # TODO: info?
    total_reward += r
end
close(mdp.env)

println("Total reward of $total_reward")
if mdp.env.info.get("collision")
    println("Found collision failure!")
end
