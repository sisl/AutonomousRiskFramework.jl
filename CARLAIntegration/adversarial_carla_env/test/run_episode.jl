using Distributed

if nprocs() <= 1
    procid = first(addprocs(1))
    @show procid
end

@everywhere begin
    using Revise
    using POMDPs
    using POMDPGym
    using PyCall
    pyimport("adv_carla")

    function main()
        sensors = [
            Dict(
                "id" => "GPS",
                "lat" => Dict("mean" => 0, "std" => 0.0001, "upper" => 10, "lower" => -10),
                "lon" => Dict("mean" => 0, "std" => 0.0001, "upper" => 10, "lower" => -10),
                "alt" => Dict("mean" => 0, "std" => 0.00000001, "upper" => 0.0000001, "lower" => 0),
            ),
        ]

        params = (sensors=sensors, seed=0xC0FFEE, scenario_type="Scenario2")
        mdp = GymPOMDP(Symbol("adv-carla"); params...)
        env = mdp.env

        # env.pyenv.hardreset(; params...)
        # env.done = false
        # reset!(env) # rest already called during the GymPOMDP constructor

        σ = 12 # noise variance
        while !env.done
            action = σ*rand(3)
            reward, obs, info = step!(env, action)
            render(env)
        end
        close(env)
        println("Total reward of $(env.total_reward)")
        return env.total_reward
    end
end # @everywhere

@info remotecall_fetch(main, procid)
rmprocs(procid)