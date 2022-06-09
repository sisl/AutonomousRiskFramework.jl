using Revise
using AVExperiments

cd(@__DIR__)

leaf_noise = true

for seed in [0xC0FFEE, 0xFACADE, 0x0FF1CE]
    for use_tree_is in [true, false]
        for agent in [WorldOnRails, NEAT]
            config = ExperimentConfig(
                seed=seed,
                agent=agent,
                N=500,
                use_tree_is=use_tree_is,
                leaf_noise=leaf_noise,
            )

            generate_dirname!(config)
            @info "Running: $(config.dir)"
            # run_carla_experiment(config)
        end
    end
end
