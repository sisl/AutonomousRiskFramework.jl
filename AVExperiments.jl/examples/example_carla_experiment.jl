using Revise # TODO: Remove
using AVExperiments

config = ExperimentConfig(
    seed=0xC0FFEE,
    agent=GNSS,
    N=1,
    dir="results_gnss",
    use_tree_is=true,
    leaf_noise=true,
    resume=false,
)

generate_dirname!(config)

results = run_carla_experiment(config)
