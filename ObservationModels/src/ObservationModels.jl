module ObservationModels
    using Distributions
    using Parameters
    using Random
    using LinearAlgebra
    using Statistics
    using AdversarialDriving
    using AutomotiveVisualization
    using Distributed
    using AutomotiveSimulator
    using Flux
    using Zygote
    using CUDA
    using Printf
    using Cairo

    export Landmark, SensorObservation, Building, position, noise, BuildingMap
    include("structs.jl")

    export update_veh_noise
    include("utils.jl")

    export Satellite, GPSRangeMeasurement, measure_gps, GPS_fix, update_gps_noise!, estimate_gps_noise
    include(joinpath("sensor_models", "gps", "structs.jl"))
    include(joinpath("sensor_models", "gps", "measure_gps.jl"))
    include(joinpath("sensor_models", "gps", "gps_positioning.jl"))
    include(joinpath("sensor_models", "gps", "update_entity.jl"))

    export update_noiseless!
    include(joinpath("sensor_models", "noiseless", "update_entity.jl"))

    export update_gaussian_noise!
    include(joinpath("sensor_models", "gaussian", "update_entity.jl"))

    export RangeAndBearingMeasurement, measure_range_bearing, RB_fix, update_rb_noise!, estimate_rb_noise
    include(joinpath("sensor_models", "range_bearing", "structs.jl"))
    include(joinpath("sensor_models", "range_bearing", "measure_rb.jl"))
    include(joinpath("sensor_models", "range_bearing", "rb_positioning.jl"))
    include(joinpath("sensor_models", "range_bearing", "update_entity.jl"))
    
    export INormal_Uniform, INormal_GMM, Fsig_Normal

    include(joinpath("distributions", "inormal_uniform.jl"))
    include(joinpath("distributions", "inormal_gmm.jl"))
    include(joinpath("distributions", "fixedsig_normal.jl"))
    include(joinpath("distributions", "noise_distribution.jl"))

    export DistParams, sample_and_fit, extract_feature, preprocess_data, postprocess_data, construct_mdn, MDNParams, train_nnet!, evaluate_logprob
    include(joinpath("learned_prob", "structs.jl"))
    include(joinpath("learned_prob", "distribution_fit.jl"))
    include(joinpath("learned_prob", "feature_extraction.jl"))
    include(joinpath("learned_prob", "mixture_density_network.jl"))
end