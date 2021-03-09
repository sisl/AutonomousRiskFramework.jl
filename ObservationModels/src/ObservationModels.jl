module ObservationModels
    using Distributions
    using Parameters
    using Random
    using LinearAlgebra
    using Statistics
    using AdversarialDriving
    using Distributed
    using AutomotiveSimulator
    using Flux
    using Zygote
    using CUDA

    export Landmark, SensorObservation
    include("structs.jl")

    include("utils.jl")

    export Satellite, GPSRangeMeasurement, measure_gps, GPS_fix, update_gps_noise!
    include("gps.jl")

    export RangeAndBearingMeasurement, update_rb_noise!
    include("range_bearing.jl")

    export INormal_Uniform, INormal_GMM, Fsig_Normal

    include(joinpath("distributions", "inormal_uniform.jl"))
    include(joinpath("distributions", "inormal_gmm.jl"))
    include(joinpath("distributions", "fixedsig_normal.jl"))

    export DistParams, gps_distribution_estimate, traj_logprob, simple_distribution_fit, calc_logprobs
    include("estimate.jl")
end