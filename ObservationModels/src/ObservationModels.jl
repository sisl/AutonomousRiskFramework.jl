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
    using CUDA

    export Landmark, SensorObservation
    include("structs.jl")

    export Satellite, GPSRangeMeasurement, measure_gps, GPS_fix
    include("gps.jl")

    export INormal_Uniform, INormal_GMM, Fsig_Normal

    include(joinpath("distributions", "inormal_uniform.jl"))
    include(joinpath("distributions", "inormal_gmm.jl"))
    include(joinpath("distributions", "fixedsig_normal.jl"))

    export DistParams, gps_distribution_estimate, traj_logprob, simple_distribution_fit, set_logprob
    include("estimate.jl")
end