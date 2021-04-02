@with_kw struct DistParams
    num_samples::Int = 50 # Estimation samples
    satpos::Vector{Satellite{Float64}} = Satellite[]
    buildingmap::BuildingMap = BuildingMap()
    gps_range_noise::Sampleable = Normal(0.0, 1e-5)
    rb_range_noise::Sampleable = Normal(0.0, 1e-5)
    rb_bearing_noise::Sampleable = Normal(0.0, 1e-5)
end;

@with_kw struct MDNParams
    N_modes::Int = 2
    in_dims::Int = 8
    out_dims::Int = 6
    hidden_dims::Int = 20
    lr::Float64 = 0.01
    momentum::Float64 = 0.99
    batch_size::Int = 10
    N_epoch::Int = 10
end;

stop_gradient(f) = f()
Zygote.@nograd stop_gradient
