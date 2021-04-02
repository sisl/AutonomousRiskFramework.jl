"""
Generate samples from noise distributions and fit probability model of noise
"""
function sample_and_fit(ego::Entity, scene::Scene, params::DistParams)
    fit_distributions = Dict{Int64, ContinuousMultivariateDistribution}()
    for ent in scene
        if ent.id==ego.id
            noise = Array{Noise}(undef, params.num_samples)
            for i in 1:params.num_samples
                sensor_noise = [rand(params.gps_range_noise) for i in 1:length(params.satpos)]
                noise[i] = estimate_gps_noise(ego, scene, params.buildingmap, params.satpos, sensor_noise)
            end
            fit_distributions[ent.id] = Distributions.fit(MvNormal, noise)
        else
            noise = Array{Noise}(undef, params.num_samples)
            for i in 1:params.num_samples
                sensor_noise = [rand(params.rb_range_noise), rand(params.rb_bearing_noise)]
                noise[i] = estimate_rb_noise(ego, ent, scene, sensor_noise)
            end
            fit_distributions[ent.id] = Distributions.fit(MvNormal, noise)
        end
    end
    fit_distributions
end