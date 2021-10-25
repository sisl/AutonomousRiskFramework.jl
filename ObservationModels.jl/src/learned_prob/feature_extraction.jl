"""
Function to create neural network features from scene
"""
function extract_feature(ego::Entity, scene::Scene)
    n_feat = 5
    n_err = 3
    feature = Array{Float32}(undef, n_feat*length(scene))
    error = Array{Float32}(undef, n_err*length(scene))
    ego_pos = posg(ego)
    ego_noise = AdversarialDriving.noise(ego)
    for ent in scene
        if ent.id == ego.id
            ent_pos = posg(ent)
            ent_noise = AdversarialDriving.noise(ent)
            feature[(ent.id-1)*n_feat + 1] = (ent_pos.x)/100.   # x-coord ego
            feature[(ent.id-1)*n_feat + 2] = (ent_pos.y)/10.   # y-coord ego
            feature[(ent.id-1)*n_feat + 3] = sin(ent_pos.θ)        # heading ego
            feature[(ent.id-1)*n_feat + 4] = cos(ent_pos.θ)        # heading ego
            feature[(ent.id-1)*n_feat + 5] = (vel(ent))/20.   # velocity ego  

            error[(ent.id-1)*n_err + 1] = ent_noise.pos.x  # x-noise ego
            error[(ent.id-1)*n_err + 2] = ent_noise.pos.y  # y-noise ego
            error[(ent.id-1)*n_err + 3] = ent_noise.vel  # vel-noise ego
        else
            ent_pos = posg(ent)
            ent_noise = AdversarialDriving.noise(ent)
            feature[(ent.id-1)*n_feat + 1] = (ent_pos.x - ego_pos.x)/100.   # x-coord relative
            feature[(ent.id-1)*n_feat + 2] = (ent_pos.y - ego_pos.y)/10.   # y-coord relative
            feature[(ent.id-1)*n_feat + 3] = sin(ent_pos.θ - ego_pos.θ)        # heading relative
            feature[(ent.id-1)*n_feat + 4] = cos(ent_pos.θ - ego_pos.θ)        # heading relative
            feature[(ent.id-1)*n_feat + 5] = (vel(ent) - vel(ego))/20.   # velocity relative

            error[(ent.id-1)*n_err + 1] = ent_noise.pos.x  # x-noise relative
            error[(ent.id-1)*n_err + 2] = ent_noise.pos.y  # y-noise relative
            error[(ent.id-1)*n_err + 3] = ent_noise.vel  # vel-noise relative
        end
    end
    (feature, error)
end

"""
Preprocess input scenes (labels from noise) for training/testing
"""
function preprocess_data(egoid::Int, scenes::Vector{Scene{T}}) where T
    feat = []
    y = []
    for scene in scenes
        for veh in scene
            if veh.id==egoid
                (_feat, _y) = extract_feature(veh, scene)
                push!(feat, _feat)
                push!(y, _y)
                break
            end
        end
    end
    
    hcat(feat...), hcat(y...)
end

# postprocess neural network outputs 
function postprocess_data!(pi, mu, sigma; n_comp=2)
    for i in 1:2
        pi[1 + (i-1)*n_comp:i*n_comp, :] = softmax(pi[1 + (i-1)*n_comp:i*n_comp, :])
    end 
    # outs[1:sig_offset, :] = (outs[1:sig_offset, :] .- 0.5)*40
    # outs[sig_offset+1:end, :] = (exp.(outs[sig_offset+1:end, :]/2))
end
