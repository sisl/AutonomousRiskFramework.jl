"""
Update all entities in scene with Gaussian noise
"""
function update_gaussian_noise!(ego::Entity, scene::Scene; sigma=1.0)
    for (i, ent) in enumerate(scene)
        if ent.id==ego.id
            continue
        end
        noise = Noise(pos=VecE2(sigma*randn(), sigma*randn()))  # Position noise only
        # noise = Noise(pos=VecE2(sigma*randn(), sigma*randn()), vel=sigma*randn()) # Pos + velocity noise
        scene[i] = Entity(update_veh_noise(ent.state, noise), ent.def, ent.id)
    end
end