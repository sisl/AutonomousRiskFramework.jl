"""
Update all entities in scene with Gaussian noise
"""
function update_gaussian_noise!(ego::Entity, scene::Scene)
    # Predetermined constants
    sigma = 0.2

    for ent in scene
        if ent.id==ego.id
            continue
        end
        noise = Noise(pos=VecE2(sigma*randn(), sigma*randn()))
        scene[ent.id] = Entity(update_veh_noise(ent.state, noise), ent.def, ent.id)
    end
end