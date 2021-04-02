"""
Update all entities in scene without noise
"""
function update_noiseless!(ego::Entity, scene::Scene)
    for ent in scene
        if ent.id==ego.id
            continue
        end
        scene[ent.id] = ent
    end
end