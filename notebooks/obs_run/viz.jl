# Instructions for rendering the noisy pedestrian
function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, ped::Entity{NoisyPedState, VehicleDef, Int64})
    reg_ped = Entity(ped.state.veh_state, ped.def, ped.id)
    add_renderable!(rendermodel, FancyPedestrian(ped=reg_ped))
    noisy_ped = Entity(noisy_entity(ped, ad_mdp.roadway).state.veh_state, ped.def, ped.id)
    ghost_color = weighted_color_mean(0.3, colorant"blue", colorant"white")
    add_renderable!(rendermodel, FancyPedestrian(ped=noisy_ped, color=ghost_color))
    return rendermodel
end