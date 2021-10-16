"""
Creates a copy of the vehicle state with the new specified noise
"""
update_veh_noise(s::NoisyPedState, noise::Noise) = NoisyPedState(s.veh_state, noise)
update_veh_noise(s::BlinkerState, noise::Noise) = BlinkerState(s.veh_state, s.blinker, s.goals, noise)


"""
Probability of gaussian distribution
"""
function gaussian_distribution(y, μ, σ)
    return 1 ./ ((sqrt(2π).*σ)).*exp.(-0.5((y .- μ)./σ).^2)
end;

function log_gaussian_distribution(y, μ, σ)
    term1 = 0.5((y .- μ)./σ).^2
    term1 = clamp(term1, 1e-3, 1e10)
    return -0.5*log(2π).-log(σ).-term1
end;