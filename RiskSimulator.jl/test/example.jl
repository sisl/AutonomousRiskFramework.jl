using RiskSimulator

av1 = Vehicle()
av2 = Vehicle()
scene = Scenario()
sim = Simulator([av1, av2], scene)

simulate(sim)
evaluate(sim)