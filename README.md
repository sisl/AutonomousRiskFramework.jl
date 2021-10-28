# Autonomous Vehicle Risk Assessment Framework

[![Report](https://img.shields.io/badge/research-report-2d716f.svg)](http://web.stanford.edu/~mossr/pdf/Autonomous_Vehicle_Risk_Assessment.pdf)
[![Build Status](https://github.com/sisl/AutonomousRiskFramework/actions/workflows/CI.yml/badge.svg)](https://github.com/sisl/AutonomousRiskFramework/actions/workflows/CI.yml)

**Note:** Julia v1.5+ is recommended for AutomotiveSimulator and POMDPStressTesting.

```julia
julia install.jl
```

## Example

```julia
using RiskSimulator

system = IntelligentDriverModel()
scenario = get_scenario(MERGING)
planner = setup_ast(sut=system, scenario=scenario)

search!(planner)

fail_metrics = failure_metrics(planner)
α = 0.2 # risk tolerance
risk_metrics = risk_assessment(planner, α)
risk = overall_area(planner, α=α)
```


## Related packages
- **Adaptive stress testing**:
    - https://github.com/sisl/POMDPStressTesting.jl
- **Signal temporal logic computation graphs**:
    - https://github.com/StanfordASL/stlcg
- **Automotive simulator**:
    - https://github.com/sisl/AutomotiveSimulator.jl (note AutomotiveDrivingModels.jl is deprecated)
    - **Driving visualizations**:
        - https://github.com/sisl/AutomotiveVisualization.jl (note AutoViz.jl is deprecated)
- **Adversarial driving**:
    - https://github.com/sisl/AdversarialDriving.jl
- **Interpretability for validation**:
    - https://github.com/sisl/InterpretableValidation.jl
    - **Defining expression rules (for STL)**:
        - https://github.com/sisl/ExprRules.jl
- **POMDPs framework**:
    - https://github.com/JuliaPOMDP/POMDPs.jl


## Publications

- See [PUBLICATIONS.md](https://github.com/sisl/AutonomousRiskFramework/blob/master/PUBLICATIONS.md)


## CARLA Installation

- See [CARLAIntegration](https://github.com/sisl/AutonomousRiskFramework/tree/master/CARLAIntegration/adversarial_carla_env).

## Code style

- See: https://github.com/invenia/BlueStyle


## Contacts
- Stanford Intelligent Systems Laboratory (SISL)
    - Robert Moss: [mossr](https://github.com/mossr)
    - Kyu-Young Kim: [kykim0](https://github.com/kykim0)
- Navigation and Autonomous Vehicles Laboratory (NAV Lab)
    - Shubh Gupta: [shubhg1996](https://github.com/shubhg1996)
- Stanford Autonomous Systems Laboratory (ASL)
    - Karen Leung: [karenl7](https://github.com/karenl7)
    - Robert Dyro: [rdyro](https://github.com/rdyro)
