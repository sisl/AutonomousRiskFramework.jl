# Autonomous Vehicle Risk Assessment Framework

[![Report](https://img.shields.io/badge/research-report-2d716f.svg)](http://web.stanford.edu/~mossr/pdf/Autonomous_Vehicle_Risk_Assessment.pdf)
[![Build Status](https://github.com/sisl/AutonomousRiskFramework/actions/workflows/CI.yml/badge.svg)](https://github.com/sisl/AutonomousRiskFramework/actions/workflows/CI.yml)

## Installation
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
risk_metrics = metrics(planner, α)
risk = overall_area(planner, α=α)
```

### CARLA experiment
[See adversarial CARLA environment instructions here.](./CARLAIntegration/adversarial_carla_env)


```julia
using AVExperiments

config = ExperimentConfig(
    seed=0xC0FFEE,
    agent=WorldOnRails,
    N=100,
    dir="results_wor",
    use_tree_is=true,
    leaf_noise=true,
    resume=false,
)

results = run_carla_experiment(config)
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

## Known issues

#### INTEL MKL ERROR: Library not loaded: @rpath/libiomp5.dylib
This error was observed on MacOS and appears to be related to Conda.
One solution that worked was to run

```julia
using Conda
Conda.rm("mkl")
Conda.add("nomkl")
```
See https://github.com/JuliaPy/PyPlot.jl/issues/315 for relevant discussions.


#### Python version conflicts
Some versions of Python e.g., 3.9 are incompatible with the framework as they do not support packages such as `pytorch` that are needed.
It is possible to switch to a working version of Python as follows:
```julia
using Conda
Conda.add("python=3.7.5")
```
However, note that if you were using an incompatible of Python before, you might have installed Python packages of versions
that can conflict with the new compatible version of Python as the packages remain under the Conda directory.
You may see error messages like the following if this is the case:
```
Package enum34 conflicts for:
pyopenssl -> cryptography[version='>=2.8'] -> enum34
cryptography -> enum34
brotlipy -> enum34
urllib3 -> brotlipy[version='>=0.6.0'] -> enum34
pyqt -> enum34
```
If so, you may need to remove the Conda directory to remove the packages and resintall them *after* setting Conda to use a correct version of Python i.e.,
```shell
rm -R ~/.julia/conda/  # Make sure this is okay to do in your case
```
then,
```julia
using Conda
Conda.add("python=3.7.5")
using RiskSimulator
```


## Contacts
- Stanford Intelligent Systems Laboratory (SISL)
    - Robert Moss: [mossr](https://github.com/mossr)
    - Kyu-Young Kim: [kykim0](https://github.com/kykim0)
- Navigation and Autonomous Vehicles Laboratory (NAV Lab)
    - Shubh Gupta: [shubhg1996](https://github.com/shubhg1996)
- Stanford Autonomous Systems Laboratory (ASL)
    - Karen Leung: [karenl7](https://github.com/karenl7)
    - Robert Dyro: [rdyro](https://github.com/rdyro)
