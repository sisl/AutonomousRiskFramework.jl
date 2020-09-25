# RiskSimulator.jl

Main framework for autonomous vehicle risk assessment.

## Installation

```bash
cd AutonomousRiskFramework/RiskSimulator.jl/
julia
```
then "develop" this package via:
```julia
] dev .
```
where `]` gets you into the Pkg-mode.


## Development
Install the Revise.jl package:
```julia
] add Revise
```
Then load Revise before you load the RiskSimulator package. This way, any changes to the RiskSimulator package will be automatically re-compiled:
```julia
using Revise
using RiskSimulator
```


## Testing
Open Julia, and run:
```julia
] test RiskSimulator
```
which runs the `test/runtests.jl` file.