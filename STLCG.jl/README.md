# STLCG.jl

Defines STL operators and formulas, and computes robustness using computation graphs.

## Installation

```bash
cd AutonomousRiskFramework/STLCG.jl/
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
Then load Revise before you load the STLCG package. This way, any changes to the STLCG package will be automatically re-compiled:
```julia
using Revise
using STLCG
```


## Testing
Open Julia, and run:
```julia
] test STLCG
```
which runs the `test/runtests.jl` file.