# Observation Models

A Julia package containing sensor observation models and state estimation tools. Currently supports [AdversarialDriving](https://github.com/sisl/AdversarialDriving.jl) and [AutomotiveSimulator](https://github.com/sisl/AutomotiveSimulator.jl)

For visualization code please see [AutomotiveVisualization](https://github.com/sisl/AutomotiveVisualization.jl).

# Interface
To use the sensor models and fitting tools provided in this package, the user has to first define an environment using the `AutomotiveSimulator` and `AdversarialDriving` interface outlined in [AutomotiveSimulator](https://github.com/sisl/AutomotiveSimulator.jl) and [AdversarialDriving](https://github.com/sisl/AdversarialDriving.jl).

Additional structures for constructing Landmarks and Buildings are provided.

* [`Landmark`](https://github.com/shubhg1996/ObservationModels.jl/blob/main/src/structs.jl)
* [`Building`](https://github.com/shubhg1996/ObservationModels.jl/blob/main/src/structs.jl)
* [`BuildingMap`](https://github.com/shubhg1996/ObservationModels.jl/blob/main/src/structs.jl)

The package interacts with `Scene` objects from the interface and leverages `Noise` objects in modeling. Functions ending with `!` may modify the `Scene` object in place.

# Sensor Models
Several sensor models are implemented.

#### Noiseless Sensor Model
* [`noiseless`](https://github.com/shubhg1996/ObservationModels.jl/tree/main/src/sensor_models/noiseless)

#### Gaussian Noise Sensor Model
* [`gaussian`](https://github.com/shubhg1996/ObservationModels.jl/tree/main/src/sensor_models/gaussian)

#### GPS Sensor Model (w/ signal reflection errors from buildings)
* [`gps`](https://github.com/shubhg1996/ObservationModels.jl/tree/main/src/sensor_models/gps)

#### Range-Bearing Sensor Model
* [`range_bearing`](https://github.com/shubhg1996/ObservationModels.jl/tree/main/src/sensor_models/range_bearing)

# Probability Model Fitting
Two probability fitting implementations are included in this package.

#### Sample-based Distribution Fitting
* [`distribution_fit`](https://github.com/shubhg1996/ObservationModels.jl/blob/main/src/learned_prob/distribution_fit.jl)
Additional Importance sampling distributions (other than the ones already in [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)) for fitting are implemented in `src/distributions`.
* `Fsig_Normal` is Normal distribution with fixed standard deviation.
* `INormal_GMM` is uses Gaussian Mixture Model distribution for sampling and Normal distribution for probability evaluation. 
* `INormal_Uniform` uses uniform distribution for sampling and Normal distribution for probability evaluation.

#### Mixture Density Network Fitting ([Ref](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf))
* [`mixture_density_network`](https://github.com/shubhg1996/ObservationModels.jl/blob/main/src/learned_prob/mixture_density_network.jl)

# Example

An example implementation of the building model is provided for a simple case:
* **Julia source**: [`examples/generate_buildings.jl`](https://github.com/shubhg1996/ObservationModels.jl/blob/main/examples/generate_buildings.jl)

An example implementation of the MDN training on simulated scenes is provided for a simple case:
* **Julia source**: [`examples/train_mdn.jl`](https://github.com/shubhg1996/ObservationModels.jl/blob/master/examples/train_mdn.jl)

# Installation 

Install `AdversarialDriving.jl`, `Vec.jl` and then the `ObservationModels.jl` package via:
```julia 
using Pkg
pkg"add https://github.com/sisl/AdversarialDriving.jl"
pkg"add https://github.com/sisl/Vec.jl.git"
pkg"add https://github.com/shubhg1996/ObservationModels.jl"
```

### Testing
To run the test suite, you can use the Julia package manager.
```julia
] test ObservationModels
```

---
Package maintained by Shubh Gupta: shubhg1996@stanford.edu
