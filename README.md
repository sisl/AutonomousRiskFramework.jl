# Autonomous Vehicle Risk Assessment Framework

**Note:** Julia v1.3+ is required for AutomotiveSimulator.

Change directory to each ".jl" folder, then within `julia` run:
```julia
] dev .
```

Without cloning the entire repository, each sub-package can be added via:
```julia
using Pkg
Pkg.develop(PackageSpec(url="<PATH_TO_DIR>/AutonomousRiskFramework/RiskSimulator.jl"))
Pkg.develop(PackageSpec(url="<PATH_TO_DIR>/AutonomousRiskFramework/STLCG.jl"))
```

## Notebooks
[![Toy problem](https://img.shields.io/badge/pluto-running%20example-8c1515)](./notebooks/automotive_notebook.jl)

Open Julia, install Pluto via `] add Pluto` (where `]` gets to the package manager). Then from the Julia prompt, run `using Pluto; Pluto.run()` and open the notebook located in `notebooks/`. 


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

## Related publications
- **Adaptive stress testing (AST)**:
    - **AST formulation**:
        - [M. Koren, A. Corso, M. J. Kochenderfer, "The Adaptive Stress Testing Formulation"](https://arxiv.org/abs/2004.04293) (Quick 2-pager)

        - [R. Lee, O. J. Mengshoel, A. Saksena, R. Gardner, D. Genin, J. Silbermann, M. Owen, M. J. Kochenderfer, "Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning", Arxiv, 2020](https://arxiv.org/abs/1811.02188)

    - **AST applications**:
        - [M. Koren, S. Alsaif, R. Lee, M. J. Kochenderfer, "Adaptive Stress Testing for Autonomous Vehicles", IEEE Intelligent Vehicles Symposium (IV), 2018](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8500400)

        - [A. Corso, P. Du, K. Driggs-Campbell, M. J. Kochenderfer, "Adaptive Stress Testing with Reward Augmentation for Autonomous Vehicle Validation", IEEE Intelligent Transportation Systems Conference, 2019](https://arxiv.org/abs/1908.01046)

        - [R. Lee, M. Kochenderfer, O. Mengshoel, G. Brat, and M. Owen, "Adaptive stress testing of airborne collision avoidance systems", Digital Avionics Systems Conference (DASC), 2015](https://ieeexplore.ieee.org/document/7311613)
    - **AST tools**:
        - [R. J. Moss, "POMDPStressTesting.jl: Adaptive Stress Testing for Black-Box Systems", JOSS, 2020.](https://github.com/sisl/POMDPStressTesting.jl/blob/master/joss/paper.pdf)

- **Interpretability**:
    - [A. Corso, M. J. Kochenderfer, "Interpretable Safety Validation for Autonomous Vehicles", Arxiv, 2020](https://arxiv.org/abs/2004.06805)

- **STL with computation graphs (stlcg)**:
    - [K. Leung, N. Aréchiga, and M. Pavone, "Back-propagation through STL specifications: Infusing logical structure into gradient-based methods," in Workshop on Algorithmic Foundations of Robotics, Oulu, Finland, 2020.](https://arxiv.org/abs/2008.00097)

    - [J. DeCastro, K. Leung, N. Aréchiga, and M. Pavone, "Interpretable Policies from Formally-Specified Temporal Properties,"" in Proc. IEEE Int. Conf. on Intelligent Transportation Systems, Rhodes, Greece, 2020.](http://asl.stanford.edu/wp-content/papercite-data/pdf/DeCastro.Leung.ea.ITSC20.pdf)

    - [K. Leung, N. Arechiga, and M. Pavone, "Backpropagation for Parametric STL," in IEEE Intelligent Vehicles Symposium: Workshop on Unsupervised Learning for Automated Driving, Paris, France, 2019.](http://asl.stanford.edu/wp-content/papercite-data/pdf/Leung.Arechiga.ea.ULAD19.pdf)

- **Surveys**:
    - [A. Corso, R. J. Moss, M. Koren, R. Lee, M. J. Kochenderfer, "A Survey of Algorithms for Black-Box Safety Validation", Submitted to JAIR, Arxiv, 2020](https://arxiv.org/abs/2005.02979)

## Contacts
- Stanford Intelligent Systems Laboratory (SISL)
    - Robert Moss: [mossr](https://github.com/mossr)
- Stanford Autonomous Systems Laboratory (ASL)
    - Karen Leung: [karenl7](https://github.com/karenl7)
    - Robert Dyro: [rdyro](https://github.com/rdyro)
- Navigation and Autonomous Vehicles Laboratory (NAV Lab)
    - Shubh Gupta: [shubhg1996](https://github.com/shubhg1996)