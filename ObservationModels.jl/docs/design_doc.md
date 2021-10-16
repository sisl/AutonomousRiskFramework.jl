# Adaptive Stress Testing with Sensor Observation Errors

## Overview

Safety validation of autonomous vehicle software is done through stress testing on simulations. Adaptive Stress Testing (AST) [] is a framework that has been previously proposed to automate the task of finding the most likely path to a failure scenario in an autonomous driving setting using reinforcement learning.   

In this project, we will extend the capabilities of AST to find failures in environments where the driving behavior depends on observations from multiple simulated sensors. By simulating sensor observations for multiple driving trajectories in the environment, we will learn a model of the probability distribution of the disturbances in agent states that result from the observations. Finally, we will utilize the learnt probability distribution in AST to find failures. Next, the failure-causing driving trajectories obtained from AST will be used to simulate new sensor observations and the process would be repeated till convergence.

We will implement this framework in Julia and develop interfaces to query high-fidelity simulators (CARLA, hardware-in-the-loop GNSS) for sensor observations and to execute the failure search. We will develop the probability distribution learning codebase using `Flux.jl` and the failure search codebase using `AderversarialDriving.jl` , `POMDPStressTesting.jl` and `CrossEntropyMethod.jl`.  


## Background: 
 
This section includes background information on the types of sensors and the AST framework.

### Sensor Observations

The project requires interacting with different types of sensor observations. Therefore, we describe each kind of observation that we utilize in this section.

#### GNSS sensor

Global Navigation Satellite System (GNSS) collectively describes all the satellite-based navigation systems available throughout the world, including as the USA's Global Positioning System (GPS). GNSS is one of the most common sensors available on an autonomous vehicle, and therefore we include it in our analysis. 

The GNSS sensor is used to acquire position coordinates of a vehicle anywhere on the Earth. These coordinates are obtained by processing RF signals from satellites orbiting the Earth to determine their time-of-flight, and performing least-squares optimization to determine the position. In ideal conditions, GNSS achieves positioning accuracies upto ~1m. However, this is far from the case in urban environments. This is because urban environments contain several tall structures (buildings, trees, etc.) that occlude or reflect GNSS signals coming from space. This severely deteriorates GNSS performance in these environments. Furthermore, the resulting error probability distributions are non-Gaussian with significant heavy tails because the signals are subject to plenty of error sources with small probabilities.

Therefore, we consider errors at the signal level to accurately model the resulting position errors. GNSS signal consists of three components:
* _Carrier Phase_: Radio frequency sinusoid signal
* _Code phase_: Pseudo-random sequences of 0s and 1s known to the receiver. Since the start and end of the signal is unique, this allows the receiver to determine the time-of-flight of the signal.
* _Navigation data_: A binary-coded message providing information on the satellite ephemeris (Keplerian elements or satellite position and velocity), clock bias parameters, almanac (with a reduced accuracy ephemeris data set), satellite health status, and other complementary information.

One the signals are processed, the time-of-flight of the signals is determined using the source time information in the navigation data and reception time. This time of flight is then divided by the speed of the signal to estimate the ranging distance between the receiver and the satellite. This distance is commonly termed as the pseudo-range
$$\rho = \|x_{rec} - x_{sat}\| + c|\delta t_{rec}-\delta t_{rec}| + \epsilon + b$$

### Adaptive Stress Testing

The original codebase for AST is available through Julia packages. Additionally, the codebase includes a simple driving example with one car and one pedestrian on a crosswalk to demonstrate the performance of AST.

### Current Goals

This project comprises of several parts, and therefore we will tackle it in multiple phases:

#### Phase I: Simulated GNSS sensor observations

In Phase I, we will use the simple vehicle and pedestrian crosswalk scenario available in the AST codebase. We will simulate simple sensor observations from a GNSS sensor on top of the available simulation. The sensor observations will be pseudo-range measurements from 5 different satellites at fixed points in the sky. The corresponding probability distribution of vehicle positions will be learned as a function of the vehicles true position in the environment using a simple 2-layer feedforward neural network. I will then test AST by varying the structure of errors in the GNSS observations for 3 cases:
* No satellite blockage or reflection
* One satellite blocked but no reflection
* One signal reflected

#### Phase II: Range and bearing sensor observations of the pedestrian
Phase II will build on Phase I by adding sensor observations for the pedestrian as seen from the autonomous vehicle along with the GNSS observations. We will simulate range and bearing measurements for this task. The scenario will be evaluated for 3 cases:
* Error in range but not in bearing
* Error in both range and bearing
* Error in range and bearing, and reflection in one satellite signal

#### Phase III: Evaluating probability distribution learning techniques

In Phase III, we will evaluate learning methods for the probability distribution other than the feedforward neural network. We will consider 3 methods:
* Fixed Normal distribution fitting
* Feed-forward neural network (Normal distribution conditional on position)
* Gaussian Mixture model conditional on position

### Additional Goals
* We will prepare the code as a Julia package which can be installed and used separately.
* We will add tests for various code components within the Julia framework
* We will add detailed instructions and examples for installing and using the package

### Non-Goals

We will not attempt to manage interactions with github, e.g. creating the repository for the user or uploading it. We will also not do much to help users set up AWS/GCP instances. We will not (easily) support upgrading basic scripts after the user has made some changes (this might necessitate using branches in the future). We won't support windows natively (though the zip file should work fine). 

### Future Goals

* Use github command line tool to interface with git making it easy to set up tracking repo
* `Documenter.jl` Julia package for automatic documentation from docstrings
* Github actions and Github pages to host documentation.
* All the stuff in non-goals above.


## Third Party dependencies
The package will depend on `AdversarialDriving.jl` and `AutomotiveSimulator.jl` packages. 

