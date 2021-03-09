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

### Non-Goals

We will not attempt to manage interactions with github, e.g. creating the repository for the user or uploading it. We will also not do much to help users set up AWS/GCP instances. We will not (easily) support upgrading basic scripts after the user has made some changes (this might necessitate using branches in the future). We won't support windows natively (though the zip file should work fine). 

### Future Goals

* Use github command line tool to interface with git making it easy to set up tracking repo
* All the stuff in non-goals above.


## Detailed Design


The system will be composed of a number of components that all work together to produce our desired output. The user will say something like "create me a scaffolding for a project called P1 that has a python commandline interface, and web interface that can be deployed to AWS lambda", and we will send the user back a zip file which has all the appropriate files with user input substituted in.

Code for the system will be in `repo/py`.


In Phase I, we will provide the following features:
* A git repository with the initial commit already set up
* A python venv setup, with a basica requirements.txt
* A python library file where you'll put your main code/logic (filled out with a simple example)
* A python command-line interface which will allow you to call functions in your library file
* A python/flask website which will provide a web interface for the library
* A simple unittest setup that will test some code in the library file
* A makefile that will allow creation of a Docker image, running tests, setting up the virtual environment, and potentially even deploying the flask code to AWS serverless.
* Various documentation files

To use the system, the user will proceed to a website, and submit a form with customization parameters. The site will then provide a zip file with the above features. The user will then expand the zip file and proceed to run a configuration script that will push the code to github and set up virtual environemnts. The user will then modify code in the python files to suit his/her needs.  Unnecessary files can be deleted. In future phases, the user will be able to select the subset of features from above required so as to not have to delete files.

### File Templates

Every user request will generate a unique set of files. We will compute these by creating certain code template files, which will then be rendered using data provided by the user. We will use the Jinja2 templating system. In Phase I, all files will be nested under the following directory: `repo/py/data/code_templates/py3_expansion/`. Under here, the file system structure will be replicated into the eventual zip file.  In future phases, we will have a manifest for each plugin area with a bit of logic, i.e. which choices are incompatible, and which ones rely on other ones. This will be loaded and used to render the user form.

### Template expansion

We will write python code to expand the file templates against user input. This will be tested. The following parameters will be provided:
* language_name = {py3, anaconda, R, etc.}
* program_name = 
* author_name = name, can include spaces but no other whitespace TODO: utf-8?
* author_email = validated email
* repository_name = github_repo_name
* selected_features = dictionary of user selected features. for now {docker, serverless, unittest} are options.

This will generate files in a filesystem tree. We will ideally use a temporary directory or in-memory filesystem to perform this expansion.  


### Git repo creation

We will use `gitpython` to create a git repository in python. We will create a commit using the filesystem described above. In the future we should be able to do this directly by creating blobs.  We need to handle +x mode appropriately; in phase I, we can just copy the +x state from the template file, but in the future we should probably control this via a manifest file.

### Web interface

We will write a `flask`-based webapp that can be deployed to aws lambda. In Phase I, the webapp will render a landing page which will allow the user to put in values for the above template params. When the form is submitted, a different handler will respond with a `application/zip` file to be downloaded.

In future phases, the form will be rendered based on data gathered from manifest files in the appropriate directories, making the deployment of plugins easier. 


### User requirements

The user is expected to be a smart person with some experience coding, but limited software engineering expertise. Thus, the system should be relatively simple, defaults should set the user up for success, and documentation should explain clearly how to connect the downloaded repo to a github repository.

### What APIs will you use/change?
We will use serverless to abstract AWS/GCP command-line apis.


### Throughput/latency/cost/efficiency concerns?
none.

### Data validation/what are potential error states?

All user input will be validated. For instance all initial user inputs will conform to the regex `[0-9A-Za-z_\]+` to prevent any RCE


### Logging/monitoring/observability
We won't worry about this for now

### Security/Privacy
We will only collect user emails for contacting the user in the future. We should probably have a security page and a privacy policy on the web page.

### What will you test?
We're going to test the template framework extensively to ensure that code that is created works.

## Third Party dependencies

## Work Estimates
Phase I should take approximately 3 hours to write, other phases TBD

## Related Work?

Java spring has a similar [generator](https://start.spring.io).

