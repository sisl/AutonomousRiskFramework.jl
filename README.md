# Docker Container for Autonomous Risk Framework

This readme document is intended to lead through the installation of Docker and the necessary packages. 

## Installation of Docker and NVIDIA Docker on the Host Machine
To fully use Carla, the NVIDIA extension for Docker must be installed. For this section, it is not assumed that Docker is installed on the host macine at all. The installation has been tested on Ubuntu 20.04 as host system. The following instructions can also be found with more explanations [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Using the following command (assuming curl is installed with `sudo apt-get install curl`) will install Docker.

    curl https://get.docker.com | sh && sudo systemctl --now enable docker

The following command will make the NVIDIA Container Toolkit available for installation on the system.

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

Finally the NVIDIA Docker Toolkit can be installed on the system.

    sudo apt-get update
    
    sudo apt-get install -y nvidia-docker2

To apply the changes, it is necessary to restart the Docker service.

    sudo systemctl restart docker
 
To check if the installation was successful, the following command can be run:

    sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
    
This should create an output that looks approximately like the following:

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.80       Driver Version: 460.80       CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 1080    Off  | 00000000:01:00.0  On |                  N/A |
    | 31%   59C    P2    45W / 180W |    821MiB /  8111MiB |      4%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    +-----------------------------------------------------------------------------+

## Using Pre-Built Docker Images
For stability reasons, two separate Docker images are used: One image for the Carla server and one image for AST (including all the Python and Julia dependencies. For the Carla image, the offical Carla Docker image will be used. At the time of writing these instructions, the most current version of Carla is 0.9.11 and we will use this version. The image can be pulled using the following command:

    sudo docker pull carlasim/carla:0.9.11
    
This should download and extract the official Carla Docker image. To create a Docker container (an instance of the image), the following command can be used:

    sudo docker run -p 2222-2224:2222-2224 --runtime=nvidia --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it carlasim/carla:0.9.11 bash
    
The command above creates a Docker container of the official Carla image and starts the container. Since the communication betwen the carla server and the AST environment will be over TCP Port 2222-2224 (any other port could be used as well), those ports are exposed published to the host machine. Now, Carla can be launched using the following command:

    /home/carla/CarlaUE4.sh -carla-rpc-port=2222 -windowed -ResX=640 -ResY=480 -benchmark -fps=10 -quality-level=Epic -opengl

The `-opengl` tag is for compatibility, as found that `-vulkan`, despite being advertised as working doesn't always work. Also, when comparing performance between the two options, we did not find significant performance differences. Especially when running the entire setup *off-screen*, the `-opengl` option is mandatory because the Vulkan engine isn't supported for the version of Unreal Engine that is used with the current version of Carla. When starting carla inside the Docker container, it is normal to get warning messages about sound card drivers that can't be found. This shouldn't be concerning, as currently no sounds are used. 

If carla was launched successfully, a new terminal window can be opened and in a similar procedure, the image for AST can be downloaded.

    sudo docker pull marcschlichting/risk_assessment:0.1
    
After downloading this image and extracting it, a *normal* Docker container can be started and launched (the NVIDIA extension is not necessary for this part). However, the `net=host` option enables the Docker container to communicate to the Carla container over localhost.

    sudo docker run --net=host -it marcschlichting/risk_assessment:0.1 bash
    
All communications and AST will be handelede from this container. 

## Running AST for Pre-Compiled Containers
In the terminal window that belongs to the AST Docker container (formally `risk_assessment`), the spectator position can be set to the area of interest using the command:

    python3 /root/AutonomousRiskFramework/CARLAIntegration/util/config.py --map Town01 --port 2222 --spectator-loc 80.37 25.30 0.0 
    
This location is for the standard example of the T-intersection with the bicyclist that has been shown in previous demonstrations. To run the actual AST, we change the directory and execute the respective python script. Switching to a different scenario and route or agent is as easy changing the files provided by the tags.

    cd /root/AutonomousRiskFramework/CARLAIntegration/scenario_runner/

    python-jl scenario_runner_ast.py --route ./srunner/data/routes_ast.xml ./srunner/data/ast_scenarios.json --port 2222 --agent ./srunner/autoagents/ast_agent.py --record recordings

## Building the Docker Image from the Dockerfile
This section is only intended if the image should be built from scratch and the outcome should be identical to the provided image on Dockerhub. However, we still provide the documentation in case it is of use to anybody. The file to build the image, the `Dockerfile` is located in the `Docker` directory. Building the image can be done using the following command:

    sudo docker build -t autonomous_risk_framework:0.1 /PATH_TO_AUTONOMOUSRISKFRAMEWORK/Docker
    
To verify that the build was successful, the following command can be used to list locally available images.

    sudo docker images
    
The output should look similar to this:

    REPOSITORY                  TAG       IMAGE ID       CREATED          SIZE
    autonomous_risk_framework   0.1       bc29d0c5140a   21 seconds ago   10.8GB
    carlasim/carla              0.9.11    95ae4b17d967   6 months ago     9.86GB

## Configuration Inside the Container
As the github repository is not public yet and building the Docker image is a non-interactive process, some configuration must be done manually at this point inside the Docker container. This mainly includes downloading the repository and installing all the dependencies.
To open the image and create a new container, use the following command (the options are not necessary at the moment, but it is assumed that we want to open Carla later).
    
    sudo docker run --net=host -it marcschlichting/risk_assessment:0.1 bash
    
The output should look like

    carla@a182e68e7243:~$
    
where `a182e68e7243` is the container ID. In the next step, the github repository is cloned in the home directory and the branch is changed to the branch dedicated for the docker container.

    git clone https://github.com/sisl/AutonomousRiskFramework.git
    
    cd AutonomousRiskFramework
    
    git checkout marc/carla_docker
   
To install the necessary python packages, we install

    pip3 install -r /root/AutonomousRiskFramework/CARLAIntegration/scenario_runner/requirements.txt && pip3 install -r /root/AutonomousRiskFramework/CARLAIntegration/scenario_runner/ephem_requirement.txt

*Note: The required Julia packages have already been installed as part of the Dockerfile, in the future as soon as the repository is public, this can also easily be done for the Python packages.*

## Running Carla
Using a new terminal window, the offical Carla image can be launched and the ports can be exposed and published to the host system:

    sudo docker run -p 2222-2224:2222-2224 --runtime=nvidia --net=bridge --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it carlasim/carla:0.9.11 bash

Then Carla can be launched using:

    /home/carla/CarlaUE4.sh -carla-rpc-port=2222 -windowed -ResX=640 -ResY=480 -benchmark -fps=10 -quality-level=Epic -opengl

The Carla window should open at this point and (for Carla 0.9.11) show a roundabout with a fountain. 

To interact with the Carla server, we switch over to the old container where we downloaded all the dependencies. We can now change the map and the position of the spectator using the modified `config.py` file that is located in the `./CARLAIntegration/util/` directory. Such a command could look like:

    python3 /root/AutonomousRiskFramework/CARLAIntegration/util/config.py --map Town01 --port 2222 --spectator-loc 80.37 25.30 0.0  

## Running AST
Running the actual failure search requires `python-jl` which was already installed when creating the docker image. This is necessary for the Python-Julia-Python bridge to work correctly. Details can be found [here](https://pyjulia.readthedocs.io/en/latest/troubleshooting.html). For starting the actual AST failure search, first navigate to the `scenario_runner` directory (in the AST container):

    cd /root/AutonomousRiskFramework/CARLAIntegration/scenario_runner/
    
From there, run the `scenario_runner_ast.py` file using `python-jl`:

    python-jl scenario_runner_ast.py --route ./srunner/data/routes_ast.xml ./srunner/data/ast_scenarios.json --port 2222 --agent ./srunner/autoagents/ast_agent.py --record recordings
    
When running for the first time after building the image, it is possible that some CUDA dependencies for Julia need to be downloaded. This should happen automatically. After this, the command line output in the second terminal as well as rendered view should show a car taking a left turn at an intersection with a pedestrian crossing the street after the intersection. If this happens, the installation was successful.

## Saving the container as a new image
After the successful test, it is advised to save the container as a new image to not always have to download the github repository again (which can't be done as part of the building process of the Docker image as this doesn't allow for the interactive process that is required for cloning private github repositories). After the test, exit the container in both terminals and use the following command (on the host system):

    sudo docker commit a182e68e7243 marcschlichting/risk_assessment:0.1
    
where `a182e68e7243` should be replaced with your container ID for the `autonomous_risk_framework` container. If the ID is unknown, using the `sudo docker ps` from a new terminal window, that lists all active containers, can help. The new image name will then be `marcschlichting/risk_assessment:0.1` and can be opened like a pre-compiled image.

## Contacts
- Stanford Intelligent Systems Laboratory (SISL)
    - Marc Schlichting: [MarcSchlichting](https://github.com/MarcSchlichting)
