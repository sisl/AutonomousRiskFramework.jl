# Docker Container for Autonomous Risk Framework

This readme document is intended to lead through the installation of Docker and the necessary packages. 

## Installation of Docker and NVIDIA Docker on the Host Machine
To fully Carla, the NVIDIA extension for Docker must be installed. For this section, it is not assumed that Docker is installed on the host macine at all. The installation has been tested on Ubuntu 20.04 as host system. Using the following command (assuming curl is installed with `sudo apt-get install curl`) will install Docker.

    curl https://get.docker.com | sh && sudo systemctl --now enable docker

The following command will set up the NVIDIA Container Toolkit.

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list



## Contacts
- Stanford Intelligent Systems Laboratory (SISL)
    - Marc Schlichting: [MarcSchlichting](https://github.com/MarcSchlichting)
