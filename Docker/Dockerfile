FROM carlasim/carla:0.9.11
USER root
RUN apt-get update && apt-get install -y python3.7
RUN rm /usr/bin/python3
RUN ln -s python3.7 /usr/bin/python3
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-dev
RUN apt-get install -y wget
WORKDIR "/home/carla"
RUN mkdir software
WORKDIR "/home/carla/software"
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.2-linux-x86_64.tar.gz
RUN tar zxvf julia-1.6.2-linux-x86_64.tar.gz
RUN echo 'export PATH="$PATH:/home/carla/software/julia-1.6.2/bin"'>>/root/.bashrc

RUN apt-get install -y git nano 
RUN apt-get install -y libjpeg-turbo8
RUN apt-get install -y libfontconfig1
RUN apt-get install -y libtiff5
WORKDIR /home/carla
USER carla
RUN echo 'export PATH="$PATH:/home/carla/software/julia-1.6.2/bin"'>>/home/carla/.bashrc
RUN echo 'export CARLA_ROOT="/home/carla"'>>/home/carla/.bashrc
RUN echo 'export PYTHONPATH="$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla"'>>/home/carla/.bashrc

ENV PATH="$PATH:/home/carla/software/julia-1.6.2/bin"
