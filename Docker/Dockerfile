FROM carlasim/carla:0.9.11 as carla-image

FROM ubuntu:18.04

RUN apt-get update -y 
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-dev
RUN apt-get install -y wget
RUN apt-get install -y xdg-user-dirs
RUN mkdir software
WORKDIR "/root/software"
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.2-linux-x86_64.tar.gz
RUN tar zxvf julia-1.6.2-linux-x86_64.tar.gz
RUN echo 'export PATH="$PATH:/root/software/julia-1.6.2/bin"'>>/root/.bashrc

RUN apt-get install -y git nano 
RUN apt-get install -y libjpeg-turbo8
RUN apt-get install -y libfontconfig1
RUN apt-get install -y libtiff5
WORKDIR /root

RUN pip3 install julia

RUN mkdir carla

COPY --from=carla-image /home/carla/PythonAPI /root/carla/PythonAPI

RUN echo 'export PATH="$PATH:/root/software/julia-1.6.2/bin"'>>/root/.bashrc
RUN echo 'export CARLA_ROOT="/root/carla"'>>/root/.bashrc
RUN echo 'export PYTHONPATH="$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla"'>>/root/.bashrc

ENV PATH="$PATH:/root/software/julia-1.6.2/bin"

#RUN julia -e 'import Pkg; Pkg.add(url="https://github.com/sisl/AdversarialDriving.jl")'
RUN julia -e 'import Pkg; Pkg.add("ColorSchemes"); Pkg.add("Convex"); Pkg.add("Dates"); Pkg.add("DependencyWalker"); Pkg.add("Distributions")'
RUN julia -e 'using Pkg; Pkg.develop(PackageSpec("CrossEntropyMethod"))'
RUN julia -e 'import Pkg; Pkg.precompile()'
RUN julia -e 'import Pkg; Pkg.add("FileIO"); Pkg.add("LaTeXStrings");Pkg.add("LightXML");Pkg.add("LinearAlgebra")'
RUN julia -e 'import Pkg; Pkg.add("OpenSSL_jll");Pkg.add("PGFPlotsX");Pkg.add("Parameters");Pkg.add("PyPlot")'
RUN julia -e 'using Pkg; pkg"add https://github.com/JuliaPOMDP/RLInterface.jl"; pkg"add https://github.com/sisl/POMDPStressTesting.jl"'
RUN julia -e 'import Pkg; Pkg.add("Random");Pkg.add("SCS");Pkg.add("StatsBase")'
RUN julia -e 'import Pkg; Pkg.add(url="https://github.com/sisl/Vec.jl.git")'
RUN julia -e 'import Pkg; Pkg.add("FFMPEG"); Pkg.add("Plots")'
RUN julia -e 'import Pkg; Pkg.add("PyCall")'
