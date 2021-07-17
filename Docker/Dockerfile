FROM ubuntu:18.04 as base-image

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.1.243
ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-$CUDA_PKG_VERSION \
    cuda-compat-10-1 \
    && ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# COPY NGC-DL-CONTAINER-LICENSE /

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"



FROM base-image as vulkan-image

ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-xcb-dev \
    libxkbcommon-dev \
    libwayland-dev \
    libxrandr-dev \
    libegl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*
    
FROM carlasim/carla:0.9.11 as carla-image

FROM vulkan-image as carla-surrogate

RUN packages='libsdl2-2.0 xserver-xorg libvulkan1' \
	&& apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y $packages --no-install-recommends \
    && VULKAN_API_VERSION=`dpkg -s libvulkan1 | grep -oP 'Version: [0-9|\.]+' | grep -oP '[0-9|\.]+'` && \
	mkdir -p /etc/vulkan/icd.d/ && \
	echo \
	"{\
		\"file_format_version\" : \"1.0.0\",\
		\"ICD\": {\
			\"library_path\": \"libGLX_nvidia.so.0\",\
			\"api_version\" : \"${VULKAN_API_VERSION}\"\
		}\
	}" > /etc/vulkan/icd.d/nvidia_icd.json \
	&& rm -rf /var/lib/apt/lists/*


RUN useradd -m carla

COPY --from=carla-image /home/carla /home/carla

USER carla
WORKDIR /home/carla

# Select the video driver between offsreen and x11
# x11 is recommended for vulkan support
ENV SDL_VIDEODRIVER=x11


FROM carla-surrogate

USER root
RUN apt-get update -y 
RUN apt-get upgrade -y
#&& apt-get install -y python3.7
#RUN rm /usr/bin/python3
#RUN ln -s python3.7 /usr/bin/python3
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

RUN pip3 install julia

USER carla
RUN echo 'export PATH="$PATH:/home/carla/software/julia-1.6.2/bin"'>>/home/carla/.bashrc
RUN echo 'export CARLA_ROOT="/home/carla"'>>/home/carla/.bashrc
RUN echo 'export PYTHONPATH="$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla"'>>/home/carla/.bashrc

ENV PATH="$PATH:/home/carla/software/julia-1.6.2/bin"

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



