FROM nvidia/cuda:10.0-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    cmake \
    make \
    g++ \
    software-properties-common \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Update gcc/g++ to v9
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update -qq
RUN apt-get install -qq gcc-9 g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9

#Install clang
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - \
    && apt-get update \
    && apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-6.0 main" \
    && apt-get install -y clang-6.0 lld-6.0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/clang-6.0 /usr/bin/clang \
    && ln -s /usr/bin/clang++-6.0 /usr/bin/clang++ \
    && ln -s /usr/bin/llc-6.0 /usr/bin/llc

#Update cmake to v3.10
RUN cd /usr/local/src \
    && wget https://cmake.org/files/v3.10/cmake-3.10.3.tar.gz \
    && tar xvf cmake-3.10.3.tar.gz \
    && cd cmake-3.10.3 \
    && ./bootstrap \
    && make \
    && make install \
    && cd .. \
    && rm -rf cmake*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
RUN adduser user sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# https://github.com/anibali/docker-pytorch
# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name gforce python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=gforce
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya

# CUDA 10.0-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    "pytorch=1.4.0=py3.6_cuda10.0.130_cudnn7.6.3_0" \
    "torchvision=0.5.0=py36_cu100" \
 && conda clean -ya

RUN conda install -y pip numpy sympy pyyaml scipy ipython mkl mkl-include ninja cython typing pybind11\
 && conda clean -ya

WORKDIR /app
RUN git clone "https://github.com/Lucieno/SEAL-Python"

# Build SEAL-Python
WORKDIR /app/SEAL-Python/SEAL/native/src
RUN cmake .
RUN make
RUN sudo make install

WORKDIR /app/SEAL-Python/src
RUN pip install -r requirements.txt
RUN python setup.py build_ext -i && \
    python setup.py install

WORKDIR /app
#RUN git clone "https://github.com/Lucieno/GForce"
