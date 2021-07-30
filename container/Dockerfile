FROM nvidia/cudagl:9.2-devel-ubuntu18.04

# install dependencies
ENV NCCL_VERSION 2.6.4
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
RUN apt-get update && apt-get install -y  --no-install-recommends \
    cuda-nvml-dev-9.2 \
    cuda-command-line-tools-9.2 \
    cuda-nvprof-9.2 \
    cuda-npp-dev-9.2 \
    cuda-libraries-dev-9.2 \
    cuda-minimal-build-9.2 \
    cuda-cublas-dev-9-2 \
    cython3 \
    git \
    libboost-all-dev \
    libgoogle-glog-dev \
    libgles2-mesa-dev \
    libatlas-base-dev \
    libgflags-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libnccl-dev=2.4.8-1+cuda9.2 \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-setuptools \
    python-scipy \
    python3-sklearn \
    python3-sklearn-lib && \
    rm -rf /var/lib/apt/lists/* && \
    apt-mark hold libnccl-dev && \
    apt-mark hold cuda-cublas-dev-9-2

RUN pip3 install scikit-image easydict protobuf pyyaml

# Compile MatterSim
RUN git clone --recurse-submodules https://github.com/peteanderson80/Matterport3DSimulator /code//Matterport3DSimulator && \
    cd /code/Matterport3DSimulator && \
    mkdir build && \
    cd build && \
    cmake -DEGL_RENDERING=ON .. && \
    make -j4 && \
    cd ../
ENV PYTHONPATH="/code/Matterport3DSimulator/build"


# HDF5 (required for Pycaffe)
RUN wget http://h5cpp.org/download/hdf5-1.10.6.tar.gz && \
    tar -xvzf hdf5-1.10.6.tar.gz && \
    cd hdf5-1.10.6/ && \
    ./configure --prefix=/usr/local --enable-build-mode=production && \
    make -j4 && \
    make install && \
    ldconfig

# Compile Bottom-up attention
RUN git clone --recurse-submodules https://github.com/peteanderson80/bottom-up-attention.git /code/bottom-up-attention && \
    cd /code/bottom-up-attention/caffe && \
    wget https://raw.githubusercontent.com/arjunmajum/vln-bert/master/scripts/matterport3D-updown-features/Makefile.config && \
    make -j4 && make -j4 pycaffe 
ENV PYTHONPATH="/code/bottom-up-attention/caffe/python:${PYTHONPATH}"
RUN cd /code/bottom-up-attention/lib && \
    2to3-2.7 -w . && \
    python3 setup.py build_ext --inplace
