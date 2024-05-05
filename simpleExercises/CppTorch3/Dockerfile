FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && apt-get install -y \
    cmake \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy LibTorch GPU release into the container
COPY libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch /app/libtorch

# Set the CXX environment variable 
ENV CXX /urs/bin/g++

# Copy your project files into the container
COPY . /app

# Set the working directory
WORKDIR /app

# Build your project
RUN cmake .
RUN cmake --build .

# Set the entry point
ENTRYPOINT ["./torchTest"]
