cmake_minimum_required(VERSION 3.14)
project(torchTest)

# Set the path to the installation of LibTorch
set(LIBTORCH_PATH /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch)

# Find PyTorch
find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})

# Add PyTorch include directories to the project
include_directories(${TORCH_INCLUDE_DIRS})

# Add PyTorch libraries
set(ADDITIONAL_LIB_PATH /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib)
link_directories(${ADDITIONAL_LIB_PATH})

include_directories(include)
# Create an executable with the main.cpp source file
add_executable(${PROJECT_NAME} include/network.h src/main.cpp)

# Link against PyTorch libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})


# mkdir build
# cd build
# cmake ..
# cmake --build .
# ./
