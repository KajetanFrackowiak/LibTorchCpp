cmake_minimum_required(VERSION 3.14)
project(torchTest)

# Set the path to the installation of LibTorch
set(LIBTORCH_PATH /home/kajetan/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.2.1+cu121/libtorch)

# Find PyTorch
find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})

# Add PyTorch include directories to the project
include_directories(${TORCH_INCLUDE_DIRS})

# Add PyTorch libraries
set(ADDITIONAL_LIB_PATH /home/kajetan/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.2.1+cu121/libtorch/lib)
link_directories(${ADDITIONAL_LIB_PATH})

# Create an executable with the main.cpp source file
add_executable(${PROJECT_NAME} main.cpp)

# Link against PyTorch libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})