cmake_minimum_required(VERSION 3.1)
project(torchTest)

list(APPEND CMAKE_PREFIX_PATH "/home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/")
set(CMAKE_MODULE_PATH "/home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/share/cmake/Torch")

# Set the path to the NVToolsExt library
set(LIBNVTOOLSEXT "/home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libnvToolsExt-847d78f2.so.1")

# Find PyTorch
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Add PyTorch include directories to the project
include_directories(${TORCH_INCLUDE_DIRS})

# Create an executable with the main.cpp source file
add_executable(${PROJECT_NAME} src/main.cpp src/dataset.cpp include/dataset.h)

# Link against PyTorch libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${LIBNVTOOLSEXT})

target_include_directories(${PROJECT_NAME} PRIVATE include)

# mkdir build
# cd build
# cmake ..
# cmake --build .
# ./