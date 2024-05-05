cmake_minimum_required(VERSION 3.1)
project(torchTest)

# Set the path to the installation of LibTorch
set(LIBTORCH_PATH /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch)
# Set the path to the NVToolsExt library
set(LIBNVTOOLSEXT "/home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libnvToolsExt-847d78f2.so.1")
# Set the path to the matplotlib-cpp library
set(MATPLOTLIB_CPP_PATH /home/frackowiak/Documents/INCLUDES/matplotlib-cpp)
# Set the path to the tqdm library
# Find PyTorch
find_package(Torch REQUIRED)

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Find NumPy
# find_package(NumPy REQUIRED)

# Add include directories for Python and NumPy
include_directories(${Python_INCLUDE_DIRS} )
# include_directories(${NumPy_INCLUDE_DIRS})

# Add PyTorch include directory to the project
include_directories(${TORCH_INCLUDE_DIRS})
# Add matplotlib-cpp include directory to the project
include_directories(${MATPLOTLIB_CPP_PATH})

# Create an executable with the main.cpp source file
add_executable(${PROJECT_NAME} main.cpp)

# Link against PyTorch libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${LIBNVTOOLSEXT} ${Python_LIBRARIES})
