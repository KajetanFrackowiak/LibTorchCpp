# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/frackowiak/Documents/LibTorchCpp/CppTorch3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/frackowiak/Documents/LibTorchCpp/CppTorch3/build

# Include any dependencies generated for this target.
include CMakeFiles/torchTest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/torchTest.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/torchTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/torchTest.dir/flags.make

CMakeFiles/torchTest.dir/src/main.cpp.o: CMakeFiles/torchTest.dir/flags.make
CMakeFiles/torchTest.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/torchTest.dir/src/main.cpp.o: CMakeFiles/torchTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frackowiak/Documents/LibTorchCpp/CppTorch3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/torchTest.dir/src/main.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/torchTest.dir/src/main.cpp.o -MF CMakeFiles/torchTest.dir/src/main.cpp.o.d -o CMakeFiles/torchTest.dir/src/main.cpp.o -c /home/frackowiak/Documents/LibTorchCpp/CppTorch3/src/main.cpp

CMakeFiles/torchTest.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/torchTest.dir/src/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/frackowiak/Documents/LibTorchCpp/CppTorch3/src/main.cpp > CMakeFiles/torchTest.dir/src/main.cpp.i

CMakeFiles/torchTest.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/torchTest.dir/src/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/frackowiak/Documents/LibTorchCpp/CppTorch3/src/main.cpp -o CMakeFiles/torchTest.dir/src/main.cpp.s

CMakeFiles/torchTest.dir/src/dataset.cpp.o: CMakeFiles/torchTest.dir/flags.make
CMakeFiles/torchTest.dir/src/dataset.cpp.o: ../src/dataset.cpp
CMakeFiles/torchTest.dir/src/dataset.cpp.o: CMakeFiles/torchTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frackowiak/Documents/LibTorchCpp/CppTorch3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/torchTest.dir/src/dataset.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/torchTest.dir/src/dataset.cpp.o -MF CMakeFiles/torchTest.dir/src/dataset.cpp.o.d -o CMakeFiles/torchTest.dir/src/dataset.cpp.o -c /home/frackowiak/Documents/LibTorchCpp/CppTorch3/src/dataset.cpp

CMakeFiles/torchTest.dir/src/dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/torchTest.dir/src/dataset.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/frackowiak/Documents/LibTorchCpp/CppTorch3/src/dataset.cpp > CMakeFiles/torchTest.dir/src/dataset.cpp.i

CMakeFiles/torchTest.dir/src/dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/torchTest.dir/src/dataset.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/frackowiak/Documents/LibTorchCpp/CppTorch3/src/dataset.cpp -o CMakeFiles/torchTest.dir/src/dataset.cpp.s

# Object files for target torchTest
torchTest_OBJECTS = \
"CMakeFiles/torchTest.dir/src/main.cpp.o" \
"CMakeFiles/torchTest.dir/src/dataset.cpp.o"

# External object files for target torchTest
torchTest_EXTERNAL_OBJECTS =

torchTest: CMakeFiles/torchTest.dir/src/main.cpp.o
torchTest: CMakeFiles/torchTest.dir/src/dataset.cpp.o
torchTest: CMakeFiles/torchTest.dir/build.make
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
torchTest: /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libtorch.so
torchTest: /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libc10.so
torchTest: /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libkineto.a
torchTest: /usr/lib/x86_64-linux-gnu/libcuda.so
torchTest: /home/frackowiak/anaconda3/lib/libnvrtc.so
torchTest: /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libnvToolsExt-847d78f2.so.1
torchTest: /home/frackowiak/anaconda3/lib/libcudart.so
torchTest: /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libc10_cuda.so
torchTest: /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libnvToolsExt-847d78f2.so.1
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
torchTest: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
torchTest: /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libc10_cuda.so
torchTest: /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libc10.so
torchTest: /home/frackowiak/anaconda3/lib/libcudart.so
torchTest: /home/frackowiak/anaconda3/lib/libnvToolsExt.so
torchTest: /home/frackowiak/anaconda3/lib/libcudart.so
torchTest: /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libc10_cuda.so
torchTest: CMakeFiles/torchTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frackowiak/Documents/LibTorchCpp/CppTorch3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable torchTest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/torchTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/torchTest.dir/build: torchTest
.PHONY : CMakeFiles/torchTest.dir/build

CMakeFiles/torchTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/torchTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/torchTest.dir/clean

CMakeFiles/torchTest.dir/depend:
	cd /home/frackowiak/Documents/LibTorchCpp/CppTorch3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frackowiak/Documents/LibTorchCpp/CppTorch3 /home/frackowiak/Documents/LibTorchCpp/CppTorch3 /home/frackowiak/Documents/LibTorchCpp/CppTorch3/build /home/frackowiak/Documents/LibTorchCpp/CppTorch3/build /home/frackowiak/Documents/LibTorchCpp/CppTorch3/build/CMakeFiles/torchTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/torchTest.dir/depend

