# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/jianwei/software/clion-2017.3/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/jianwei/software/clion-2017.3/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jianwei/code/SLAM/loop_closure

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jianwei/code/SLAM/loop_closure/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/loop_closure.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/loop_closure.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/loop_closure.dir/flags.make

CMakeFiles/loop_closure.dir/main.cpp.o: CMakeFiles/loop_closure.dir/flags.make
CMakeFiles/loop_closure.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jianwei/code/SLAM/loop_closure/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/loop_closure.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/loop_closure.dir/main.cpp.o -c /home/jianwei/code/SLAM/loop_closure/main.cpp

CMakeFiles/loop_closure.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/loop_closure.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jianwei/code/SLAM/loop_closure/main.cpp > CMakeFiles/loop_closure.dir/main.cpp.i

CMakeFiles/loop_closure.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/loop_closure.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jianwei/code/SLAM/loop_closure/main.cpp -o CMakeFiles/loop_closure.dir/main.cpp.s

CMakeFiles/loop_closure.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/loop_closure.dir/main.cpp.o.requires

CMakeFiles/loop_closure.dir/main.cpp.o.provides: CMakeFiles/loop_closure.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/loop_closure.dir/build.make CMakeFiles/loop_closure.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/loop_closure.dir/main.cpp.o.provides

CMakeFiles/loop_closure.dir/main.cpp.o.provides.build: CMakeFiles/loop_closure.dir/main.cpp.o


CMakeFiles/loop_closure.dir/fileIO.cpp.o: CMakeFiles/loop_closure.dir/flags.make
CMakeFiles/loop_closure.dir/fileIO.cpp.o: ../fileIO.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jianwei/code/SLAM/loop_closure/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/loop_closure.dir/fileIO.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/loop_closure.dir/fileIO.cpp.o -c /home/jianwei/code/SLAM/loop_closure/fileIO.cpp

CMakeFiles/loop_closure.dir/fileIO.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/loop_closure.dir/fileIO.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jianwei/code/SLAM/loop_closure/fileIO.cpp > CMakeFiles/loop_closure.dir/fileIO.cpp.i

CMakeFiles/loop_closure.dir/fileIO.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/loop_closure.dir/fileIO.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jianwei/code/SLAM/loop_closure/fileIO.cpp -o CMakeFiles/loop_closure.dir/fileIO.cpp.s

CMakeFiles/loop_closure.dir/fileIO.cpp.o.requires:

.PHONY : CMakeFiles/loop_closure.dir/fileIO.cpp.o.requires

CMakeFiles/loop_closure.dir/fileIO.cpp.o.provides: CMakeFiles/loop_closure.dir/fileIO.cpp.o.requires
	$(MAKE) -f CMakeFiles/loop_closure.dir/build.make CMakeFiles/loop_closure.dir/fileIO.cpp.o.provides.build
.PHONY : CMakeFiles/loop_closure.dir/fileIO.cpp.o.provides

CMakeFiles/loop_closure.dir/fileIO.cpp.o.provides.build: CMakeFiles/loop_closure.dir/fileIO.cpp.o


CMakeFiles/loop_closure.dir/Tracking.cpp.o: CMakeFiles/loop_closure.dir/flags.make
CMakeFiles/loop_closure.dir/Tracking.cpp.o: ../Tracking.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jianwei/code/SLAM/loop_closure/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/loop_closure.dir/Tracking.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/loop_closure.dir/Tracking.cpp.o -c /home/jianwei/code/SLAM/loop_closure/Tracking.cpp

CMakeFiles/loop_closure.dir/Tracking.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/loop_closure.dir/Tracking.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jianwei/code/SLAM/loop_closure/Tracking.cpp > CMakeFiles/loop_closure.dir/Tracking.cpp.i

CMakeFiles/loop_closure.dir/Tracking.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/loop_closure.dir/Tracking.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jianwei/code/SLAM/loop_closure/Tracking.cpp -o CMakeFiles/loop_closure.dir/Tracking.cpp.s

CMakeFiles/loop_closure.dir/Tracking.cpp.o.requires:

.PHONY : CMakeFiles/loop_closure.dir/Tracking.cpp.o.requires

CMakeFiles/loop_closure.dir/Tracking.cpp.o.provides: CMakeFiles/loop_closure.dir/Tracking.cpp.o.requires
	$(MAKE) -f CMakeFiles/loop_closure.dir/build.make CMakeFiles/loop_closure.dir/Tracking.cpp.o.provides.build
.PHONY : CMakeFiles/loop_closure.dir/Tracking.cpp.o.provides

CMakeFiles/loop_closure.dir/Tracking.cpp.o.provides.build: CMakeFiles/loop_closure.dir/Tracking.cpp.o


CMakeFiles/loop_closure.dir/Database.cpp.o: CMakeFiles/loop_closure.dir/flags.make
CMakeFiles/loop_closure.dir/Database.cpp.o: ../Database.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jianwei/code/SLAM/loop_closure/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/loop_closure.dir/Database.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/loop_closure.dir/Database.cpp.o -c /home/jianwei/code/SLAM/loop_closure/Database.cpp

CMakeFiles/loop_closure.dir/Database.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/loop_closure.dir/Database.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jianwei/code/SLAM/loop_closure/Database.cpp > CMakeFiles/loop_closure.dir/Database.cpp.i

CMakeFiles/loop_closure.dir/Database.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/loop_closure.dir/Database.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jianwei/code/SLAM/loop_closure/Database.cpp -o CMakeFiles/loop_closure.dir/Database.cpp.s

CMakeFiles/loop_closure.dir/Database.cpp.o.requires:

.PHONY : CMakeFiles/loop_closure.dir/Database.cpp.o.requires

CMakeFiles/loop_closure.dir/Database.cpp.o.provides: CMakeFiles/loop_closure.dir/Database.cpp.o.requires
	$(MAKE) -f CMakeFiles/loop_closure.dir/build.make CMakeFiles/loop_closure.dir/Database.cpp.o.provides.build
.PHONY : CMakeFiles/loop_closure.dir/Database.cpp.o.provides

CMakeFiles/loop_closure.dir/Database.cpp.o.provides.build: CMakeFiles/loop_closure.dir/Database.cpp.o


# Object files for target loop_closure
loop_closure_OBJECTS = \
"CMakeFiles/loop_closure.dir/main.cpp.o" \
"CMakeFiles/loop_closure.dir/fileIO.cpp.o" \
"CMakeFiles/loop_closure.dir/Tracking.cpp.o" \
"CMakeFiles/loop_closure.dir/Database.cpp.o"

# External object files for target loop_closure
loop_closure_EXTERNAL_OBJECTS =

loop_closure: CMakeFiles/loop_closure.dir/main.cpp.o
loop_closure: CMakeFiles/loop_closure.dir/fileIO.cpp.o
loop_closure: CMakeFiles/loop_closure.dir/Tracking.cpp.o
loop_closure: CMakeFiles/loop_closure.dir/Database.cpp.o
loop_closure: CMakeFiles/loop_closure.dir/build.make
loop_closure: /root/anaconda3/lib/libopencv_xphoto.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_xobjdetect.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_tracking.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_surface_matching.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_structured_light.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_stereo.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_saliency.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_rgbd.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_reg.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_plot.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_optflow.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_line_descriptor.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_hdf.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_fuzzy.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_dpm.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_dnn.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_datasets.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_ccalib.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_bioinspired.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_bgsegm.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_aruco.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_videostab.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_superres.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_stitching.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_photo.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_text.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_face.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_ximgproc.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_xfeatures2d.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_shape.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_video.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_objdetect.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_calib3d.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_features2d.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_ml.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_highgui.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_videoio.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_imgcodecs.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_imgproc.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_flann.so.3.1.0
loop_closure: /root/anaconda3/lib/libopencv_core.so.3.1.0
loop_closure: CMakeFiles/loop_closure.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jianwei/code/SLAM/loop_closure/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable loop_closure"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/loop_closure.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/loop_closure.dir/build: loop_closure

.PHONY : CMakeFiles/loop_closure.dir/build

CMakeFiles/loop_closure.dir/requires: CMakeFiles/loop_closure.dir/main.cpp.o.requires
CMakeFiles/loop_closure.dir/requires: CMakeFiles/loop_closure.dir/fileIO.cpp.o.requires
CMakeFiles/loop_closure.dir/requires: CMakeFiles/loop_closure.dir/Tracking.cpp.o.requires
CMakeFiles/loop_closure.dir/requires: CMakeFiles/loop_closure.dir/Database.cpp.o.requires

.PHONY : CMakeFiles/loop_closure.dir/requires

CMakeFiles/loop_closure.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/loop_closure.dir/cmake_clean.cmake
.PHONY : CMakeFiles/loop_closure.dir/clean

CMakeFiles/loop_closure.dir/depend:
	cd /home/jianwei/code/SLAM/loop_closure/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jianwei/code/SLAM/loop_closure /home/jianwei/code/SLAM/loop_closure /home/jianwei/code/SLAM/loop_closure/cmake-build-debug /home/jianwei/code/SLAM/loop_closure/cmake-build-debug /home/jianwei/code/SLAM/loop_closure/cmake-build-debug/CMakeFiles/loop_closure.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/loop_closure.dir/depend
