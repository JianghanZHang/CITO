# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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

# Produce verbose output by default.
VERBOSE = 1

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
CMAKE_COMMAND = /home/jianghan/miniconda3/envs/mim_cio/bin/cmake

# The command to remove a file.
RM = /home/jianghan/miniconda3/envs/mim_cio/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jianghan/Devel/workspace_autogait/src/CITO

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jianghan/Devel/workspace_autogait/src/CITO/build

# Utility rule file for cito-distorig.

# Include any custom commands dependencies for this target.
include CMakeFiles/cito-distorig.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cito-distorig.dir/progress.make

CMakeFiles/cito-distorig:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/jianghan/Devel/workspace_autogait/src/CITO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating orig tarball for cito..."
	cmake -E copy cito-0.0.0.tar.gz cito-0.0.0.orig.tar.gz

cito-distorig: CMakeFiles/cito-distorig
cito-distorig: CMakeFiles/cito-distorig.dir/build.make
.PHONY : cito-distorig

# Rule to build all files generated by this target.
CMakeFiles/cito-distorig.dir/build: cito-distorig
.PHONY : CMakeFiles/cito-distorig.dir/build

CMakeFiles/cito-distorig.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cito-distorig.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cito-distorig.dir/clean

CMakeFiles/cito-distorig.dir/depend:
	cd /home/jianghan/Devel/workspace_autogait/src/CITO/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jianghan/Devel/workspace_autogait/src/CITO /home/jianghan/Devel/workspace_autogait/src/CITO /home/jianghan/Devel/workspace_autogait/src/CITO/build /home/jianghan/Devel/workspace_autogait/src/CITO/build /home/jianghan/Devel/workspace_autogait/src/CITO/build/CMakeFiles/cito-distorig.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/cito-distorig.dir/depend

