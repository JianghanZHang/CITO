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

# Utility rule file for NightlyConfigure.

# Include any custom commands dependencies for this target.
include CMakeFiles/NightlyConfigure.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/NightlyConfigure.dir/progress.make

CMakeFiles/NightlyConfigure:
	/home/jianghan/miniconda3/envs/mim_cio/bin/ctest -D NightlyConfigure

NightlyConfigure: CMakeFiles/NightlyConfigure
NightlyConfigure: CMakeFiles/NightlyConfigure.dir/build.make
.PHONY : NightlyConfigure

# Rule to build all files generated by this target.
CMakeFiles/NightlyConfigure.dir/build: NightlyConfigure
.PHONY : CMakeFiles/NightlyConfigure.dir/build

CMakeFiles/NightlyConfigure.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/NightlyConfigure.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NightlyConfigure.dir/clean

CMakeFiles/NightlyConfigure.dir/depend:
	cd /home/jianghan/Devel/workspace_autogait/src/CITO/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jianghan/Devel/workspace_autogait/src/CITO /home/jianghan/Devel/workspace_autogait/src/CITO /home/jianghan/Devel/workspace_autogait/src/CITO/build /home/jianghan/Devel/workspace_autogait/src/CITO/build /home/jianghan/Devel/workspace_autogait/src/CITO/build/CMakeFiles/NightlyConfigure.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/NightlyConfigure.dir/depend

