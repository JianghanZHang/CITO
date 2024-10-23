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

# Utility rule file for cito-release.

# Include any custom commands dependencies for this target.
include CMakeFiles/cito-release.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cito-release.dir/progress.make

CMakeFiles/cito-release:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/jianghan/Devel/workspace_autogait/src/CITO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Create a new release for cito"
	cd /home/jianghan/Devel/workspace_autogait/src/CITO && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/ros/foxy/lib/x86_64-linux-gnu:/home/jianghan/miniconda3/envs/mim_cio/lib:/opt/openrobots/lib:/usr/local/cuda/lib64:/opt/ros/foxy/lib/x86_64-linux-gnu:/lib:/opt/openrobots/lib:/opt/ros/foxy/opt/yaml_cpp_vendor/lib:/opt/ros/foxy/opt/rviz_ogre_vendor/lib:/opt/ros/foxy/lib/x86_64-linux-gnu:/opt/ros/foxy/lib::/opt/openrobots/lib && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/ros/foxy/lib/x86_64-linux-gnu:/home/jianghan/miniconda3/envs/mim_cio/lib:/opt/openrobots/lib:/usr/local/cuda/lib64:/opt/ros/foxy/lib/x86_64-linux-gnu:/lib:/opt/openrobots/lib:/opt/ros/foxy/opt/yaml_cpp_vendor/lib:/opt/ros/foxy/opt/rviz_ogre_vendor/lib:/opt/ros/foxy/lib/x86_64-linux-gnu:/opt/ros/foxy/lib::/opt/openrobots/lib && export PYTHONPATH=/home/jianghan/miniconda3/envs/mim_cio/lib/python3.12/site-packages:/opt/openrobots/lib/python3.8/site-packages:/lib/python3.12/site-packages:/opt/openrobots/lib/python3.8/site-packages:/opt/ros/foxy/lib/python3.8/site-packages && ! test x$$VERSION = x || ( echo Please\ set\ a\ version\ for\ this\ release && false ) && if [ -f package.xml ]; then ( /home/jianghan/miniconda3/envs/mim_cio/bin/cmake --build /home/jianghan/Devel/workspace_autogait/src/CITO/build --target cito-release_package_xml ) fi && if [ -f pyproject.toml ]; then ( /home/jianghan/miniconda3/envs/mim_cio/bin/cmake --build /home/jianghan/Devel/workspace_autogait/src/CITO/build --target cito-release_pyproject_toml ) fi && if [ -f CHANGELOG.md ]; then ( /home/jianghan/miniconda3/envs/mim_cio/bin/cmake --build /home/jianghan/Devel/workspace_autogait/src/CITO/build --target cito-release_changelog ) fi && if [ -f pixi.toml ]; then ( /home/jianghan/miniconda3/envs/mim_cio/bin/cmake --build /home/jianghan/Devel/workspace_autogait/src/CITO/build --target cito-release_pixi_toml ) fi && if [ -f CITATION.cff ]; then ( /home/jianghan/miniconda3/envs/mim_cio/bin/cmake --build /home/jianghan/Devel/workspace_autogait/src/CITO/build --target cito-release_citation_cff ) fi && /usr/bin/git tag -s v$$VERSION -m Release\ of\ version\ $$VERSION. && cd /home/jianghan/Devel/workspace_autogait/src/CITO/build && cmake /home/jianghan/Devel/workspace_autogait/src/CITO && /home/jianghan/miniconda3/envs/mim_cio/bin/cmake --build /home/jianghan/Devel/workspace_autogait/src/CITO/build --target cito-distcheck || ( echo Please\ fix\ distcheck\ first. && cd /home/jianghan/Devel/workspace_autogait/src/CITO && /usr/bin/git tag -d v$$VERSION && cd /home/jianghan/Devel/workspace_autogait/src/CITO/build && cmake /home/jianghan/Devel/workspace_autogait/src/CITO && false ) && /home/jianghan/miniconda3/envs/mim_cio/bin/cmake --build /home/jianghan/Devel/workspace_autogait/src/CITO/build --target cito-dist && /home/jianghan/miniconda3/envs/mim_cio/bin/cmake --build /home/jianghan/Devel/workspace_autogait/src/CITO/build --target cito-distclean && echo Please,\ run\ 'git\ push\ --tags'\ and\ upload\ the\ tarball\ to\ github\ to\ finalize\ this\ release.

cito-release: CMakeFiles/cito-release
cito-release: CMakeFiles/cito-release.dir/build.make
.PHONY : cito-release

# Rule to build all files generated by this target.
CMakeFiles/cito-release.dir/build: cito-release
.PHONY : CMakeFiles/cito-release.dir/build

CMakeFiles/cito-release.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cito-release.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cito-release.dir/clean

CMakeFiles/cito-release.dir/depend:
	cd /home/jianghan/Devel/workspace_autogait/src/CITO/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jianghan/Devel/workspace_autogait/src/CITO /home/jianghan/Devel/workspace_autogait/src/CITO /home/jianghan/Devel/workspace_autogait/src/CITO/build /home/jianghan/Devel/workspace_autogait/src/CITO/build /home/jianghan/Devel/workspace_autogait/src/CITO/build/CMakeFiles/cito-release.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/cito-release.dir/depend

