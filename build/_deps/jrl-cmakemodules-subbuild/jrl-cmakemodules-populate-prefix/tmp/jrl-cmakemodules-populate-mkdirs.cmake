# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/jianghan/Devel/workspace_autogait/src/CITO/build/_deps/jrl-cmakemodules-src"
  "/home/jianghan/Devel/workspace_autogait/src/CITO/build/_deps/jrl-cmakemodules-build"
  "/home/jianghan/Devel/workspace_autogait/src/CITO/build/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix"
  "/home/jianghan/Devel/workspace_autogait/src/CITO/build/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/tmp"
  "/home/jianghan/Devel/workspace_autogait/src/CITO/build/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/src/jrl-cmakemodules-populate-stamp"
  "/home/jianghan/Devel/workspace_autogait/src/CITO/build/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/src"
  "/home/jianghan/Devel/workspace_autogait/src/CITO/build/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/src/jrl-cmakemodules-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/jianghan/Devel/workspace_autogait/src/CITO/build/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/src/jrl-cmakemodules-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/jianghan/Devel/workspace_autogait/src/CITO/build/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/src/jrl-cmakemodules-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
