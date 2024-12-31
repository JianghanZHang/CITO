cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)  # Or set it to 14, 17, or higher depending on your project requirements.


set(PROJECT_NAMESPACE cito)
set(PROJECT_NAME cito)
set(PROJECT_DESCRIPTION "contact-implicit trajectory optimization project")
set(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})

add_compile_options(-fno-stack-protector)

option(BUILD_PYTHON_INTERFACE "Build the python bindings" ON)

# set(CXX_DISABLE_WERROR ON)

set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(CUSTOM_HEADER_DIR ${PROJECT_NAME})

# Check if the submodule cmake has been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  IF(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    MESSAGE(FATAL_ERROR "\nPlease run the following command first:\ngit submodule update --init\n")
  ELSE()
    include(FetchContent)
    FetchContent_Declare("jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  ENDIF()
ENDIF()

# JRL-cmakemodule setup
include("${JRL_CMAKE_MODULES}/base.cmake")
include("${JRL_CMAKE_MODULES}/boost.cmake")
include("${JRL_CMAKE_MODULES}/python.cmake")

# Project definition
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})
# Assuming you want to link MuJoCo from the Conda environment where the Python package is installed

set(MUJOCO_INCLUDE_DIR "${CMAKE_PREFIX_PATH}/include/mujoco")

if(UNIX AND NOT APPLE) # Check if the platform is Linux
# Search for the MuJoCo library dynamically
  file(GLOB MUJOCO_LIBRARIES "${CMAKE_PREFIX_PATH}/lib/libmujoco.so*")

  # Check if any library is found
  if(MUJOCO_LIBRARIES)
      # Take the first match
      list(SORT MUJOCO_LIBRARIES)
      list(GET MUJOCO_LIBRARIES 0 MUJOCO_LIBRARY)
      message(STATUS "MuJoCo library found: ${MUJOCO_LIBRARY}")
  else()
      message(FATAL_ERROR "MuJoCo library not found in ${CMAKE_PREFIX_PATH}/lib")
  endif()
elseif(APPLE)
  set(MUJOCO_LIBRARY "${CMAKE_PREFIX_PATH}/lib/libmujoco.dylib")
endif()

# Find the MuJoCo headers and libraries
find_path(MUJOCO_INCLUDE_DIR mujoco.h
  PATHS ${MUJOCO_INCLUDE_DIR}
)

find_library(MUJOCO_LIBRARY mujoco
  PATHS ${MUJOCO_LIBRARY}
)

set(IPOPT_LIBRARY_DIRS "${CMAKE_PREFIX_PATH}/lib")
set(IPOPT_INCLUDE_DIRS "${CMAKE_PREFIX_PATH}/include")

message("ipopt binary path is: " ${IPOPT_LIBRARY_DIRS} )
message("ipopt include path is: " ${IPOPT_INCLUDE_DIRS} )

include_directories(${IPOPT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
link_directories(${IPOPT_LIBRARY_DIRS})


# Check if MuJoCo was found
if(NOT MUJOCO_LIBRARY OR NOT MUJOCO_INCLUDE_DIR)
  message(FATAL_ERROR "MuJoCo library or headers not found in the Conda environment. Please ensure MuJoCo is installed in the Conda environment.")
else()
  message(STATUS "MuJoCo library found: ${MUJOCO_LIBRARY}")
  message(STATUS "MuJoCo headers found: ${MUJOCO_INCLUDE_DIR}")
endif()

# Link the manually found MuJoCo
add_library(mujoco STATIC IMPORTED)
set_target_properties(mujoco PROPERTIES
  IMPORTED_LOCATION ${MUJOCO_LIBRARY}
  INTERFACE_INCLUDE_DIRECTORIES ${MUJOCO_INCLUDE_DIR}
)


# Add project dependencies
add_project_dependency(pinocchio REQUIRED)
add_project_dependency(crocoddyl REQUIRED)
add_project_dependency(eigenpy REQUIRED)

if(BUILD_PYTHON_INTERFACE)
  FINDPYTHON()
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
  SEARCH_FOR_BOOST_PYTHON()
endif()

# Define headers and sources
set(${PROJECT_NAME}_HEADERS
  include/${CUSTOM_HEADER_DIR}/fwd.hpp
  include/${CUSTOM_HEADER_DIR}/python.hpp
  include/${CUSTOM_HEADER_DIR}/actuations/floating-base-manipulation.hpp
  include/${CUSTOM_HEADER_DIR}/actions/differential-contact-mujoco.hpp
  include/${CUSTOM_HEADER_DIR}/actions/differential-contact-mujoco.hxx
  include/${CUSTOM_HEADER_DIR}/actions/integrated-contact-mujoco.hpp
  include/${CUSTOM_HEADER_DIR}/actions/integrated-contact-mujoco.hxx
  include/${CUSTOM_HEADER_DIR}/utils/pin-mj-conversion-finger.hpp
  include/${CUSTOM_HEADER_DIR}/utils/pyboost11-converter.hpp
)

set(${PROJECT_NAME}_SOURCES
  src/utils/pin-mj-conversion-finger.cpp
)

# Create a library target for mim-cio
add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)


# Add executable for the main program
target_link_libraries(${PROJECT_NAME} ipopt)
add_executable(${PROJECT_NAMESPACE}-${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} pinocchio::pinocchio)
target_link_libraries(${PROJECT_NAME} crocoddyl::crocoddyl)
target_link_libraries(${PROJECT_NAME} mujoco)
target_link_libraries(${PROJECT_NAMESPACE}-${PROJECT_NAME} ${PROJECT_NAME})
target_include_directories(${PROJECT_NAMESPACE}-${PROJECT_NAME} PRIVATE ${MUJOCO_INCLUDE_DIR})

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")

# Build Python interface if enabled
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings)
  target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
endif()

# Install Python bindings
if(BUILD_PYTHON_INTERFACE)
  install(TARGETS ${PY_NAME}_pywrap DESTINATION ${${PY_NAME}_INSTALL_DIR})
endif()
# Add the main include directory to the parent project's target

# Installation
install(TARGETS ${PROJECT_NAME} EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)
install(TARGETS ${PROJECT_NAMESPACE}-${PROJECT_NAME} RUNTIME DESTINATION bin)
