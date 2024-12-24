set(${PY_NAME}_SOURCES
  main.cpp
  differential-contact-mujoco.cpp
  integrated-contact-mujoco.cpp
  floating-base-manipulation.cpp
)

set(${PROJECT_NAME}_HEADERS
  utils/copyable.hpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=0")

# Add the Python wrapper library (using only the sources)
add_library(${PY_NAME}_pywrap SHARED ${${PY_NAME}_SOURCES})

set_target_properties(${PY_NAME}_pywrap PROPERTIES
  INSTALL_RPATH "\$ORIGIN/../../.."
  LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}_pywrap
  PREFIX ""
  SUFFIX ${PYTHON_EXT_SUFFIX})

# # Link the main project library and eigenpy to the Python wrapper
# target_link_libraries(${PY_NAME}_pywrap PUBLIC ${PROJECT_NAME} 
#                                                ${CMAKE_SOURCE_DIR} 
#                                                ${PARENT_DIR}/include
#                                                eigenpy::eigenpy)
target_link_libraries(${PY_NAME}_pywrap PUBLIC ${PROJECT_NAME}
                                               eigenpy::eigenpy)

target_include_directories(${PY_NAME}_pywrap PUBLIC 
                           ${CMAKE_SOURCE_DIR}/include
                           ${PARENT_DIR}/include)

# Print include directories for debugging
message("Include directories: ${CMAKE_INCLUDE_PATH}")

# Set target properties
set_target_properties(${PY_NAME}_pywrap PROPERTIES
  INSTALL_RPATH "\$ORIGIN/../../.."
  LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}_pywrap
  PREFIX ""
  SUFFIX ${PYTHON_EXT_SUFFIX}
)

# Link Boost Python (if needed)
TARGET_LINK_BOOST_PYTHON(${PY_NAME}_pywrap PRIVATE)

# Install the library
install(TARGETS ${PY_NAME}_pywrap DESTINATION ${${PY_NAME}_INSTALL_DIR})

# Install Python scripts
set(${PY_NAME}_PYTHON
  __init__.py
)

foreach(python ${${PY_NAME}_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME} ${python})
endforeach()
