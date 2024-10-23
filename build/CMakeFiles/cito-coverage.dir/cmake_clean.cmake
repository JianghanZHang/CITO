file(REMOVE_RECURSE
  "doc/cito.doxytag"
  "doc/doxygen-html"
  "doc/doxygen.log"
  "CMakeFiles/cito-coverage"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/cito-coverage.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
