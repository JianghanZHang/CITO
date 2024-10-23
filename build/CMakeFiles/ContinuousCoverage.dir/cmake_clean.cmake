file(REMOVE_RECURSE
  "doc/cito.doxytag"
  "doc/doxygen-html"
  "doc/doxygen.log"
  "CMakeFiles/ContinuousCoverage"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/ContinuousCoverage.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
