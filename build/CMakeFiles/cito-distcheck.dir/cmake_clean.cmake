file(REMOVE_RECURSE
  "doc/cito.doxytag"
  "doc/doxygen-html"
  "doc/doxygen.log"
  "CMakeFiles/cito-distcheck"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/cito-distcheck.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
