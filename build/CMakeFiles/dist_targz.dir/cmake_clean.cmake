file(REMOVE_RECURSE
  "doc/cito.doxytag"
  "doc/doxygen-html"
  "doc/doxygen.log"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/dist_targz.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
