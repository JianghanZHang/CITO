file(REMOVE_RECURSE
  "doc/cito.doxytag"
  "doc/doxygen-html"
  "doc/doxygen.log"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/release_citation_cff.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
