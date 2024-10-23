/*
 * This file has been automatically generated by the jrl-cmakemodules.
 * Please see https://github.com/jrl-umi3218/jrl-cmakemodules/blob/master/warning.hh.cmake for details.
*/

#ifndef CITO_WARNING_HH
# define CITO_WARNING_HH

// Emits a warning in a portable way.
//
// To emit a warning, one can insert:
//
// #pragma message CITO_WARN("your warning message here")
//
// The use of this syntax is required as this is /not/ a standardized
// feature of C++ language or preprocessor, even if most of the
// compilers support it.

# define CITO_WARN_STRINGISE_IMPL(x) #x
# define CITO_WARN_STRINGISE(x) \
         CITO_WARN_STRINGISE_IMPL(x)
# ifdef __GNUC__
#   define CITO_WARN(exp) ("WARNING: " exp)
# else
#  ifdef _MSC_VER
#   define FILE_LINE_LINK __FILE__ "(" \
           CITO_WARN_STRINGISE(__LINE__) ") : "
#   define CITO_WARN(exp) (FILE_LINE_LINK "WARNING: " exp)
#  else
// If the compiler is not recognized, drop the feature.
#   define CITO_WARN(MSG) /* nothing */
#  endif // __MSVC__
# endif // __GNUC__

#endif //! CITO_WARNING_HH
