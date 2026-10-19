# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.

# Prefer installed dependencies (including Spack prefixes), with GCC's bundled
# copy as a fallback. Clang and GPU compiler wrappers need not supply their own.
find_path(Libbacktrace_INCLUDE_DIR backtrace.h
  HINTS ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
find_library(Libbacktrace_LIBRARY NAMES backtrace
  HINTS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Libbacktrace REQUIRED_VARS
  Libbacktrace_LIBRARY Libbacktrace_INCLUDE_DIR)
mark_as_advanced(Libbacktrace_LIBRARY Libbacktrace_INCLUDE_DIR)

if(Libbacktrace_FOUND AND NOT TARGET Libbacktrace::libbacktrace)
  add_library(Libbacktrace::libbacktrace UNKNOWN IMPORTED)
  set_target_properties(Libbacktrace::libbacktrace PROPERTIES
    IMPORTED_LOCATION "${Libbacktrace_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Libbacktrace_INCLUDE_DIR}")
endif()
