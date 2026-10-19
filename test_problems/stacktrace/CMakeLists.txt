# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.

if(WIN32)
  return()
endif()

find_package(Python3 QUIET COMPONENTS Interpreter)
find_package(Threads REQUIRED)

add_executable(stacktrace_fixture fixture.cpp)
target_link_libraries(stacktrace_fixture PRIVATE octotiger_stacktrace Threads::Threads)

add_hpx_executable(stacktrace_hpx_fixture
  SOURCES hpx_fixture.cpp
  DEPENDENCIES octotiger_stacktrace)

# Source locations must remain available even when the parent build is Release.
foreach(target stacktrace_fixture stacktrace_hpx_fixture)
  set_target_properties(${target} PROPERTIES ENABLE_EXPORTS ON)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -g -fno-omit-frame-pointer -fno-optimize-sibling-calls)
  endif()
endforeach()

add_test(NAME test_problems.stacktrace.lifecycle
  COMMAND stacktrace_fixture lifecycle)
set_tests_properties(test_problems.stacktrace.lifecycle PROPERTIES TIMEOUT 30 LABELS stacktrace)
if(NOT Python3_Interpreter_FOUND)
  message(STATUS "Python 3 unavailable; skipping stacktrace subprocess tests")
  return()
endif()
add_test(NAME test_problems.stacktrace.thread_crash
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
    thread-crash $<TARGET_FILE:stacktrace_fixture>)
set_tests_properties(test_problems.stacktrace.thread_crash PROPERTIES TIMEOUT 30 LABELS stacktrace)

if(HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_TCP)
  foreach(mode distributed distributed-crash)
    add_test(NAME test_problems.stacktrace.${mode}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
        ${mode} $<TARGET_FILE:stacktrace_hpx_fixture>)
    set_tests_properties(test_problems.stacktrace.${mode} PROPERTIES
      TIMEOUT 60 PROCESSORS 2 RUN_SERIAL TRUE LABELS stacktrace)
  endforeach()
endif()
