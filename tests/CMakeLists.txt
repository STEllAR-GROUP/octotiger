if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(OctoTigerRadiationTests LANGUAGES CXX)
  enable_testing()
endif()

# The M1 algebra does not require the application dependencies (HPX, Silo, etc.).
add_executable(radiationM1Tests radiationM1.cpp)
target_include_directories(radiationM1Tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_features(radiationM1Tests PRIVATE cxx_std_20)
add_test(NAME radiationM1 COMMAND radiationM1Tests)

add_executable(fpeGuardTests fpeGuard.cpp)
target_include_directories(fpeGuardTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_features(fpeGuardTests PRIVATE cxx_std_20)
add_test(NAME fpeGuard COMMAND fpeGuardTests)

# This exercises the same compatibility helpers as the application parser.
# Boost is available when tests are built from the supported top-level build;
# the dependency-free standalone test project simply skips this target.
if(TARGET Boost::program_options)
  add_executable(optionsCompatibilityParserTests options_compatibility.cpp)
  target_include_directories(optionsCompatibilityParserTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
  target_compile_features(optionsCompatibilityParserTests PRIVATE cxx_std_20)
  target_link_libraries(optionsCompatibilityParserTests PRIVATE Boost::program_options)
  add_test(NAME optionsCompatibilityParser COMMAND optionsCompatibilityParserTests)
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME optionsCompatibility
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/options_compatibility_test.py"
  )
endif()
