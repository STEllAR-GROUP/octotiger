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

add_executable(modularPhysicsTests modularPhysics.cpp)
target_include_directories(modularPhysicsTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_features(modularPhysicsTests PRIVATE cxx_std_20)
if(NOT TARGET octotigerSubgrid)
  include(../cmake/modularLibraries.cmake)
endif()
target_link_libraries(modularPhysicsTests PRIVATE OctoTiger::Subgrid)
add_test(NAME modularPhysics COMMAND modularPhysicsTests)

add_executable(hydroExchangeTests hydroExchange.cpp)
target_link_libraries(hydroExchangeTests PRIVATE OctoTiger::Subgrid)
add_test(NAME hydroExchange COMMAND hydroExchangeTests)

add_executable(radiationExchangeTests radiationExchange.cpp)
target_link_libraries(radiationExchangeTests PRIVATE OctoTiger::Subgrid)
add_test(NAME radiationExchange COMMAND radiationExchangeTests)

if(TARGET OctoTiger::Output)
  add_executable(modularSiloTests modularSilo.cpp)
  target_link_libraries(modularSiloTests PRIVATE OctoTiger::Output)
  add_test(NAME modularSilo COMMAND modularSiloTests)
  set_tests_properties(modularSilo PROPERTIES LABELS "modular;silo")
endif()

option(OCTOTIGER_WITH_MODULAR_RUNTIME_TESTS
  "Run the modular HPX application and Silo emission smoke tests" OFF)
if(OCTOTIGER_WITH_MODULAR_RUNTIME_TESTS)
  if(NOT TARGET octotiger)
    message(FATAL_ERROR "Modular runtime tests require the full application build")
  endif()
  foreach(dimensionCount RANGE 1 3)
    add_test(NAME modularRuntime${dimensionCount}D
      COMMAND ${CMAKE_COMMAND}
        -Dapplication=$<TARGET_FILE:octotiger>
        -DdimensionCount=${dimensionCount}
        -DtestRoot=${CMAKE_CURRENT_BINARY_DIR}/modularSmoke
        -P ${CMAKE_CURRENT_SOURCE_DIR}/modularSmoke.cmake)
    set_tests_properties(modularRuntime${dimensionCount}D PROPERTIES
      TIMEOUT 180 LABELS "modular;runtime;silo")
    add_test(NAME modularRadiationRuntime${dimensionCount}D
      COMMAND ${CMAKE_COMMAND}
        -Dapplication=$<TARGET_FILE:octotiger>
        -DdimensionCount=${dimensionCount} -DphysicsMode=radiation
        -DtestRoot=${CMAKE_CURRENT_BINARY_DIR}/modularSmoke
        -P ${CMAKE_CURRENT_SOURCE_DIR}/modularSmoke.cmake)
    set_tests_properties(modularRadiationRuntime${dimensionCount}D PROPERTIES
      TIMEOUT 180 LABELS "modular;runtime;radiation;silo")
  endforeach()
  add_test(NAME modularMixedRuntime2D COMMAND ${CMAKE_COMMAND}
    -Dapplication=$<TARGET_FILE:octotiger> -DdimensionCount=2 -DphysicsMode=mixed
    -DtestRoot=${CMAKE_CURRENT_BINARY_DIR}/modularSmoke
    -P ${CMAKE_CURRENT_SOURCE_DIR}/modularSmoke.cmake)
  set_tests_properties(modularMixedRuntime2D PROPERTIES TIMEOUT 180 LABELS "modular;runtime;radiation;silo")
endif()

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

add_executable(greyOpacityTests greyOpacity.cpp)
target_include_directories(greyOpacityTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_features(greyOpacityTests PRIVATE cxx_std_20)
add_test(NAME greyOpacity COMMAND greyOpacityTests)
if(Python3_Interpreter_FOUND)
  add_test(NAME greyOpacitySources COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../verification_results/radiation/validate_so.py"
    --cxx "${CMAKE_CXX_COMPILER}" --opacity-checks)
endif()

if(Python3_Interpreter_FOUND)
  add_test(NAME radiationSubcycleReference COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/validateFinalRadiation.py" --cxx "${CMAKE_CXX_COMPILER}")
  add_test(NAME verificationHarness COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}/../verification_results/tests")
  set_tests_properties(verificationHarness PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()
