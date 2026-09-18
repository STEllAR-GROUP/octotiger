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
