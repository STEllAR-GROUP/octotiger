# The numerical fixtures also build without HPX or application dependencies.
cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(octotiger_fld_numerics LANGUAGES CXX)
  enable_testing()
  set(OCTOTIGER_WITH_GRIDDIM 8 CACHE STRING "Interior cells per block axis")
endif()

add_compile_definitions(OCTOTIGER_GRIDDIM=${OCTOTIGER_WITH_GRIDDIM})

get_filename_component(fld_source_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(fld_numerics test_numerics.cpp
  "${fld_source_root}/src/multigrid/block.cpp"
  "${fld_source_root}/src/multigrid/reaction_diffusion.cpp")
target_include_directories(fld_numerics PRIVATE "${fld_source_root}")
target_compile_features(fld_numerics PRIVATE cxx_std_17)
add_test(NAME fld_numerics COMMAND fld_numerics)
set_tests_properties(fld_numerics PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS fld)

add_executable(fld_hydro_numerics test_hydro.cpp)
target_include_directories(fld_hydro_numerics PRIVATE "${fld_source_root}")
target_compile_features(fld_hydro_numerics PRIVATE cxx_std_17)
add_test(NAME fld_hydro_numerics COMMAND fld_hydro_numerics)
set_tests_properties(fld_hydro_numerics PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS fld)

add_executable(fld_thermal_numerics test_thermal.cpp
  "${fld_source_root}/src/fld_grid.cpp"
  "${fld_source_root}/src/multigrid/block.cpp"
  "${fld_source_root}/src/multigrid/reaction_diffusion.cpp")
target_include_directories(fld_thermal_numerics PRIVATE "${fld_source_root}")
target_compile_features(fld_thermal_numerics PRIVATE cxx_std_17)
add_test(NAME fld_thermal_numerics COMMAND fld_thermal_numerics)
set_tests_properties(fld_thermal_numerics PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS fld)

foreach(fixture opacity material boundary scf_geometry)
  add_executable(fld_${fixture}_numerics test_${fixture}.cpp)
  target_include_directories(fld_${fixture}_numerics PRIVATE "${fld_source_root}")
  target_compile_features(fld_${fixture}_numerics PRIVATE cxx_std_17)
  add_test(NAME fld_${fixture}_numerics COMMAND fld_${fixture}_numerics)
  set_tests_properties(fld_${fixture}_numerics PROPERTIES PROCESSORS 1 TIMEOUT 120 LABELS fld)
endforeach()

find_package(Threads REQUIRED)
target_link_libraries(fld_opacity_numerics PRIVATE Threads::Threads)

if(TARGET octotiger)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  foreach(case uniform pulse amr rotation restart dual_energy nonconvergence)
    add_test(NAME fld_application_${case}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_fld.py
        --exe $<TARGET_FILE:octotiger>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${case}
        --case ${case})
    set_tests_properties(fld_application_${case} PROPERTIES
      PROCESSORS 1 TIMEOUT 660 LABELS fld RUN_SERIAL TRUE)
  endforeach()
endif()

if(TARGET octotiger)
  foreach(case cold heated independent amr restart shock)
    add_test(NAME fld_wd_${case}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_dwd_fld.py
        --exe $<TARGET_FILE:octotiger>
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/wd_${case}
        --case ${case})
    set_tests_properties(fld_wd_${case} PROPERTIES
      PROCESSORS 1 TIMEOUT 660 LABELS "fld;wd" RUN_SERIAL TRUE)
  endforeach()
endif()
