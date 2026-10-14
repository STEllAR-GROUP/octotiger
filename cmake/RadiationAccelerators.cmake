# Radiation-only accelerators can be enabled without changing hydro/gravity kernels.
set(_radiation_cuda_default OFF)
set(_radiation_kokkos_default OFF)
if(OCTOTIGER_WITH_RADIATION AND OCTOTIGER_WITH_CUDA)
  set(_radiation_cuda_default ON)
endif()
if(OCTOTIGER_WITH_RADIATION AND OCTOTIGER_WITH_KOKKOS)
  set(_radiation_kokkos_default ON)
endif()
option(OCTOTIGER_WITH_RADIATION_CUDA "Build the opt-in radiation CUDA backend" ${_radiation_cuda_default})
option(OCTOTIGER_WITH_RADIATION_KOKKOS "Build the opt-in radiation Kokkos backend" ${_radiation_kokkos_default})
unset(_radiation_cuda_default)
unset(_radiation_kokkos_default)
get_filename_component(_OCTOTIGER_RADIATION_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Language initialization must happen in directory scope: doing it inside the
# target helper loses CUDA compiler feature information in standalone builds.
if(OCTOTIGER_WITH_RADIATION_KOKKOS)
  # Keep the normal C++ compiler rather than redirecting the entire application
  # through nvcc_wrapper. CUDA-enabled Kokkos sources are explicitly CUDA below.
  find_package(Kokkos REQUIRED COMPONENTS separable_compilation)
endif()
if(OCTOTIGER_WITH_RADIATION_CUDA OR (OCTOTIGER_WITH_RADIATION_KOKKOS AND Kokkos_ENABLE_CUDA))
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
get_filename_component(_radiation_cxx_driver "${CMAKE_CXX_COMPILER}" NAME)
if(OCTOTIGER_WITH_RADIATION_KOKKOS AND Kokkos_ENABLE_CUDA AND NOT OCTOTIGER_WITH_KOKKOS
    AND NOT _radiation_cxx_driver MATCHES "nvcc|hipcc"
    AND NOT CMAKE_CXX_FLAGS MATCHES "--cuda|-x[ =]cuda")
  # nvcc_wrapper installations export CUDA link flags guarded as CXX. Ordinary
  # C++ consumers of our static library use the host linker, so those flags must
  # instead apply only when CMake actually links with the CUDA compiler.
  get_target_property(_radiation_kokkos_link_options Kokkos::kokkoscore INTERFACE_LINK_OPTIONS)
  if(_radiation_kokkos_link_options)
    string(REPLACE "LINK_LANGUAGE:CXX" "LINK_LANGUAGE:CUDA"
      _radiation_kokkos_link_options "${_radiation_kokkos_link_options}")
    set_property(TARGET Kokkos::kokkoscore PROPERTY INTERFACE_LINK_OPTIONS "${_radiation_kokkos_link_options}")
  endif()
  unset(_radiation_kokkos_link_options)
endif()
unset(_radiation_cxx_driver)

function(octotiger_add_radiation_accelerators)
  set(_source "${_OCTOTIGER_RADIATION_SOURCE_DIR}")
  add_library(radiation_accelerators STATIC "${_source}/src/radiation/accelerator.cpp")
  target_include_directories(radiation_accelerators PUBLIC "${_source}")
  target_compile_features(radiation_accelerators PUBLIC cxx_std_20)
  set_target_properties(radiation_accelerators PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if(OCTOTIGER_WITH_RADIATION_CUDA OR OCTOTIGER_WITH_RADIATION_KOKKOS)
    target_compile_definitions(radiation_accelerators PUBLIC OCTOTIGER_HAVE_RADIATION_ACCELERATORS)
  endif()

  if(OCTOTIGER_WITH_RADIATION_CUDA)
    add_library(radiation_accelerator_cuda STATIC "${_source}/src/radiation/accelerator_cuda.cu")
    target_include_directories(radiation_accelerator_cuda PRIVATE "${_source}")
    target_link_libraries(radiation_accelerator_cuda PRIVATE CUDA::cudart)
    set_target_properties(radiation_accelerator_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
      POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(radiation_accelerators PUBLIC OCTOTIGER_HAVE_RADIATION_CUDA)
    target_link_libraries(radiation_accelerators PRIVATE radiation_accelerator_cuda)
  endif()

  if(OCTOTIGER_WITH_RADIATION_KOKKOS)
    add_library(radiation_accelerator_kokkos STATIC "${_source}/src/radiation/accelerator_kokkos.cpp")
    target_include_directories(radiation_accelerator_kokkos PRIVATE "${_source}")
    target_link_libraries(radiation_accelerator_kokkos PRIVATE Kokkos::kokkos)
    target_compile_features(radiation_accelerator_kokkos PRIVATE cxx_std_20)
    set_target_properties(radiation_accelerator_kokkos PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(Kokkos_ENABLE_CUDA)
      set_source_files_properties("${_source}/src/radiation/accelerator_kokkos.cpp" PROPERTIES LANGUAGE CUDA)
      set_target_properties(radiation_accelerator_kokkos PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
      if(Kokkos_CUDA_ARCHITECTURES)
        set_target_properties(radiation_accelerator_kokkos PROPERTIES CUDA_ARCHITECTURES "${Kokkos_CUDA_ARCHITECTURES}")
      endif()
    endif()
    target_compile_definitions(radiation_accelerators PUBLIC OCTOTIGER_HAVE_RADIATION_KOKKOS)
    target_link_libraries(radiation_accelerators PRIVATE radiation_accelerator_kokkos)
  endif()
endfunction()
