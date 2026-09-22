# HPX-independent mesh and numerical libraries, shared by the application
# build and the dependency-free standalone test build.
set(modularSourceRoot "${CMAKE_CURRENT_LIST_DIR}/..")

add_library(octotigerMesh STATIC
  ${modularSourceRoot}/src/mesh.cpp
  ${modularSourceRoot}/src/mesh/meshHierarchy.cpp)
add_library(octotigerPhysics STATIC
  ${modularSourceRoot}/src/physics/finiteVolume.cpp)
target_link_libraries(octotigerPhysics PUBLIC octotigerMesh)

add_library(octotigerHydro STATIC
  ${modularSourceRoot}/src/hydro/hydroSystem.cpp)
target_link_libraries(octotigerHydro PUBLIC octotigerPhysics octotigerMesh)
add_library(octotigerRadiation STATIC
  ${modularSourceRoot}/src/radiation/radiationTransport.cpp)
target_link_libraries(octotigerRadiation PUBLIC octotigerPhysics octotigerMesh)
add_library(octotigerGravity STATIC
  ${modularSourceRoot}/src/gravity/gravityFields.cpp)
target_link_libraries(octotigerGravity PUBLIC octotigerMesh)

add_library(octotigerSubgrid STATIC
  ${modularSourceRoot}/src/subgrid/hydroExchange.cpp
  ${modularSourceRoot}/src/subgrid/radiationExchange.cpp
  ${modularSourceRoot}/src/subgrid/shadowError.cpp
  ${modularSourceRoot}/src/subgrid/subgrid.cpp
  ${modularSourceRoot}/src/subgrid/subgridStepper.cpp)
target_link_libraries(octotigerSubgrid PUBLIC
  octotigerMesh octotigerHydro octotigerRadiation octotigerGravity)

set(modularCppStandard 20)
if(DEFINED OCTOTIGER_CXX_STANDARD)
  set(modularCppStandard ${OCTOTIGER_CXX_STANDARD})
endif()
foreach(modularTarget IN ITEMS octotigerMesh octotigerPhysics octotigerHydro
    octotigerRadiation octotigerGravity octotigerSubgrid)
  target_include_directories(${modularTarget} PUBLIC "${modularSourceRoot}")
  target_compile_features(${modularTarget} PUBLIC cxx_std_20)
  set_target_properties(${modularTarget} PROPERTIES
    CXX_STANDARD ${modularCppStandard}
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    FOLDER "Octo-Tiger/Modules")
endforeach()

add_library(OctoTiger::Mesh ALIAS octotigerMesh)
add_library(OctoTiger::Physics ALIAS octotigerPhysics)
add_library(OctoTiger::Hydro ALIAS octotigerHydro)
add_library(OctoTiger::Radiation ALIAS octotigerRadiation)
add_library(OctoTiger::Gravity ALIAS octotigerGravity)
add_library(OctoTiger::Subgrid ALIAS octotigerSubgrid)

if(TARGET Silo::silo)
  add_library(octotigerOutput STATIC ${modularSourceRoot}/src/subgrid/modularSilo.cpp)
  target_link_libraries(octotigerOutput PUBLIC OctoTiger::Subgrid Silo::silo)
  set_target_properties(octotigerOutput PROPERTIES
    CXX_STANDARD ${modularCppStandard} CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON FOLDER "Octo-Tiger/Modules")
  add_library(OctoTiger::Output ALIAS octotigerOutput)
endif()
