# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

project(octotiger CXX)

cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)

################################################################################
# enable solution folders for MSVC
if(MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

################################################################################
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(MPI REQUIRED)

################################################################################
include_directories(${CMAKE_SOURCE_DIR}/src ~/include)

#set( CMAKE_CXX_FLAGS "-mt_mpi -xHOST -std=c++14" )

add_hpx_executable(
  octotiger
  SOURCES 	
      	src/eos.cpp
		src/geometry.cpp
		src/grid.cpp
		src/grid_fmm.cpp
		src/grid_output.cpp
		src/grid_scf.cpp
		src/lane_emden.cpp
		src/main.cpp
		src/new.cpp
		src/node_client.cpp
		src/node_location.cpp
		src/node_server.cpp
        src/node_server/check_for_refinement.cpp
        src/node_server/copy_to_locality.cpp
        src/node_server/diagnostics.cpp
        src/node_server/find_omega_part.cpp
        src/node_server/force_nodes_to_exist.cpp
        src/node_server/form_tree.cpp
        src/node_server/get_child_client.cpp
        src/node_server/get_nieces.cpp
        src/node_server/get_ptr.cpp
        src/node_server/gravity_boundary.cpp
        src/node_server/gravity_expansions.cpp
        src/node_server/gravity_multipoles.cpp
        src/node_server/hydro_boundary.cpp
        src/node_server/hydro_children.cpp
        src/node_server/hydro_flux_correct.cpp
        src/node_server/line_of_centers.cpp
        src/node_server/load.cpp
        src/node_server/regrid.cpp
        src/node_server/regrid_gather.cpp
        src/node_server/regrid_scatter.cpp
        src/node_server/save.cpp
        src/node_server/set_aunt.cpp
        src/node_server/set_grid.cpp
        src/node_server/solve_gravity.cpp
        src/node_server/start_run.cpp
        src/node_server/step.cpp
        src/node_server/timestep_driver.cpp
        src/node_server/timestep_driver_ascend.cpp
        src/node_server/timestep_driver_descend.cpp
        src/node_server/velocity_inc.cpp
 		src/options.cpp
		src/problem.cpp
		src/real.cpp
		src/roe.cpp
		src/stack_trace.cpp
		src/taylor.cpp
     	src/util.cpp
   HEADERS 
        src/diagnostics.hpp
		src/channel.hpp
		src/defs.hpp
		src/eos.hpp
		src/future.hpp
		src/geometry.hpp
		src/grid.hpp
		src/lane_emden.hpp
		src/node_client.hpp
		src/node_location.hpp
		src/node_server.hpp
		src/options.hpp
		src/problem.hpp
		src/real.hpp
		src/util.hpp
		src/roe.hpp
		src/simd.hpp
		src/space_vector.hpp
		src/taylor.hpp
  FOLDER  "Octotiger"
)


