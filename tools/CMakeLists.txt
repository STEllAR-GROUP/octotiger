# Copyright (c) 2019 AUTHORS
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# Set up bfilter target
################################################################################
add_hpx_executable(
  bfilter
  DEPENDENCIES
    Silo::silo Boost::boost Boost::program_options
  SOURCES
    bfilter/bfilter.cpp
)


################################################################################
# Set up silo_compare target
################################################################################
add_executable(silo_compare compare/compare.cpp)
target_link_libraries(silo_compare Silo::silo)
set_property(TARGET silo_compare PROPERTY FOLDER "Tools")
if(MSVC)
  target_compile_definitions(silo_compare PRIVATE
    _CRT_SECURE_NO_WARNINGS)
  target_compile_options(silo_compare PRIVATE /EHsc)
endif()


################################################################################
# Set up silo_convert
################################################################################
add_hpx_executable(
  silo_convert
  DEPENDENCIES
    Silo::silo Boost::boost Boost::program_options
  SOURCES
    silo_convert/plain_silo.cpp
    silo_convert/silo_convert.cpp 
    silo_convert/silo_convert.hpp    
    silo_convert/silo_out.cpp    
    silo_convert/split_silo.cpp 
)

################################################################################
# Set up silo_post
################################################################################
add_hpx_executable(
  silo_post
  DEPENDENCIES
    Silo::silo Boost::boost Boost::program_options
  SOURCES
    silo_post/silo_post.cpp 
)


################################################################################
# Set up gen_rotating_star_init target
################################################################################
add_executable(gen_rotating_star_init gen_rotating_star_init/make_bin.cpp)
set_property(TARGET gen_rotating_star_init PROPERTY FOLDER "Tools")
if(MSVC)
  target_compile_definitions(gen_rotating_star_init PRIVATE
    _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(gen_rotating_star_init PRIVATE
    -Wno-ignored-attributes -Wno-attributes -Wno-deprecated-declarations
    -Wno-unused-result -ffast-math -O3)
  target_link_libraries(gen_rotating_star_init Threads::Threads)
endif()

################################################################################
# Set up silo_planes target
################################################################################
add_executable(silo_planes silo_planes/silo_planes.cpp)

target_link_libraries(silo_planes Silo::silo)

set_property(TARGET silo_planes PROPERTY FOLDER "Tools")
if(MSVC)
  target_compile_definitions(silo_planes PRIVATE
    _CRT_SECURE_NO_WARNINGS)
  target_compile_options(silo_planes PRIVATE /EHsc)
endif()


################################################################################
# Set up silo_counter target
################################################################################
add_executable(silo_counter silo_counter/silo_counter.cpp)

target_link_libraries(silo_counter Silo::silo)

set_property(TARGET silo_counter PROPERTY FOLDER "Tools")
if(MSVC)
  target_compile_options(silo_counter PRIVATE /EHsc)
endif()
