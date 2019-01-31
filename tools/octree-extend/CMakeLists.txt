cmake_minimum_required(VERSION 3.10)
project(full_octree_id CXX)

add_executable(full_octree_id full_octree_id.cpp)

set_target_properties(full_octree_id PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(MSVC)
  set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY
    VS_STARTUP_PROJECT full_octree_id)
  configure_file(cmake/project.vcxproj.user.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.vcxproj.user @ONLY)
endif()
