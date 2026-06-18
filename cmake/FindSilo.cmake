# Copyright (c) 2008-2012 Sandia Corporation, Kitware Inc.
# Copyright (c) 2014-2014 Andreas Schäfer
# Copyright (c) 2019 Parsa Amini

if(NOT MSVC)
  # HDF5 is needed for linking on non-MSVC builds
  find_package(HDF5 REQUIRED)
  find_package(ZLIB REQUIRED)
  find_package(Threads REQUIRED)

  if(NOT TARGET octotiger::hdf5)
    add_library(octotiger::hdf5 INTERFACE IMPORTED)

    set_property(TARGET octotiger::hdf5
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${HDF5_INCLUDE_DIRS})

    set_property(TARGET octotiger::hdf5
      PROPERTY INTERFACE_LINK_LIBRARIES
      "${HDF5_LIBRARIES};ZLIB::ZLIB;dl;Threads::Threads")
  endif()

  find_path(Silo_INCLUDE_DIR silo.h
    PATHS
      /usr/local/include
      /usr/include
      ${Silo_DIR}/include)

  find_library(Silo_LIBRARY NAMES siloh5
    PATHS
      /usr
      /usr/local
      ${Silo_DIR}
    PATH_SUFFIXES lib lib64)

  find_program(Silo_BROWSER NAMES browser
    PATHS
      /usr/bin
      /usr/local/bin
      ${Silo_DIR}/bin)
else()
  find_path(Silo_H_INCLUDE_DIR silo.h
    PATHS ${Silo_DIR}/SiloWindows/include)

  find_path(Silo_X_INCLUDE_DIR silo_exports.h
    PATHS ${Silo_DIR}/src/silo)

  set(Silo_INCLUDE_DIR ${Silo_H_INCLUDE_DIR} ${Silo_X_INCLUDE_DIR})

  unset(Silo_H_INCLUDE_DIR)
  unset(Silo_X_INCLUDE_DIR)

  find_library(Silo_LIBRARY NAMES silohdf5
    PATHS ${Silo_DIR}/SiloWindows/MSVC2012/x64/Release)

  find_program(Silo_BROWSER NAMES browser
    PATHS ${Silo_DIR}/SiloWindows/MSVC2012/x64/Release)
endif()

set(Silo_FOUND OFF)

if(Silo_INCLUDE_DIR)
  if(Silo_LIBRARY)
    set(Silo_LIBRARIES ${Silo_LIBRARY})
    set(Silo_FOUND ON)
  else()
    if(Silo_FIND_REQUIRED)
      message(SEND_ERROR "Unable to find the requested Silo libraries.")
    endif()
  endif()
endif()

mark_as_advanced(
  Silo_INCLUDE_DIR
  Silo_LIBRARY
  Silo_BROWSER)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Silo DEFAULT_MSG
  Silo_LIBRARY
  Silo_INCLUDE_DIR
  Silo_BROWSER)

if(Silo_FOUND AND NOT TARGET Silo::silo)
  add_library(Silo::silo INTERFACE IMPORTED)

  set_property(TARGET Silo::silo
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${Silo_INCLUDE_DIR})

  set_property(TARGET Silo::silo
    PROPERTY INTERFACE_LINK_LIBRARIES
    ${Silo_LIBRARY})
endif()

if(NOT MSVC AND TARGET Silo::silo)
  set_property(TARGET Silo::silo
    APPEND PROPERTY INTERFACE_LINK_LIBRARIES
    octotiger::hdf5)
endif()
