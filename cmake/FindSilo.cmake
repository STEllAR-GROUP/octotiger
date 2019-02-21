# Copyright (c) 2008-2012 Sandia Corporation, Kitware Inc.
# Copyright (c) 2014-2014 Andreas Schäfer
# Copyright (c) 2019 Parsa Amini
#
# Sandia National Laboratories, New Mexico
# PO Box 5800
# Albuquerque, NM 87185
#
# Kitware Inc.
# 28 Corporate Drive
# Clifton Park, NY 12065
# USA
#
# Andreas Schäfer
# Informatik 3
# Martensstr. 3
# 91058 Erlangen
# Germany
#
# Under the terms of Contract DE-AC04-94AL85000, there is a
# non-exclusive license for use of this work by or on behalf of the
# U.S. Government.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#  * Neither the name of Kitware nor the names of any contributors may
#    be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ========================================================================
#
# Try to find Silo library and headers. Define Silo_DIR if Silo is
# installed in a non-standard directory.
#
# This file sets the following variables:
#
# Silo_INCLUDE_DIR, where to find silo.h, etc.
# Silo_LIBRARIES, the libraries to link against
# Silo_FOUND, If false, do not try to use Silo.
#
# Also defined, but not for general use are:
# Silo_LIBRARY, the full path to the silo library.
# Silo_INCLUDE_PATH, for CMake backward compatibility

if(NOT MSVC)
  # HDF5 is needed for linking on non-MSVC builds
  find_package(HDF5 REQUIRED)
  find_package(ZLIB REQUIRED)
  find_package(Threads REQUIRED)

  add_library(octotiger::hdf5 INTERFACE IMPORTED)
  set_property(TARGET octotiger::hdf5
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS})
  set_property(TARGET octotiger::hdf5
    PROPERTY INTERFACE_LINK_LIBRARIES
    ${HDF5_LIBRARIES} ZLIB::ZLIB dl Threads::Threads)
  
  find_path(Silo_INCLUDE_DIR silo.h
    PATHS /usr/local/include
    /usr/include
    ${Silo_DIR}/include)

  find_library(Silo_LIBRARY NAMES siloh5
    PATHS /usr
    /usr/local
    ${Silo_DIR}
    PATH_SUFFIXES lib lib64)

  find_program(Silo_BROWSER NAMES browser
    PATHS /usr/bin
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
    if(Silo_FIND_REQURIED)
      message(SEND_ERROR "Unable to find the requested Silo libraries.")
    endif()
  endif()
endif()

mark_as_advanced(
  Silo_INCLUDE_DIR
  Silo_LIBRARY
  Silo_BROWSER)

# Handle the QUIETLY and REQUIRED arguments and set Silo_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Silo DEFAULT_MSG
  Silo_LIBRARY Silo_INCLUDE_DIR Silo_BROWSER)

if(Silo_FOUND AND NOT TARGET Silo::silo)
  add_library(Silo::silo INTERFACE IMPORTED)
  set_property(TARGET Silo::silo
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Silo_INCLUDE_DIR})
  set_property(TARGET Silo::silo
    PROPERTY INTERFACE_LINK_LIBRARIES ${Silo_LIBRARY})
endif()

if(NOT MSVC)
  set_property(TARGET Silo::silo
    APPEND PROPERTY INTERFACE_LINK_LIBRARIES octotiger::hdf5)
endif()
