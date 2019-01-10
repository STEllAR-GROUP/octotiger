# Copyright (c) 2008-2012 Sandia Corporation, Kitware Inc.
# Copyright (c) 2014-2014 Andreas Schäfer
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
# Try to find HDF5 library and headers. Define HDF5_ROOT if HDF5 is
# installed in a non-standard directory.
#
# This file sets the following variables:
#
# HDF5_INCLUDE_DIR, where to find HDF5.h, etc.
# HDF5_LIBRARIES, the libraries to link against
# HDF5_FOUND, If false, do not try to use HDF5.
#
# Also defined, but not for general use are:
# HDF5_LIBRARY, the full path to the HDF5 library.
# HDF5_INCLUDE_PATH, for CMake backward compatibility

FIND_PATH( HDF5_INCLUDE_DIR HDF5.h
  PATHS /usr/local/include
  /usr/include
  ${HDF5_ROOT}/include
)

FIND_LIBRARY( HDF5_LIBRARY NAMES hdf5
  PATHS /usr/lib
  /usr/lib64
  /usr/local/lib
  ${HDF5_ROOT}/lib
  ${HDF5_ROOT}/lib64
)

SET(HDF5_FOUND "NO" )
IF(HDF5_INCLUDE_DIR)
  IF(HDF5_LIBRARY)

    SET(HDF5_LIBRARIES ${HDF5_LIBRARY})
    SET(HDF5_FOUND "YES" )

  ELSE(HDF5_LIBRARY)
    IF(HDF5_FIND_REQURIED)
      message(SEND_ERROR "Unable to find the requested HDF5 libraries.")
    ENDIF(HDF5_FIND_REQURIED)
  ENDIF(HDF5_LIBRARY)
ENDIF(HDF5_INCLUDE_DIR)

# handle the QUIETLY and REQUIRED arguments and set HDF5_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(HDF5 DEFAULT_MSG HDF5_LIBRARY HDF5_INCLUDE_DIR)

MARK_AS_ADVANCED(
  HDF5_INCLUDE_DIR
  HDF5_LIBRARY
)
