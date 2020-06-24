//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(OCTOTIGER_HAVE_CUDA)
#if defined(__CUDACC__)
#if !defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif
#define CUDA_CALLABLE_METHOD __device__
#else
#define CUDA_CALLABLE_METHOD
#endif
#else
#define CUDA_CALLABLE_METHOD
#endif



