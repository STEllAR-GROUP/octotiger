//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifdef OCTOTIGER_HAVE_CUDA

#define BOOST_NO_CXX11_ALLOCATOR
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <hpx/async_cuda/cuda_executor.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include <stream_manager.hpp>
using pool_strategy = multi_gpu_round_robin_pool<hpx::cuda::cuda_executor, round_robin_pool<hpx::cuda::cuda_executor>>;


#endif
