#pragma once
#ifdef OCTOTIGER_WITH_CUDA
#ifdef __CUDACC__
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_CALLABLE_METHOD  __device__
#else
#define CUDA_CALLABLE_METHOD
#endif
#else
#define CUDA_CALLABLE_METHOD
#endif
