#pragma once
#ifdef OCTOTIGER_HAVE_CUDA

#define BOOST_NO_CXX11_ALLOCATOR
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <hpx/compute/cuda/target.hpp>
#include <hpx/include/compute.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace octotiger { namespace util {
    struct cuda_helper
    {
    public:
        using future_type = hpx::future<void>;

        // construct a cuBLAS stream
        cuda_helper(std::size_t device = 0)
          : target_(device)
        {
            stream_ = target_.native_handle().get_stream();
        }

        ~cuda_helper() {}

        // This is a simple wrapper for any cuBLAS call, pass in the same arguments
        // that you would use for a cuBLAS call except the cuBLAS handle which is omitted
        // as the wrapper will supply that for you
        template <typename Func, typename... Args>
        void operator()(Func&& cuda_function, Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(target_.native_handle().get_device()));

            // insert the cuBLAS handle in the arg list and call the cuBLAS function
            cuda_error(cuda_function(std::forward<Args>(args)..., stream_));
        }
        template <typename Func, typename... Args>
        cudaError_t pass_through(Func&& cuda_function, Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(target_.native_handle().get_device()));

            // insert the cuBLAS handle in the arg list and call the cuBLAS function
            return cuda_function(std::forward<Args>(args)..., stream_);
        }
        template <typename... Args>
        void execute(Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(target_.native_handle().get_device()));
            cuda_error(cudaLaunchKernel(std::forward<Args>(args)..., stream_));
        }

        template <typename... Args>
        void copy_async(Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(target_.native_handle().get_device()));

            // insert the stream in the arg list and call CUDA memcpy
            cuda_error(cudaMemcpyAsync(std::forward<Args>(args)..., stream_));
        }
        template <typename... Args>
        void memset_async(Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(target_.native_handle().get_device()));

            // insert the uda stream in the arg list and call CUDA memcpy
            cuda_error(cudaMemsetAsync(std::forward<Args>(args)..., stream_));
        }

        // get the future to synchronize this cuBLAS stream with
        future_type get_future()
        {
            return target_.get_future();
        }

        // return a reference to the compute::cuda object owned by this class
        hpx::compute::cuda::target& target()
        {
            return target_;
        }

        static void cuda_error(cudaError_t err)
        {
            if (err != cudaSuccess)
            {
                std::stringstream temp;
                temp << "CUDA function returned error code "
                     << cudaGetErrorString(err);
                throw std::runtime_error(temp.str());
            }
        }

        static void print_local_targets(void)
        {
            auto targets = hpx::compute::cuda::target::get_local_targets();
            for (auto target : targets)
            {
                std::cout << "GPU Device "
                          << target.native_handle().get_device() << ": \""
                          << target.native_handle().processor_name() << "\" "
                          << "with compute capability "
                          << target.native_handle().processor_family() << "\n";
            }
        }

    private:
        cudaStream_t stream_;
        hpx::compute::cuda::target target_;
    };
}}
#endif
