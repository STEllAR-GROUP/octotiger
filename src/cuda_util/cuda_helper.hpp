#ifdef OCTOTIGER_CUDA_ENABLED
#define BOOST_NO_CXX11_ALLOCATOR
//
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_copy.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
//
#include <hpx/include/compute.hpp>
#include <hpx/compute/cuda/target.hpp>
#ifdef HPX_CUBLAS_DEMO_WITH_ALLOCATOR
# include <hpx/compute/cuda/allocator.hpp>
#endif
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
//
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

struct cuda_helper
{
public:
    using future_type    = hpx::future<void>;

    // construct a cublas stream
    cuda_helper(std::size_t device=0) : target_(device) {
        stream_ = target_.native_handle().get_stream();
    }

    ~cuda_helper() {
    }

    // This is a simple wrapper for any cublas call, pass in the same arguments
    // that you would use for a cublas call except the cublas handle which is omitted
    // as the wrapper will supply that for you
    template <typename Func, typename ...Args>
    void operator()(Func && cuda_function, Args&&... args)
    {
        // make sue we run on the correct device
        cuda_error(cudaSetDevice(target_.native_handle().get_device()));

        // insert the cublas handle in the arg list and call the cublas function
        cuda_error(cuda_function(std::forward<Args>(args)..., stream_));
    }

    template <typename ...Args>
    void copy_async(Args&&... args)
    {
        // make sue we run on the correct device
        cuda_error(cudaSetDevice(target_.native_handle().get_device()));

        // insert the uda stream in the arg list and call the cuda memcpy
        cuda_error(cudaMemcpyAsync (std::forward<Args>(args)..., stream_));
    }

    // get the future to synchronize this cublas stream with
    future_type get_future() { return target_.get_future(); }

    // return a reference to the compute::cuda object owned by this class
    hpx::compute::cuda::target & target() { return target_; }


    static void cuda_error(cudaError_t err) {
        if (err != cudaSuccess) {
            std::stringstream temp;
            temp << "cuda function returned error code " << cudaGetErrorString(err);
            throw std::runtime_error(temp.str());
        }
    }

private:
    cudaStream_t               stream_;
    hpx::compute::cuda::target target_;
};
#endif
