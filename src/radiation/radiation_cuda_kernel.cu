#if defined(OCTOTIGER_HAVE_CUDA)
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/radiation/radiation_cuda_kernel.hpp"

namespace octotiger { namespace radiation {
    __global__ void
    __launch_bounds__(1, 1)
    cuda_radiation_kernel()
    {}
}}
#endif    // OCTOTIGER_HAVE_CUDA)
