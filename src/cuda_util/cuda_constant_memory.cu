#ifdef OCTOTIGER_CUDA_ENABLED
#include <sstream>
#include "cuda_constant_memory.hpp"
namespace octotiger {
namespace fmm {
    // These global, __constant__ arrays will be created in the constant memory of each gpu
    // that is used by octotiger.
    __constant__ octotiger::fmm::multiindex<> device_stencil_const[STENCIL_SIZE];
    __constant__ double device_stencil_indicator_const[STENCIL_SIZE];
    __constant__ double device_four_constants[STENCIL_SIZE * 4];
    void copy_indicator_to_constant_memory(const double *indicator, const size_t indicator_size) {
        cudaError_t err = cudaMemcpyToSymbol(device_stencil_indicator_const, indicator, indicator_size);
        if (err != cudaSuccess) {
            std::stringstream temp;
            temp << "Copy stencil indicator to constant memory returned error code " << cudaGetErrorString(err);
            throw std::runtime_error(temp.str());
        }
    }
    void copy_stencil_to_constant_memory(const multiindex<> *stencil, const size_t stencil_size) {
        std::cout << "putting stencil onto gpu" << std::endl;
        std::cin.get();
        for(int i = 0; i < 1074; i++) {
            std::cout << stencil[i] << std::endl;
        }
        std::cin.get();
        cudaError_t err = cudaMemcpyToSymbol(device_stencil_const, stencil, stencil_size);
        if (err != cudaSuccess) {
            std::stringstream temp;
            temp << "Copy stencil to constant memory returned error code " << cudaGetErrorString(err);
            throw std::runtime_error(temp.str());
        }
    }
    void copy_constants_to_constant_memory(const double *constants, const size_t constants_size) {
        // std::cout << "putting fours onto gpu" << std::endl;
        // std::cin.get();
        // for(int i = 0; i < 1043; i++) {
        //     std::cout << constants[i*4 + 0] << " " << constants[i*4 + 1] << " " << constants[i*4 + 2] << "  "
        //               << constants[i*4 + 3] << std::endl;
        // }
        // std::cin.get();
        cudaError_t err = cudaMemcpyToSymbol(device_four_constants, constants, constants_size);
        if (err != cudaSuccess) {
            std::stringstream temp;
            temp << "Copy four-constants to constant memory returned error code " << cudaGetErrorString(err);
            throw std::runtime_error(temp.str());
        }
    }
}    // namespace fmm
}    // namespace octotiger
#endif
