//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
#include "octotiger/cuda_util/cuda_scheduler.hpp"
#include "octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp"
#include "octotiger/multipole_interactions/util/calculate_stencil.hpp"

#include "octotiger/options.hpp"
#include "octotiger/real.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#undef interface

namespace octotiger {
namespace fmm {
#ifdef OCTOTIGER_HAVE_CUDA
    void cuda_error(cudaError_t err) {
        if (err != cudaSuccess) {
            std::stringstream temp;
            temp << "CUDA function returned error code " << cudaGetErrorString(err);
            throw std::runtime_error(temp.str());
        }
    }
#endif
    kernel_scheduler& kernel_scheduler::scheduler() {
        static thread_local kernel_scheduler scheduler_;
        return scheduler_;
    }
    kernel_scheduler::kernel_scheduler() {}
    void kernel_scheduler::init_constants() {
        std::size_t gpu_count = opts().cuda_number_gpus;
        // Create necessary data and add padding
        two_phase_stencil const stencil = multipole_interactions::calculate_stencil();
        auto p2p_stencil_pair = monopole_interactions::calculate_stencil();

        // new p2p stencil stuff
        auto p2p_stencil_mask_pair =
            monopole_interactions::calculate_stencil_masks(p2p_stencil_pair.first);
        auto p2p_stencil_mask = p2p_stencil_mask_pair.first;
        auto p2p_four_constants = p2p_stencil_mask_pair.second;
        std::unique_ptr<bool[]> stencil_masks = std::make_unique<bool[]>(FULL_STENCIL_SIZE);
        std::unique_ptr<real[]> four_constants_tmp =
            std::make_unique<real[]>(4 * FULL_STENCIL_SIZE);
        for (auto i = 0; i < FULL_STENCIL_SIZE; ++i) {
            four_constants_tmp[i * 4 + 0] = p2p_four_constants[i][0];
            four_constants_tmp[i * 4 + 1] = p2p_four_constants[i][1];
            four_constants_tmp[i * 4 + 2] = p2p_four_constants[i][2];
            four_constants_tmp[i * 4 + 3] = p2p_four_constants[i][3];
            if (p2p_stencil_mask[i]) {
                stencil_masks[i] = true;
            } else {
                stencil_masks[i] = false;
            }
        }

        // new multipole stencil
        auto multipole_stencil_pair = multipole_interactions::calculate_stencil_masks(stencil);
        auto multipole_stencil = multipole_stencil_pair.first;
        auto multipole_inner_stencil = multipole_stencil_pair.second;
        std::unique_ptr<bool[]> multipole_stencil_masks =
            std::make_unique<bool[]>(FULL_STENCIL_SIZE);
        std::unique_ptr<bool[]> multipole_inner_stencil_masks =
            std::make_unique<bool[]>(FULL_STENCIL_SIZE);
        for (auto i = 0; i < FULL_STENCIL_SIZE; ++i) {
            if (multipole_inner_stencil[i]) {
                multipole_inner_stencil_masks[i] = true;
            } else {
                multipole_inner_stencil_masks[i] = false;
            }
            if (multipole_stencil[i]) {
                multipole_stencil_masks[i] = true;
            } else {
                multipole_stencil_masks[i] = false;
            }
        }

        // Move data to constant memory, once per gpu
        for (std::size_t gpu_id = 0; gpu_id < gpu_count; ++gpu_id) {
            /*cuda_error(cudaSetDevice(gpu_id));
            cuda_error(cudaMemcpyToSymbol(multipole_interactions::device_constant_stencil_masks,
                multipole_stencil_masks.get(), full_stencil_size / sizeof(double) * sizeof(bool)));
            cuda_error(cudaMemcpyToSymbol(multipole_interactions::device_stencil_indicator_const,
                multipole_inner_stencil_masks.get(),
                full_stencil_size / sizeof(double) * sizeof(bool)));*/
            monopole_interactions::init_stencil(
                gpu_id, std::move(stencil_masks), std::move(four_constants_tmp));
            multipole_interactions::init_stencil(gpu_id, std::move(multipole_stencil_masks),
                std::move(multipole_inner_stencil_masks));
            /*cuda_error(cudaMemcpyToSymbol(
                monopole_interactions::device_stencil_masks, multipole_stencil_masks.get(),
                full_stencil_size / sizeof(double) * sizeof(bool)));
            cuda_error(cudaMemcpyToSymbol(
            monopole_interactions::device_four_constants,
            four_constants_tmp.get(), full_stencil_size * 4));*/
        }
    }
    kernel_scheduler::~kernel_scheduler() {}

}    // namespace fmm
}    // namespace octotiger
#endif
