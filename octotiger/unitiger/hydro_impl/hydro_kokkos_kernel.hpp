//  Copyright (c) 2021-2023 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#pragma once
#include <hpx/config/compiler_specific.hpp>

#include <Kokkos_Core.hpp>
#include <aggregation_manager.hpp>
#include <hpx/futures/future.hpp>
#include <stream_manager.hpp>
#include <type_traits>
#include <utility>
#include "octotiger/unitiger/hydro.hpp"
#ifdef OCTOTIGER_HAVE_KOKKOS
#include <hpx/kokkos/executors.hpp>
#include "octotiger/common_kernel/kokkos_util.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"    // required for wrappers

#include "octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp"
#include "octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp"

#include "octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp"

#ifdef HPX_HAVE_APEX
#include <apex_api.hpp>
#endif

#include "octotiger/aggregation_util.hpp"

// padding for arrays (useful for masked operations). 
constexpr int padding = 128;
// number of tasks to be used for CPU execution per kernel invocation. 1 usually works best but 
// it depends on the available workload
constexpr int number_chunks = OCTOTIGER_KOKKOS_HYDRO_TASKS; 

template <typename storage>
const storage& get_flux_host_masks() {
    static storage masks(NDIM * q_inx3);
    static bool initialized = false;
    if (!initialized) {
        fill_masks(masks);
        initialized = true;
    }
    return masks;
}

template <typename storage, typename storage_host, typename executor_t>
const storage& get_flux_device_masks(executor_t& exec2, const size_t gpu_id = 0) {
    static std::vector<storage> masks;
    static bool initialized = false;
    /* if (agg_exec.parent.gpu_id == 1) */
    if (!initialized) {
        const storage_host& tmp_masks = get_flux_host_masks<storage_host>();
        for (int gpu_id_loop = 0; gpu_id_loop < opts().number_gpus; gpu_id_loop++) {
            stream_pool::select_device<executor_t,
                  round_robin_pool<executor_t>>(gpu_id_loop);
          executor_t exec{hpx::kokkos::execution_space_mode::independent};
          masks.emplace_back(gpu_id_loop, NDIM * q_inx3);
          Kokkos::deep_copy(exec.instance(), masks[gpu_id_loop], tmp_masks);
          exec.instance().fence();
        }
        initialized = true;
    }
    return masks[gpu_id];
}


/// Team-less version of the kokkos flux impl
/** Meant to be run on host, though it can be used on both host and device.
 * Does not use any team utility as those cause problems in the kokkos host executions spaces
 * (Kokkos serial) when using one execution space per kernel execution (not thread-safe it appears).
 * This is a stop-gap solution until teams work properly on host as well.
 */
template <typename simd_t, typename simd_mask_t, typename agg_executor_t,
         typename kokkos_buffer_t, typename kokkos_int_buffer_t,
         typename kokkos_mask_t>
void flux_impl_teamless(
    agg_executor_t &agg_exec,
    const kokkos_buffer_t& q_combined, const kokkos_buffer_t& x_combined,
    kokkos_buffer_t& f_combined,
    kokkos_buffer_t& amax, kokkos_int_buffer_t& amax_indices, kokkos_int_buffer_t& amax_d,
    const kokkos_mask_t& masks, const double omega, const kokkos_buffer_t& dx,
    const double A_, const double B_, const int nf, const double fgamma, const double de_switch_1,
    const int number_blocks, const int team_size
    ) {
    // Supported team_sizes need to be the power of two! Team size of 1 is a special case for usage
    // with the serial kokkos backend:
    assert((team_size == 1));
    if (agg_exec.sync_aggregation_slices()) {
        using kokkos_executor_t = typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type;
        using kokkos_backend_t = decltype(agg_exec.get_underlying_executor().instance());
        static_assert(std::is_same<hpx::kokkos::executor<kokkos_backend_t>,
                          typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type>::value,
            " backend mismatch!");
        const size_t gpu_id = agg_exec.parent.gpu_id;
        stream_pool::select_device<hpx::kokkos::executor<kokkos_backend_t>,
              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
        const int number_slices = agg_exec.number_slices;
        const int max_slices = opts().max_kernels_fused;
        auto policy = Kokkos::Experimental::require(
            Kokkos::RangePolicy<decltype(agg_exec.get_underlying_executor().instance())>(
                agg_exec.get_underlying_executor().instance(), 0, (number_blocks) * number_slices),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        // Incrase chunk size if it's an hpx executor to use 1-task optimization
        if constexpr (std::is_same_v<
                          typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type,
                          hpx::kokkos::hpx_executor>) {
            static_assert(number_chunks >= 1);
            static_assert(number_chunks <= 64);
            assert(number_blocks * number_slices / number_chunks >= 1); 
            policy.set_chunk_size(number_blocks * number_slices / number_chunks);
        }

        Kokkos::parallel_for(
            "kernel hydro flux teamless", policy, KOKKOS_LAMBDA(int idx) {
                // Index helpers:
                const int slice_idx = idx % (number_blocks);
                const int slice_id = idx / (number_blocks);
                const int blocks_per_dim = (number_blocks) / NDIM;
                const int dim = (slice_idx / blocks_per_dim);
                const int index = (slice_idx % blocks_per_dim) * team_size * simd_t::size();
                const int block_id = slice_idx;

                // Map to slice subviews
                auto [q_combined_slice, x_combined_slice, f_combined_slice] =
                      map_views_to_slice(slice_id, max_slices, q_combined, x_combined, f_combined);
                auto [amax_slice, amax_indices_slice, amax_d_slice] =
                      map_views_to_slice(slice_id, max_slices, amax, amax_indices, amax_d);

                // Set during cmake step with
                // -DOCTOTIGER_WITH_MAX_NUMBER_FIELDS
                // assumes maximal number (given by cmake) of species in a simulation.  Not the most
                // elegant solution and rather old-fashion but one that works.  May be changed to a
                // more flexible sophisticated object.
                std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS> local_f;
                std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS> local_q;
                std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS> local_q_flipped;
                for (int f = 0; f < nf; f++) {
                    local_f[f] = simd_t(0.0);
                }
                std::array<simd_t, NDIM> local_x; 
                std::array<simd_t, NDIM> local_vg; 
                for (int dim = 0; dim < NDIM; dim++) {
                  local_x[dim] = simd_t(0.0);
                  local_vg[dim] = simd_t(0.0);
                }
                // Reset (as we only add to the results...)
                for (int f = 0; f < nf; f++) {
                    for (int i = 0; i < simd_t::size(); i++) {
                      f_combined_slice[dim * nf * q_inx3 + f * q_inx3 + index + i] = 0.0;
                    }
                }
                amax_slice[block_id] = 0.0;
                amax_indices_slice[block_id] = 0;
                amax_d_slice[block_id] = 0;
                double current_amax = 0.0;
                int current_d = 0;
                int current_i = index;

                // Calculate the flux:
                if (index + simd_t::size() > q_inx * q_inx + q_inx && index < q_inx3) {
                    // Workaround to set the mask - usually I'd like to set it
                    // component-wise but kokkos-simd currently does not support this!
                    // hence the mask_helpers
                    const simd_t mask_helper1(1.0);
                    std::array<double, simd_t::size()> mask_helper2_array;
                    for (int i = 0; i < simd_t::size(); i++) {
                        mask_helper2_array[i] = masks[index + dim * dim_offset + i];
                    }
                    const simd_t mask_helper2(
                        mask_helper2_array.data(), SIMD_NAMESPACE::element_aligned_tag{});
                    const simd_mask_t mask = mask_helper1 == mask_helper2;
                    if (SIMD_NAMESPACE::any_of(mask)) {
                        simd_t this_ap = 0.0, this_am = 0.0;
                        for (int fi = 0; fi < 9; fi++) { // TODO replace 9 with the actual constexpr
                            const int d = faces[dim][fi];
                            const int flipped_dim = flip_dim(d, dim);
                            for (int dim = 0; dim < NDIM; dim++) {
                                local_x[dim] =
                                    simd_t(x_combined_slice.data() + dim * q_inx3 + index,
                                        SIMD_NAMESPACE::element_aligned_tag{}) +
                                    simd_t(0.5 * xloc[d][dim] * dx[slice_id]);
                            }
                            local_vg[0] = -omega *
                                (simd_t(x_combined_slice.data() + q_inx3 + index,
                                     SIMD_NAMESPACE::element_aligned_tag{}) +
                                    simd_t(0.5 * xloc[d][1] * dx[slice_id]));
                            local_vg[1] = +omega *
                                (simd_t(x_combined_slice.data() + index,
                                     SIMD_NAMESPACE::element_aligned_tag{}) +
                                    simd_t(0.5 * xloc[d][0] * dx[slice_id]));
                            local_vg[2] = simd_t(0.0);

                            for (int f = 0; f < nf; f++) {
                                local_q[f].copy_from(q_combined_slice.data() + f * face_offset +
                                        dim_offset * d + index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                                assert(index - compressedH_DN[dim] >= 0 &&
                                    index - compressedH_DN[dim] < q_inx3);
                                local_q_flipped[f].copy_from(q_combined_slice.data() +
                                        f * face_offset + dim_offset * flipped_dim -
                                        compressedH_DN[dim] + index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                            }
                            // Call the actual compute method
                            cell_inner_flux_loop_simd<simd_t>(omega, nf, A_, B_, local_q, local_q_flipped,
                                local_f, local_x, local_vg, this_ap, this_am, dim, d, dx[slice_id],
                                fgamma, de_switch_1,
                                face_offset);

                            // Update maximum values
                            this_ap = SIMD_NAMESPACE::choose(mask, this_ap, simd_t(0.0));
                            this_am = SIMD_NAMESPACE::choose(mask, this_am, simd_t(0.0));
                            // Add results to the final flux buffer
                            for (int f = 1; f < nf; f++) {
                              simd_t current_val(
                                  f_combined_slice.data() + dim * nf * q_inx3 + f * q_inx3 + index,
                                  SIMD_NAMESPACE::element_aligned_tag{});
                              current_val = current_val +
                                SIMD_NAMESPACE::choose(mask, quad_weights[fi] * local_f[f],
                                    simd_t(0.0));
                              current_val.copy_to(f_combined_slice.data() + dim
                                * nf * q_inx3 + f * q_inx3 + index,
                                SIMD_NAMESPACE::element_aligned_tag{});
                            }
                        }
                        const simd_t amax_tmp = SIMD_NAMESPACE::max(this_ap, (-this_am));
                        // Reduce
                        std::array<double, simd_t::size()> max_helper;
                        amax_tmp.copy_to(max_helper.data(), SIMD_NAMESPACE::element_aligned_tag{});
                        for (int i = 0; i < simd_t::size(); i++) {
                          if (max_helper[i] > current_amax) {
                              current_amax = max_helper[i];
                              current_d = faces[dim][8];
                              current_i = index + i;
                          }
                        }
                    }
                    simd_t current_val = 0.0;
                    for (int f = spc_i; f < nf; f++) {
                        simd_t current_field_val(
                            f_combined_slice.data() + dim * nf * q_inx3 + f * q_inx3 + index,
                            SIMD_NAMESPACE::element_aligned_tag{});
                        current_val = current_val +
                          SIMD_NAMESPACE::choose(mask, current_field_val,
                              simd_t(0.0));
                    }
                    current_val.copy_to(
                      f_combined_slice.data() + dim * nf * q_inx3 + index,
                      SIMD_NAMESPACE::element_aligned_tag{});
                }

                // Write Maximum of local team to amax:
                amax_slice[block_id] = current_amax;
                amax_indices_slice[block_id] = current_i;
                amax_d_slice[block_id] = current_d;
                // Save faces to the end of the amax buffer This avoids putting combined_q back on
                // the host side just to read those few values
                const int flipped_dim = flip_dim(amax_d_slice[block_id], dim);
                for (int f = 0; f < nf; f++) {
                    amax_slice[(number_blocks) + block_id * 2 * nf + f] =
                        q_combined_slice[amax_indices_slice[block_id] +
                            f * face_offset +
                            dim_offset * amax_d_slice[block_id]];
                    amax_slice[(number_blocks) + block_id * 2 * nf + nf + f] =
                        q_combined_slice[amax_indices_slice[block_id] -
                            compressedH_DN[dim] + f * face_offset + dim_offset * flipped_dim];
                }
            });
    }
}

template <typename agg_executor_t, typename kokkos_buffer_t, typename kokkos_int_buffer_t, typename kokkos_mask_t>
void flux_impl(
    agg_executor_t& agg_exec,
    const kokkos_buffer_t& q_combined, const kokkos_buffer_t& x_combined,
    kokkos_buffer_t& f_combined, kokkos_buffer_t& amax, kokkos_int_buffer_t& amax_indices,
    kokkos_int_buffer_t& amax_d, const kokkos_mask_t& masks, const double omega,
    const kokkos_buffer_t& dx, const double A_, const double B_, const int nf, const double fgamma,
    const double de_switch_1, const int number_blocks, const int team_size) {
    // Supported team_sizes need to be the power of two! Team size of 1 is a special case for usage
    // with the serial kokkos backend:
    assert((team_size == 256) || (team_size == 128) || (team_size == 64) || (team_size == 32) ||
        (team_size == 1));

    if (agg_exec.sync_aggregation_slices()) {
        using kokkos_executor_t = typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type;
        using kokkos_backend_t = decltype(agg_exec.get_underlying_executor().instance());
        static_assert(std::is_same<hpx::kokkos::executor<kokkos_backend_t>,
                          typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type>::value,
            " backend mismatch!");
        const size_t gpu_id = agg_exec.parent.gpu_id;
        stream_pool::select_device<hpx::kokkos::executor<kokkos_backend_t>,
              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
        const int number_slices = agg_exec.number_slices;
        const int max_slices = opts().max_kernels_fused;
        // Set policy via executor and allocate enough scratch memory:
        auto policy =
            Kokkos::Experimental::require(Kokkos::TeamPolicy<decltype(agg_exec.get_underlying_executor().instance())>(
                                              agg_exec.get_underlying_executor().instance(),
                                              number_blocks * number_slices, team_size),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        using policytype = decltype(policy);
        using membertype = typename decltype(policy)::member_type;
        if (team_size > 1)
            policy.set_scratch_size(
                0, Kokkos::PerTeam(team_size * (sizeof(double) + sizeof(int) *
                    2)));

        // TODO This method does not contain full simd support yet, mostly because neither the
        // HPX backend nor the Serial backend in Kokkos offer support for team sizes > 1
        // If they ever the we can replace the remaining doubles in the following parallel_for by
        // the proper simd_t types and subsequently turn these simd_t to avx or other instruction sets
        using simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
        using simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;

        // Start kernel using policy (and through it the passed executor):
        Kokkos::parallel_for(
            "kernel hydro flux", policy, KOKKOS_LAMBDA(const membertype& team_handle) {
                // Index helpers:
                const int teams_per_slice = number_blocks;
                const int slice_id = team_handle.league_rank() / teams_per_slice;
                const int unsliced_team_league = team_handle.league_rank() % teams_per_slice;
                const int blocks_per_dim = number_blocks / NDIM;
                const int dim = (unsliced_team_league / blocks_per_dim);
                const int tid = team_handle.team_rank();
                const int index = (unsliced_team_league % blocks_per_dim) * team_size + tid;
                const int block_id = unsliced_team_league;


                auto [q_combined_slice, x_combined_slice, f_combined_slice] =
                      map_views_to_slice(slice_id, max_slices, q_combined, x_combined, f_combined);
                auto [amax_slice, amax_indices_slice, amax_d_slice] =
                      map_views_to_slice(slice_id, max_slices, amax, amax_indices, amax_d);

                // Set during cmake step with
                // -DOCTOTIGER_WITH_MAX_NUMBER_FIELDS
                std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS> local_f;
                std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS> local_q;
                std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS> local_q_flipped;
                // assumes maximal number (given by cmake) of species in a simulation.  Not the most
                // elegant solution and rather old-fashion but one that works.  May be changed to a
                // more flexible sophisticated object.
                for (int f = 0; f < nf; f++) {
                    local_f[f] = 0.0;
                }
                std::array<simd_t, NDIM> local_x; 
                std::array<simd_t, NDIM> local_vg; 
                for (int dim = 0; dim < NDIM; dim++) {
                  local_x[dim] = simd_t(0.0);
                  local_vg[dim] = simd_t(0.0);
                }
                for (int f = 0; f < nf; f++) {
                    for (int i = 0; i < simd_t::size(); i++) {
                      f_combined_slice[dim * nf * q_inx3 + f * q_inx3 + index + i] = 0.0;
                    }
                }
                if (tid == 0) {
                    amax_slice[block_id] = 0.0;
                    amax_indices_slice[block_id] = 0;
                    amax_d_slice[block_id] = 0;
                }
                double current_amax = 0.0;
                int current_d = 0;
                int current_i = index;

                // Calculate the flux:
                if (index + simd_t::size() > q_inx * q_inx + q_inx && index < q_inx3) {
                    // Workaround to set the mask - usually I'd like to set it
                    // component-wise but kokkos-simd currently does not support this!
                    // hence the mask_helpers
                    const simd_t mask_helper1(1.0);
                    std::array<double, simd_t::size()> mask_helper2_array;
                    // TODO make masks double and load directly
                    for (int i = 0; i < simd_t::size(); i++) {
                        mask_helper2_array[i] = masks[index + dim * dim_offset + i];
                    }
                    const simd_t mask_helper2(
                        mask_helper2_array.data(), SIMD_NAMESPACE::element_aligned_tag{});
                    const simd_mask_t mask = mask_helper1 == mask_helper2;
                    if (SIMD_NAMESPACE::any_of(mask)) {
                        simd_t this_ap = 0.0, this_am = 0.0;
                        for (int fi = 0; fi < 9; fi++) { // TODO replace 9
                            const int d = faces[dim][fi];
                            const int flipped_dim = flip_dim(d, dim);
                            for (int dim = 0; dim < NDIM; dim++) {
                                local_x[dim] =
                                    simd_t(x_combined_slice.data() + dim * q_inx3 + index,
                                        SIMD_NAMESPACE::element_aligned_tag{}) +
                                    simd_t(0.5 * xloc[d][dim] * dx[slice_id]);
                            }
                            local_vg[0] = -omega *
                                (simd_t(x_combined_slice.data() + q_inx3 + index,
                                     SIMD_NAMESPACE::element_aligned_tag{}) +
                                    simd_t(0.5 * xloc[d][1] * dx[slice_id]));
                            local_vg[1] = +omega *
                                (simd_t(x_combined_slice.data() + index,
                                     SIMD_NAMESPACE::element_aligned_tag{}) +
                                    simd_t(0.5 * xloc[d][0] * dx[slice_id]));
                            local_vg[2] = simd_t(0.0);

                            for (int f = 0; f < nf; f++) {
                                local_q[f].copy_from(q_combined_slice.data() + f * face_offset +
                                        dim_offset * d + index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                                assert(index - compressedH_DN[dim] >= 0 &&
                                    index - compressedH_DN[dim] < q_inx3);
                                local_q_flipped[f].copy_from(q_combined_slice.data() +
                                        f * face_offset + dim_offset * flipped_dim -
                                        compressedH_DN[dim] + index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                            }
                            // Call the actual compute method
                            cell_inner_flux_loop_simd<simd_t>(omega, nf, A_, B_, local_q, local_q_flipped,
                                local_f, local_x, local_vg, this_ap, this_am, dim, d, dx[slice_id],
                                fgamma, de_switch_1,
                                face_offset);

                            // Update maximum values
                            this_ap = SIMD_NAMESPACE::choose(mask, this_ap, simd_t(0.0));
                            this_am = SIMD_NAMESPACE::choose(mask, this_am, simd_t(0.0));
                            // Add results to the final flux buffer
                            for (int f = 1; f < nf; f++) {
                                simd_t current_val(
                                    f_combined_slice.data() + dim * nf * q_inx3 + f * q_inx3 + index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                                current_val = current_val +
                                  SIMD_NAMESPACE::choose(mask, quad_weights[fi] * local_f[f],
                                      simd_t(0.0));
                                current_val.copy_to(f_combined_slice.data() + dim
                                  * nf * q_inx3 + f * q_inx3 + index,
                                  SIMD_NAMESPACE::element_aligned_tag{});
                            }
                        }
                        // Find max
                        const simd_t amax_tmp = SIMD_NAMESPACE::max(this_ap, (-this_am));
                        std::array<double, simd_t::size()> max_helper;
                        amax_tmp.copy_to(max_helper.data(), SIMD_NAMESPACE::element_aligned_tag{});
                        for (int i = 0; i < simd_t::size(); i++) {
                            if (max_helper[i] > current_amax) {
                                current_amax = max_helper[i];
                                current_d = faces[dim][8];
                                current_i = index + i;
                            }
                        }
                    }
                    simd_t current_val = 0.0;
                    for (int f = spc_i; f < nf; f++) {
                        simd_t current_field_val(
                            f_combined_slice.data() + dim * nf * q_inx3 + f * q_inx3 + index,
                            SIMD_NAMESPACE::element_aligned_tag{});
                        current_val = current_val +
                          SIMD_NAMESPACE::choose(mask, current_field_val,
                              simd_t(0.0));
                    }
                    current_val.copy_to(
                      f_combined_slice.data() + dim * nf * q_inx3 + index,
                      SIMD_NAMESPACE::element_aligned_tag{});
                }

                // Parallel maximum search within workgroup: Kokkos serial backend does not seem to
                // support concurrent (multiple serial executions spaces) Scratch memory accesses!
                // Hence the parallel maximum search is only done if the team size is larger than 1
                // (indicates serial backend)
                if (team_handle.team_size() > 1) {
                    Kokkos::View<double*,
                        typename policytype::execution_space::scratch_memory_space>
                        sm_amax(team_handle.team_scratch(0), team_size);
                    Kokkos::View<int*, typename policytype::execution_space::scratch_memory_space>
                        sm_i(team_handle.team_scratch(0), team_size);
                    Kokkos::View<int*, typename policytype::execution_space::scratch_memory_space>
                        sm_d(team_handle.team_scratch(0), team_size);
                    sm_amax[tid] = current_amax;
                    sm_d[tid] = current_d;
                    sm_i[tid] = current_i;
                    team_handle.team_barrier();
                    int tid_border = team_handle.team_size() / 2;
                    if (tid_border >= 32) {
                        // Max reduction with multiple warps
                        for (; tid_border >= 32; tid_border /= 2) {
                            if (tid < tid_border) {
                                if (sm_amax[tid + tid_border] > sm_amax[tid]) {
                                    sm_amax[tid] = sm_amax[tid + tid_border];
                                    sm_d[tid] = sm_d[tid + tid_border];
                                    sm_i[tid] = sm_i[tid + tid_border];
                                } else if (sm_amax[tid + tid_border] == sm_amax[tid]) {
                                    if (sm_i[tid + tid_border] < sm_i[tid]) {
                                        sm_amax[tid] = sm_amax[tid + tid_border];
                                        sm_d[tid] = sm_d[tid + tid_border];
                                        sm_i[tid] = sm_i[tid + tid_border];
                                    }
                                }
                            }
                            team_handle.team_barrier();
                        }
                    }
                    // Max reduction within one warp
                    for (; tid_border >= 1; tid_border /= 2) {
                        if (tid < tid_border) {
                            if (sm_amax[tid + tid_border] > sm_amax[tid]) {
                                sm_amax[tid] = sm_amax[tid + tid_border];
                                sm_d[tid] = sm_d[tid + tid_border];
                                sm_i[tid] = sm_i[tid + tid_border];
                            } else if (sm_amax[tid + tid_border] == sm_amax[tid]) {
                                if (sm_i[tid + tid_border] < sm_i[tid]) {
                                    sm_amax[tid] = sm_amax[tid + tid_border];
                                    sm_d[tid] = sm_d[tid + tid_border];
                                    sm_i[tid] = sm_i[tid + tid_border];
                                }
                            }
                        }
                    }
                    if (tid == 0) {
                        amax_slice[block_id] = sm_amax[0];
                        amax_indices_slice[block_id] = sm_i[0];
                        amax_d_slice[block_id] = sm_d[0];
                    }
                }

                // Write Maximum of local team to amax:
                if (tid == 0) {
                    // Kokkos serial backend does not seem to support concurrent (multiple serial
                    // executions spaces) Scratch memory accesses! Hence we cannot do the parralel
                    // sorting in scratch memory and use this work around for
                    // team sizes of 1 (as used by invocations on the serial
                    // backend)
                    if (team_handle.team_size() == 1) {
                        amax_slice[block_id] = current_amax;
                        amax_indices_slice[block_id] = index;
                        amax_d_slice[block_id] = current_d;
                    }
                    // Save face to the end of the amax buffer This avoids putting combined_q back
                    // on the host side just to read those few values
                    const int flipped_dim = flip_dim(amax_d_slice[block_id], dim);
                    for (int f = 0; f < nf; f++) {
                        amax_slice[number_blocks + block_id * 2 * nf + f] =
                            q_combined_slice[amax_indices_slice[block_id] +
                                f * face_offset +
                                dim_offset * amax_d_slice[block_id]];
                        amax_slice[number_blocks + block_id * 2 * nf + nf + f] =
                            q_combined_slice[amax_indices_slice[block_id] -
                                compressedH_DN[dim] + f * face_offset +
                                dim_offset * flipped_dim];
                    }
                }
            });
    }
}

/// Reconstruct with or without am
template <typename simd_t, typename simd_mask_t, typename agg_executor_t,
    typename kokkos_buffer_t, typename kokkos_int_buffer_t>
void reconstruct_impl(
    agg_executor_t& agg_exec,
    const double omega, const int nf_, const int angmom_index_,
    const kokkos_int_buffer_t& smooth_field_, const kokkos_int_buffer_t& disc_detect_,
    kokkos_buffer_t& combined_q, const kokkos_buffer_t& combined_x, kokkos_buffer_t& combined_u,
    kokkos_buffer_t& AM, const kokkos_buffer_t& dx, const kokkos_buffer_t& cdiscs,
    const int n_species_, const int ndir, const int nangmom,
    const size_t workgroup_size, const size_t team_size) {
    assert(team_size <= workgroup_size);
    const int blocks =
        (q_inx * q_inx * q_inx / simd_t::size() + (q_inx3 % simd_t::size() > 0 ? 1 : 0)) / workgroup_size + 1;
    
    if (agg_exec.sync_aggregation_slices()) {
        const size_t number_slices = agg_exec.number_slices;
        const size_t max_slices = opts().max_kernels_fused;
        using kokkos_executor_t = typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type;
        using kokkos_backend_t = decltype(agg_exec.get_underlying_executor().instance());
        static_assert(std::is_same<hpx::kokkos::executor<kokkos_backend_t>,
                          typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type>::value,
            " backend mismatch!");
        const size_t gpu_id = agg_exec.parent.gpu_id;
        stream_pool::select_device<hpx::kokkos::executor<kokkos_backend_t>,
              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
        auto policy = Kokkos::Experimental::require(
            Kokkos::TeamPolicy<decltype(agg_exec.get_underlying_executor().instance())>(
                agg_exec.get_underlying_executor().instance(), blocks * number_slices,
                team_size),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        using membertype = typename decltype(policy)::member_type;

        if constexpr (std::is_same_v<
                          typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type,
                          hpx::kokkos::hpx_executor>) {
            static_assert(number_chunks >= 1);
            static_assert(number_chunks <= 64);
            assert(blocks * number_slices / number_chunks >= 1); 
            policy.set_chunk_size(blocks * number_slices / number_chunks);
        }

        Kokkos::parallel_for(
            "kernel hydro reconstruct", policy, KOKKOS_LAMBDA(const membertype& team_handle) {
                const size_t slice_id = (team_handle.league_rank() / blocks);
                const int index_base = (team_handle.league_rank() % blocks) *
                workgroup_size;
                const size_t sx_i = angmom_index_;
                const size_t zx_i = sx_i + NDIM;
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_handle, workgroup_size),
                    [&](const int& work_elem) {
                        const int index = index_base + work_elem;
                        const int q_i = index * simd_t::size();

                        const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
                            (((q_i % q_inx2) / q_inx) + 2) * inx_large +
                            (((q_i % q_inx2) % q_inx) + 2);

                        auto [smooth_field_slice, disc_detect_slice, combined_q_slice, combined_x_slice,
                            combined_u_slice, AM_slice, dx_slice, cdiscs_slice] =
                            map_views_to_slice(slice_id, max_slices, smooth_field_, disc_detect_,
                                combined_q, combined_x, combined_u, AM, dx, cdiscs);

                        if (q_i < q_inx3) {
                            for (int n = 0; n < nangmom; n++) {
                                simd_t u_nangmom;
                                simd_t u;
                                // Load specialization for a as the simd value may stretch over 2
                                // bars/lines in the cube, thus the values to be loaded are not
                                // necessarily consecutive in memory
                                if (q_i % q_inx + simd_t::size() - 1 < q_inx) {
                                    // values are all in the first line/bar and
                                    // can simply be loaded
                                    u_nangmom.copy_from(combined_u_slice.data() +
                                            (zx_i + n) * u_face_offset + i,
                                        SIMD_NAMESPACE::element_aligned_tag{});
                                    u.copy_from(combined_u_slice.data() + i,
                                        SIMD_NAMESPACE::element_aligned_tag{});
                                } else {
                                    std::array<double, simd_t::size()> u_nangmom_helper;
                                    std::array<double, simd_t::size()> u_helper;
                                    size_t simd_i = 0;
                                    // load from first bar
                                    for (size_t i_line = q_i % q_inx; i_line < q_inx;
                                         i_line++, simd_i++) {
                                        u_nangmom_helper[simd_i] =
                                            combined_u_slice[(zx_i + n) * u_face_offset + i +
                                                simd_i];
                                        u_helper[simd_i] = combined_u_slice[i + simd_i];
                                    }
                                    // calculate indexing offset to check where the second line/bar
                                    // is starting
                                    size_t offset = (inx_large - q_inx);
                                    if constexpr (q_inx2 % simd_t::size() != 0) {
                                        if ((q_i + simd_i) % q_inx2 == 0) {
                                            offset += (inx_large - q_inx) * inx_large;
                                        }
                                    }
                                    // Load relevant values from second
                                    // line/bar
                                    for (; simd_i < simd_t::size(); simd_i++) {
                                        u_nangmom_helper[simd_i] =
                                            combined_u_slice[(zx_i + n) * u_face_offset + i +
                                                simd_i + offset];
                                        u_helper[simd_i] =
                                            combined_u_slice[i + simd_i + offset];
                                    }
                                    u_nangmom.copy_from(u_nangmom_helper.data(),
                                        SIMD_NAMESPACE::element_aligned_tag{});
                                    u.copy_from(u_helper.data(),
                                        SIMD_NAMESPACE::element_aligned_tag{});
                                }

                                const simd_t current_am = u_nangmom * u;
                                current_am.copy_to(
                                    AM_slice.data() + n * am_offset + q_i,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                            }
                            cell_reconstruct_inner_loop_p1_simd<simd_t, simd_mask_t>(nf_,
                                angmom_index_, smooth_field_slice.data(), disc_detect_slice.data(),
                                combined_q_slice.data(), combined_u_slice.data(), AM_slice.data(), dx_slice[0],
                                cdiscs_slice.data(), i, q_i, ndir, nangmom);
                        }
                    });
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_handle, workgroup_size),
                    [&](const int& work_elem) {
                        const int index = index_base + work_elem;
                        const int q_i = index * simd_t::size();

                        const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
                            (((q_i % q_inx2) / q_inx) + 2) * inx_large +
                            (((q_i % q_inx2) % q_inx) + 2);

                        auto [smooth_field_slice, disc_detect_slice, combined_q_slice, combined_x_slice,
                            combined_u_slice, AM_slice, dx_slice, cdiscs_slice] =
                            map_views_to_slice(slice_id, max_slices, smooth_field_, disc_detect_,
                                combined_q, combined_x, combined_u, AM, dx, cdiscs);

                        if (q_i < q_inx3) {
                            // Phase 2
                            for (int d = 0; d < ndir; d++) {
                                cell_reconstruct_inner_loop_p2_simd<simd_t, simd_mask_t>(omega,
                                    angmom_index_, combined_q_slice.data(), combined_x_slice.data(),
                                    combined_u_slice.data(), AM_slice.data(), dx_slice[0], d, i, q_i, ndir,
                                    nangmom, n_species_, nf_);
                            }
                        }
                    });
            });
    }
}

/// Reconstruct with or without am
template <typename simd_t, typename simd_mask_t, typename agg_executor_t,
    typename kokkos_buffer_t, typename kokkos_int_buffer_t>
void reconstruct_teamless_impl(
    agg_executor_t& agg_exec,
    const double omega, const int nf_, const int angmom_index_,
    const kokkos_int_buffer_t& smooth_field_, const kokkos_int_buffer_t& disc_detect_,
    kokkos_buffer_t& combined_q, const kokkos_buffer_t& combined_x, kokkos_buffer_t& combined_u,
    kokkos_buffer_t& AM, const kokkos_buffer_t& dx, const kokkos_buffer_t& cdiscs,
    const int n_species_, const int ndir, const int nangmom,
    const Kokkos::Array<long, 4>&& tiling_config) {
    const int blocks =
        (q_inx * q_inx * q_inx / simd_t::size()) / 64 + 1;
    
    if (agg_exec.sync_aggregation_slices()) {
        const int number_slices = agg_exec.number_slices;
        const int max_slices = opts().max_kernels_fused;
        using kokkos_executor_t = typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type;
        using kokkos_backend_t = decltype(agg_exec.get_underlying_executor().instance());
        static_assert(std::is_same<hpx::kokkos::executor<kokkos_backend_t>,
                          typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type>::value,
            " backend mismatch!");
        const size_t gpu_id = agg_exec.parent.gpu_id;
        stream_pool::select_device<hpx::kokkos::executor<kokkos_backend_t>,
              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
        auto policy = Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<decltype(agg_exec.get_underlying_executor().instance()),
                Kokkos::Rank<4>>(agg_exec.get_underlying_executor().instance(), {0, 0, 0, 0},
                {number_slices, blocks, 8, 8}, tiling_config),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        Kokkos::parallel_for(
            "kernel hydro reconstruct teamless", policy,
            KOKKOS_LAMBDA(int slice_id, int idx, int idy, int idz) {
                const int index = (idx) * 64 + (idy) * 8 + (idz);
                const int q_i = index * simd_t::size();
                const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
                    (((q_i % q_inx2) / q_inx) + 2) * inx_large + (((q_i % q_inx2) % q_inx) + 2);
                    
                const int sx_i = angmom_index_;
                const int zx_i = sx_i + NDIM;

                const int u_slice_offset = (nf_ * H_N3 + padding) * slice_id;
                const int am_slice_offset = (NDIM * q_inx3 + padding) * slice_id;
                if (q_i < q_inx3) {
                    auto [smooth_field_slice, disc_detect_slice, combined_q_slice, combined_x_slice,
                        combined_u_slice, AM_slice, dx_slice, cdiscs_slice] =
                        map_views_to_slice(slice_id, max_slices, smooth_field_, disc_detect_,
                            combined_q, combined_x, combined_u, AM, dx, cdiscs);

                    for (int n = 0; n < nangmom; n++) {
                        simd_t u_nangmom;
                        simd_t u;
                        // Load specialization for a as the simd value may stretch over 2
                        // bars/lines in the cube, thus the values to be loaded are not
                        // necessarily consecutive in memory
                        if (q_i % q_inx + simd_t::size() - 1 < q_inx) {
                            // values are all in the first line/bar and
                            // can simply be loaded
                            u_nangmom.copy_from(combined_u.data() +
                                    (zx_i + n) * u_face_offset + i + u_slice_offset,
                                SIMD_NAMESPACE::element_aligned_tag{});
                            u.copy_from(combined_u.data() + i + u_slice_offset,
                                SIMD_NAMESPACE::element_aligned_tag{});
                        } else {
                            std::array<double, simd_t::size()> u_nangmom_helper;
                            std::array<double, simd_t::size()> u_helper;
                            size_t simd_i = 0;
                            // load from first bar
                            for (size_t i_line = q_i % q_inx; i_line < q_inx;
                                 i_line++, simd_i++) {
                                u_nangmom_helper[simd_i] =
                                    combined_u[(zx_i + n) * u_face_offset + i +
                                        u_slice_offset + simd_i];
                                u_helper[simd_i] = combined_u[i + u_slice_offset + simd_i];
                            }
                            // calculate indexing offset to check where the second line/bar
                            // is starting
                            size_t offset = (inx_large - q_inx);
                            if constexpr (q_inx2 % simd_t::size() != 0) {
                                if ((q_i + simd_i) % q_inx2 == 0) {
                                    offset += (inx_large - q_inx) * inx_large;
                                }
                            }
                            // Load relevant values from second
                            // line/bar
                            for (; simd_i < simd_t::size(); simd_i++) {
                                u_nangmom_helper[simd_i] =
                                    combined_u[(zx_i + n) * u_face_offset + i +
                                        u_slice_offset + simd_i + offset];
                                u_helper[simd_i] =
                                    combined_u[i + u_slice_offset + simd_i + offset];
                            }
                            u_nangmom.copy_from(u_nangmom_helper.data(),
                                SIMD_NAMESPACE::element_aligned_tag{});
                            u.copy_from(u_helper.data(),
                                SIMD_NAMESPACE::element_aligned_tag{});
                        }

                        const simd_t current_am = u_nangmom * u;
                        current_am.copy_to(
                            AM.data() + n * am_offset + q_i + am_slice_offset,
                            SIMD_NAMESPACE::element_aligned_tag{});
                    }
                    cell_reconstruct_inner_loop_p1_simd<simd_t, simd_mask_t>(nf_, angmom_index_,
                        smooth_field_slice.data(), disc_detect_slice.data(),
                        combined_q_slice.data(), combined_u_slice.data(), AM_slice.data(),
                        dx_slice[0], cdiscs_slice.data(), i, q_i, ndir,
                        nangmom);
                    // Phase 2
                    for (int d = 0; d < ndir; d++) {
                        cell_reconstruct_inner_loop_p2_simd<simd_t, simd_mask_t>(omega,
                            angmom_index_, combined_q_slice.data(), combined_x_slice.data(),
                            combined_u_slice.data(), AM_slice.data(), dx_slice[0], d, i, q_i, ndir,
                            nangmom, n_species_, nf_);
                          }
                      }
            });
    }
}

/// Optimized for reconstruct without am correction
template <typename simd_t, typename simd_mask_t, typename agg_executor_t,
    typename kokkos_buffer_t, typename kokkos_int_buffer_t>
void reconstruct_no_amc_impl(
    agg_executor_t& agg_exec,
    const double omega, const int nf_, const int angmom_index_,
    const kokkos_int_buffer_t& smooth_field_, const kokkos_int_buffer_t& disc_detect_,
    kokkos_buffer_t& combined_q, const kokkos_buffer_t& combined_x, kokkos_buffer_t& combined_u,
    kokkos_buffer_t& AM, const kokkos_buffer_t& dx, const kokkos_buffer_t& cdiscs,
    const int n_species_, const int ndir, const int nangmom,
    const size_t workgroup_size, const size_t team_size) {
    assert(team_size <= workgroup_size);
    const int blocks =
        (q_inx * q_inx * q_inx / simd_t::size() + (q_inx3 % simd_t::size() > 0 ? 1 : 0)) / workgroup_size + 1;
    if (agg_exec.sync_aggregation_slices()) {
        const size_t number_slices = agg_exec.number_slices;
        const size_t max_slices = opts().max_kernels_fused;
        using kokkos_executor_t = typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type;
        using kokkos_backend_t = decltype(agg_exec.get_underlying_executor().instance());
        static_assert(std::is_same<hpx::kokkos::executor<kokkos_backend_t>,
                          typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type>::value,
            " backend mismatch!");
        const size_t gpu_id = agg_exec.parent.gpu_id;
        stream_pool::select_device<hpx::kokkos::executor<kokkos_backend_t>,
              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
        auto policy = Kokkos::Experimental::require(
            Kokkos::TeamPolicy<decltype(agg_exec.get_underlying_executor().instance())>(
                agg_exec.get_underlying_executor().instance(), blocks * number_slices, team_size),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        using membertype = typename decltype(policy)::member_type;

        if constexpr (std::is_same_v<
                          typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type,
                          hpx::kokkos::hpx_executor>) {
            static_assert(number_chunks >= 1);
            static_assert(number_chunks <= 64);
            assert(blocks * number_slices / number_chunks >= 1); 
            policy.set_chunk_size(blocks * number_slices / number_chunks);
        }
        Kokkos::parallel_for(
            "kernel hydro reconstruct_no_amc", policy,
            KOKKOS_LAMBDA(const membertype& team_handle) {
                const size_t slice_id = (team_handle.league_rank() / blocks);
                const size_t index_base = (team_handle.league_rank() % blocks) *
                workgroup_size;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_handle, workgroup_size),
                    [&](const int& work_elem) {
                        const int index = index_base + work_elem;
                        const int q_i = index * simd_t::size();

                        const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
                            (((q_i % q_inx2) / q_inx) + 2) * inx_large +
                            (((q_i % q_inx2) % q_inx) + 2);

                        auto [smooth_field_slice, disc_detect_slice, combined_q_slice, combined_x_slice,
                            combined_u_slice, AM_slice, dx_slice, cdiscs_slice] =
                            map_views_to_slice(slice_id, max_slices, smooth_field_, disc_detect_,
                                combined_q, combined_x, combined_u, AM, dx, cdiscs);

                        if (q_i < q_inx3) {
                        // TODO use new subviews and remove index mapping from reconstruct...
                                cell_reconstruct_inner_loop_p1_simd<simd_t, simd_mask_t>(nf_,
                                    angmom_index_, smooth_field_slice.data(), disc_detect_slice.data(),
                                    combined_q_slice.data(), combined_u_slice.data(), AM_slice.data(), dx_slice[0],
                                    cdiscs_slice.data(), i, q_i, ndir, nangmom);
                        }
                    });
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_handle, workgroup_size),
                    [&](const int& work_elem) {
                        const int index = index_base + work_elem;
                        const int q_i = index * simd_t::size();

                        const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
                            (((q_i % q_inx2) / q_inx) + 2) * inx_large +
                            (((q_i % q_inx2) % q_inx) + 2);

                        auto [smooth_field_slice, disc_detect_slice, combined_q_slice, combined_x_slice,
                            combined_u_slice, AM_slice, dx_slice, cdiscs_slice] =
                            map_views_to_slice(slice_id, max_slices, smooth_field_, disc_detect_,
                                combined_q, combined_x, combined_u, AM, dx, cdiscs);
                        if (q_i < q_inx3) {
                            // Phase 2
                            for (int d = 0; d < ndir; d++) { // 27 (but with one less due to != in kerenl)
                                cell_reconstruct_inner_loop_p2_simd<simd_t, simd_mask_t>(omega, // 26 * 72
                                    angmom_index_, combined_q_slice.data(), combined_x_slice.data(),
                                    combined_u_slice.data(), AM_slice.data(), dx_slice[0], d, i, q_i, ndir,
                                    nangmom, n_species_, nf_);
                            }
                        }
                    });
            });
    }
}

/// Optimized for reconstruct without am correction
template <typename simd_t, typename simd_mask_t, typename agg_executor_t,
    typename kokkos_buffer_t, typename kokkos_int_buffer_t>
void reconstruct_no_amc_teamless_impl(
    agg_executor_t& agg_exec,
    const double omega, const int nf_, const int angmom_index_,
    const kokkos_int_buffer_t& smooth_field_, const kokkos_int_buffer_t& disc_detect_,
    kokkos_buffer_t& combined_q, const kokkos_buffer_t& combined_x, kokkos_buffer_t& combined_u,
    kokkos_buffer_t& AM, const kokkos_buffer_t& dx, const kokkos_buffer_t& cdiscs,
    const int n_species_, const int ndir, const int nangmom,
    const Kokkos::Array<long, 4>&& tiling_config) {
    const int blocks =
        (q_inx * q_inx * q_inx / simd_t::size()) / 64 + 1;
    if (agg_exec.sync_aggregation_slices()) {
        const int number_slices = agg_exec.number_slices;
        const int max_slices = opts().max_kernels_fused;
        using kokkos_executor_t = typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type;
        using kokkos_backend_t = decltype(agg_exec.get_underlying_executor().instance());
        static_assert(std::is_same<hpx::kokkos::executor<kokkos_backend_t>,
                          typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type>::value,
            " backend mismatch!");
        const size_t gpu_id = agg_exec.parent.gpu_id;
        stream_pool::select_device<hpx::kokkos::executor<kokkos_backend_t>,
              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
        auto policy = Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<decltype(agg_exec.get_underlying_executor().instance()),
                Kokkos::Rank<4>>(agg_exec.get_underlying_executor().instance(), {0, 0, 0, 0},
                {number_slices, blocks, 8, 8}, tiling_config),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        Kokkos::parallel_for(
            "kernel hydro reconstruct_no_amc teamless", policy,
            KOKKOS_LAMBDA(int slice_id, int idx, int idy, int idz) {
                const int index = (idx) * 64 + (idy) * 8 + (idz);
                const int q_i = index * simd_t::size();

                const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
                    (((q_i % q_inx2) / q_inx) + 2) * inx_large + (((q_i % q_inx2) % q_inx) + 2);
                if (q_i < q_inx3) {
                    auto [smooth_field_slice, disc_detect_slice, combined_q_slice, combined_x_slice,
                        combined_u_slice, AM_slice, dx_slice, cdiscs_slice] =
                        map_views_to_slice(slice_id, max_slices, smooth_field_, disc_detect_,
                            combined_q, combined_x, combined_u, AM, dx, cdiscs);

                    cell_reconstruct_inner_loop_p1_simd<simd_t, simd_mask_t>(nf_, angmom_index_,
                        smooth_field_slice.data(), disc_detect_slice.data(),
                        combined_q_slice.data(), combined_u_slice.data(), AM_slice.data(),
                        dx_slice[0], cdiscs_slice.data(), i, q_i, ndir,
                        nangmom);
                    // Phase 2
                    for (int d = 0; d < ndir; d++) {
                        cell_reconstruct_inner_loop_p2_simd<simd_t, simd_mask_t>(omega,
                            angmom_index_, combined_q_slice.data(), combined_x_slice.data(),
                            combined_u_slice.data(), AM_slice.data(),
                            dx_slice[0], d, i, q_i, ndir,
                            nangmom, n_species_, nf_);
                    }
                }
            });
    }
}


template <typename agg_executor_t, typename kokkos_buffer_t>
void hydro_pre_recon_impl(
    agg_executor_t &agg_exec,
    const kokkos_buffer_t& large_x, const double omega, const bool angmom, kokkos_buffer_t& u,
    const int nf, const int n_species, const Kokkos::Array<long, 4>&& tiling_config) {
    const int number_slices = agg_exec.number_slices;

    if (agg_exec.sync_aggregation_slices()) {
        using kokkos_executor_t = typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type;
        using kokkos_backend_t = decltype(agg_exec.get_underlying_executor().instance());
        static_assert(std::is_same<hpx::kokkos::executor<kokkos_backend_t>,
                          typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type>::value,
            " backend mismatch!");
        const size_t gpu_id = agg_exec.parent.gpu_id;
        stream_pool::select_device<hpx::kokkos::executor<kokkos_backend_t>,
              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
      const int blocks = (inx_large * inx_large * inx_large) / 64 + 1;
      const int max_slices = opts().max_kernels_fused;
      auto policy = Kokkos::Experimental::require(
          Kokkos::MDRangePolicy<decltype(agg_exec.get_underlying_executor().instance()),
              Kokkos::Rank<4>>(agg_exec.get_underlying_executor().instance(),
                {0, 0, 0, 0},
              {number_slices, blocks, 8, 8}, tiling_config),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight);
      Kokkos::parallel_for(
            "kernel hydro pre recon", policy, KOKKOS_LAMBDA(int slice_id, int idx, int idy, int idz) {
              const int index = (idx) * 64 + (idy) * 8 + (idz);
              if (index < inx_large * inx_large * inx_large) {
                const int grid_x = index / (inx_large * inx_large);
                const int grid_y = (index % (inx_large * inx_large)) / inx_large;
                const int grid_z = (index % (inx_large * inx_large)) % inx_large;
                auto [large_x_slice, u_slice] =
                      map_views_to_slice(slice_id, max_slices, large_x, u);
                cell_hydro_pre_recon(
                    large_x_slice, omega, angmom, u_slice, nf, n_species, grid_x, grid_y, grid_z);
              }
        });
    }
}

template <typename agg_executor_t, typename kokkos_buffer_t>
void find_contact_discs_impl(
    agg_executor_t &agg_exec,
    const kokkos_buffer_t& u, kokkos_buffer_t& P, kokkos_buffer_t& disc, const double A_,
    const double B_, const double fgamma_, const double de_switch_1, const size_t ndir, const size_t nf,
    const Kokkos::Array<long, 4>&& tiling_config_phase1,
    const Kokkos::Array<long, 4>&& tiling_config_phase2) {
    const int number_slices = agg_exec.number_slices;
    if (agg_exec.sync_aggregation_slices()) {
        using kokkos_executor_t = typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type;
        using kokkos_backend_t = decltype(agg_exec.get_underlying_executor().instance());
        static_assert(std::is_same<hpx::kokkos::executor<kokkos_backend_t>,
                          typename std::remove_reference<decltype(agg_exec.get_underlying_executor())>::type>::value,
            " backend mismatch!");
        const size_t gpu_id = agg_exec.parent.gpu_id;
        stream_pool::select_device<hpx::kokkos::executor<kokkos_backend_t>,
              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
        const int blocks = (inx_normal * inx_normal * inx_normal) / 64 + 1;
        const int max_slices = opts().max_kernels_fused;
        auto policy_phase_1 = Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<decltype(agg_exec.get_underlying_executor().instance()),
                Kokkos::Rank<4>>(agg_exec.get_underlying_executor().instance(), {0, 0, 0, 0},
                {number_slices, blocks, 8, 8}, tiling_config_phase1),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        Kokkos::parallel_for(
            "kernel find contact discs 1", policy_phase_1,
            KOKKOS_LAMBDA(int slice_id, int idx, int idy, int idz) {
                const int index = (idx) * 64 + (idy) * 8 + (idz);
                if (index < inx_normal * inx_normal * inx_normal) {
                  const int grid_x = index / (inx_normal * inx_normal);
                  const int grid_y = (index % (inx_normal * inx_normal)) / inx_normal;
                  const int grid_z = (index % (inx_normal * inx_normal)) % inx_normal;
                  auto [P_slice, u_slice] =
                        map_views_to_slice(slice_id, max_slices, P, u);
                  cell_find_contact_discs_phase1(
                      P_slice, u_slice, A_, B_, fgamma_, de_switch_1, nf,
                      grid_x, grid_y, grid_z);
                }
            });

        const int blocks2 = (q_inx * q_inx * q_inx) / 64 + 1;
        auto policy_phase_2 = Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<decltype(agg_exec.get_underlying_executor().instance()),
                Kokkos::Rank<4>>(agg_exec.get_underlying_executor().instance(),
                  {0, 0, 0, 0},
                {number_slices, blocks2, 8, 8}, tiling_config_phase2),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        Kokkos::parallel_for(
            "kernel find contact discs 2", policy_phase_2,
            KOKKOS_LAMBDA(int slice_id, int idx, int idy, int idz) {
                const int index = (idx) * 64 + (idy) * 8 + (idz);
                if (index < q_inx3) {
                  const int grid_x = index / (q_inx * q_inx);
                  const int grid_y = (index % (q_inx * q_inx)) / q_inx;
                  const int grid_z = (index % (q_inx * q_inx)) % q_inx;
                  auto [P_slice, disc_slice] =
                        map_views_to_slice(slice_id, max_slices, P, disc);
                  cell_find_contact_discs_phase2(disc_slice, P_slice, fgamma_, ndir,
                      grid_x, grid_y, grid_z);
                }
            });
    }
}

template <typename executor_t,
    std::enable_if_t<is_kokkos_device_executor<executor_t>::value, int> = 0>
timestep_t device_interface_kokkos_hydro(
    const aggregated_host_buffer<double, executor_t>& combined_x,
    const aggregated_host_buffer<double, executor_t>& combined_large_x,
    aggregated_host_buffer<double, executor_t>& combined_u,
    const aggregated_host_buffer<int, executor_t>& disc_detect,
    const aggregated_host_buffer<int, executor_t>& smooth_field,
    aggregated_host_buffer<double, executor_t>& host_f, const size_t ndir, const size_t nf,
    const bool angmom, const size_t n_species, const double omega, const int angmom_index,
    const int nangmom, const aggregated_host_buffer<double, executor_t>& dx, const double A_,
    const double B_, const double fgamma,
    const double de_switch_1, typename
    Aggregated_Executor<executor_t>::Executor_Slice& agg_exec,
    Allocator_Slice<double, kokkos_host_allocator<double>, executor_t>& alloc_host_double,
    Allocator_Slice<int, kokkos_host_allocator<int>, executor_t>& alloc_host_int) {
    // How many executor slices are working together and what's our ID?
    const size_t slice_id = agg_exec.id;
    const size_t number_slices = agg_exec.number_slices;
    const size_t max_slices = opts().max_kernels_fused;

    // Required device allocators for aggregated kernels
    auto alloc_device_int =
        agg_exec
            .template make_allocator<int, kokkos_device_allocator<int>>();
    auto alloc_device_double =
        agg_exec
            .template make_allocator<double, kokkos_device_allocator<double>>();

    // Find contact discs
    aggregated_device_buffer<double, executor_t> u(
        alloc_device_double, (nf * H_N3 + padding) * max_slices);
    aggregated_deep_copy(agg_exec, u, combined_u, (nf * H_N3 + padding));
    aggregated_device_buffer<double, executor_t> P(
        alloc_device_double, (H_N3 + padding) * max_slices);
    aggregated_device_buffer<double, executor_t> disc(
        alloc_device_double, (ndir / 2 * H_N3 + padding) * max_slices);
    find_contact_discs_impl(agg_exec, u, P, disc, physics<NDIM>::A_, physics<NDIM>::B_,
        physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1, ndir, nf, {1, 1, 8,
        8}, {1, 1, 8, 8});

    // Pre recon
    aggregated_device_buffer<double, executor_t> large_x(
        alloc_device_double, (NDIM * H_N3 + padding) * max_slices);
    aggregated_deep_copy(agg_exec, large_x, combined_large_x, (NDIM * H_N3 + padding));
    hydro_pre_recon_impl(agg_exec, large_x, omega, angmom, u, nf, n_species, {1, 1,
        8, 8});

    // Reconstruct
    aggregated_device_buffer<double, executor_t> x(
        alloc_device_double, (NDIM * q_inx3 + padding) * max_slices);
    aggregated_deep_copy(agg_exec, x, combined_x, (NDIM * q_inx3 + padding));
    aggregated_device_buffer<int, executor_t>
      device_disc_detect(alloc_device_int, nf * max_slices);
    aggregated_deep_copy(agg_exec, device_disc_detect, disc_detect);
    aggregated_device_buffer<int, executor_t> device_smooth_field(
        alloc_device_int, nf * max_slices);
    aggregated_deep_copy(agg_exec, device_smooth_field, smooth_field);
    aggregated_device_buffer<double, executor_t> q(
        alloc_device_double, (nf * 27 * q_inx3 + padding) * max_slices);
    aggregated_device_buffer<double, executor_t> AM(
        alloc_device_double, (NDIM * q_inx3 + padding) * max_slices);

    aggregated_device_buffer<double, executor_t> dx_device(alloc_device_double, max_slices);
    aggregated_deep_copy(agg_exec, dx_device, dx);

    if (angmom_index > -1) {
        reconstruct_impl<device_simd_t, device_simd_mask_t>(agg_exec, omega, nf,
            angmom_index, device_smooth_field, device_disc_detect, q, x, u, AM, dx_device,
            disc, n_species, ndir, nangmom, 64, 64);
    } else {
        reconstruct_no_amc_impl<device_simd_t, device_simd_mask_t>(agg_exec, omega, nf,
            angmom_index, device_smooth_field, device_disc_detect, q, x, u, AM, dx_device, disc,
            n_species, ndir, nangmom, 64, 64);
    }

    // Flux
    const device_buffer<bool>& masks =
        get_flux_device_masks<device_buffer<bool>, host_buffer<bool>,
      executor_t>(
            agg_exec.get_underlying_executor(), agg_exec.parent.gpu_id);
    const int number_blocks_small = (q_inx3 / 128 + 1) * 1;

    aggregated_device_buffer<double, executor_t> amax(
        alloc_device_double, number_blocks_small * NDIM * (1 + 2 * nf) * max_slices);
    aggregated_device_buffer<int, executor_t> amax_indices(
        alloc_device_int, number_blocks_small * NDIM * max_slices);
    aggregated_device_buffer<int, executor_t> amax_d(
        alloc_device_int, number_blocks_small * NDIM * max_slices);

    aggregated_device_buffer<double, executor_t> f(
        alloc_device_double, (NDIM * nf * q_inx3 + padding) * max_slices);
    flux_impl(agg_exec, q, x, f, amax, amax_indices, amax_d, masks, omega, dx_device, A_, B_,
        nf, fgamma, de_switch_1, NDIM * number_blocks_small, 128);
    aggregated_host_buffer<double, executor_t> host_amax(
        alloc_host_double, number_blocks_small * NDIM * (1 + 2 * nf) * max_slices);
    aggregated_host_buffer<int, executor_t> host_amax_indices(
        alloc_host_int, number_blocks_small * NDIM * max_slices);
    aggregated_host_buffer<int, executor_t> host_amax_d(
        alloc_host_int, number_blocks_small * NDIM * max_slices);

    aggregated_deep_copy(agg_exec, host_amax, amax);
    aggregated_deep_copy(agg_exec, host_amax_indices, amax_indices);
    /* aggregated_deep_copy(agg_exec, host_amax_d, amax_d); */

    auto fut = aggregrated_deep_copy_async<executor_t>(
        agg_exec, host_f, f, (NDIM * nf * q_inx3 + padding));
    fut.get();

    auto [host_amax_slice, host_amax_indices_slice, combined_x_slice] =
          map_views_to_slice(agg_exec, host_amax, host_amax_indices, combined_x);
    octotiger::hydro::hydro_kokkos_gpu_subgrids_processed++;
    if (slice_id == 0)
        octotiger::hydro::hydro_kokkos_gpu_aggregated_subgrids_launches++;

    // Find Maximum
    size_t current_max_slot = 0;
    for (size_t dim_i = 1; dim_i < number_blocks_small * NDIM; dim_i++) {
        if (host_amax_slice[dim_i] >
            host_amax_slice[current_max_slot]) {
            current_max_slot = dim_i;
        } else if (host_amax_slice[dim_i] ==
            host_amax_slice[current_max_slot]) {
            if (host_amax_indices_slice[dim_i] <
                host_amax_indices_slice[current_max_slot])
                current_max_slot = dim_i;
        }
    }

    // Create & Return timestep_t type
    std::vector<double> URs(nf), ULs(nf);
    const size_t current_max_index = host_amax_indices_slice[current_max_slot];
    /* const size_t current_d = host_amax_d[current_max_slot]; */
    timestep_t ts;
    ts.a = host_amax_slice[current_max_slot];
    ts.x = combined_x_slice[current_max_index];
    ts.y = combined_x_slice[current_max_index + q_inx3];
    ts.z = combined_x_slice[current_max_index + 2 * q_inx3];
    const size_t current_i = current_max_slot;
    const size_t current_dim = current_max_slot / number_blocks_small;
    /* const auto flipped_dim = flip_dim(current_d, current_dim); */
    constexpr int compressedH_DN[3] = {q_inx2, q_inx, 1};
    for (int f = 0; f < nf; f++) {
        URs[f] = host_amax_slice[NDIM * number_blocks_small + current_i * 2 * nf + f];
        ULs[f] = host_amax_slice[NDIM * number_blocks_small + current_i * 2 * nf + nf + f];
    }
    ts.ul = std::move(URs);
    ts.ur = std::move(ULs);
    ts.dim = current_dim;
    return ts;
}

template <typename executor_t,
    std::enable_if_t<is_kokkos_host_executor<executor_t>::value, int> = 0>
timestep_t device_interface_kokkos_hydro(
    const aggregated_host_buffer<double, executor_t>& combined_x,
    const aggregated_host_buffer<double, executor_t>& combined_large_x,
    aggregated_host_buffer<double, executor_t>& combined_u,
    const aggregated_host_buffer<int, executor_t>& disc_detect,
    const aggregated_host_buffer<int, executor_t>& smooth_field,
    aggregated_host_buffer<double, executor_t>& f, const size_t ndir, const size_t nf,
    const bool angmom, const size_t n_species, const double omega, const int angmom_index,
    const int nangmom, const aggregated_host_buffer<double, executor_t>& dx, const double A_,
    const double B_, const double fgamma, const double de_switch_1,
    typename Aggregated_Executor<executor_t>::Executor_Slice& agg_exec,
    Allocator_Slice<double, kokkos_host_allocator<double>, executor_t>& alloc_host_double,
    Allocator_Slice<int, kokkos_host_allocator<int>, executor_t>& alloc_host_int) {
    // How many executor slices are working together and what's our ID?
    const size_t slice_id = agg_exec.id;
    const size_t number_slices = agg_exec.number_slices;
    const size_t max_slices = opts().max_kernels_fused;

    // Slice offsets
    const int u_slice_offset = nf * H_N3 + padding;
    constexpr int x_slice_offset = NDIM * q_inx3 + padding;
    const int disc_detect_slice_offset = nf;
    const int smooth_slice_offset = nf;
    constexpr int large_x_slice_offset = (H_N3 * NDIM + padding); 
    //const int q_slice_offset = (nf_ * 27 * H_N3 + padding) 
    const int f_slice_offset = (NDIM* nf *  q_inx3 + padding);
    const int disc_offset = ndir / 2 * H_N3 + padding;

    // Find contact discs
    aggregated_host_buffer<double, executor_t> P(alloc_host_double, (H_N3 + padding) * max_slices);
    aggregated_host_buffer<double, executor_t> disc(
        alloc_host_double, (ndir / 2 * H_N3 + padding) * max_slices);
    find_contact_discs_impl(agg_exec, combined_u, P, disc, physics<NDIM>::A_,
        physics<NDIM>::B_, physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1, ndir, nf,
        {1, 32, 8, 8}, {1, 32, 8, 8});

    // Pre recon
    hydro_pre_recon_impl(agg_exec, combined_large_x, omega, angmom, combined_u, nf, n_species,
        {1, 64, 8, 8});

    // Reconstruct
    aggregated_host_buffer<double, executor_t> q(
        alloc_host_double, (nf * 27 * q_inx3 + padding) * max_slices);
    aggregated_host_buffer<double, executor_t> q2(
        alloc_host_double, (nf * 27 * q_inx3 + 2*padding) * max_slices);
    aggregated_host_buffer<double, executor_t> AM(
        alloc_host_double, (NDIM * q_inx3 + padding) * max_slices);

#ifdef HPX_HAVE_APEX
    auto reconstruct_timer = apex::start("kernel hydro_reconstruct kokkos");
#endif
#ifdef OCTOTIGER_HYDRO_HOST_HPX_EXECUTOR
#pragma message "Using team impl for reconstruct"
    if (angmom_index > -1) {
        reconstruct_impl<host_simd_t, host_simd_mask_t>(agg_exec, omega, nf, angmom_index,
            smooth_field, disc_detect, q, combined_x, combined_u, AM, dx, disc, n_species, ndir,
            nangmom, 8, 1);
    } else {
        reconstruct_no_amc_impl<host_simd_t, host_simd_mask_t>(agg_exec, omega, nf,
            angmom_index, smooth_field, disc_detect, q, combined_x, combined_u, AM, dx, disc,
            n_species, ndir, nangmom, 8, 1);
    }
#else
    if (angmom_index > -1) {
        reconstruct_teamless_impl<host_simd_t, host_simd_mask_t>(agg_exec, omega, nf, angmom_index,
            smooth_field, disc_detect, q, combined_x, combined_u, AM, dx, disc, n_species, ndir,
            nangmom, {1, 1, 8, 8});
    } else {
        reconstruct_no_amc_teamless_impl<host_simd_t, host_simd_mask_t>(agg_exec, omega, nf,
            angmom_index, smooth_field, disc_detect, q, combined_x, combined_u, AM, dx, disc,
            n_species, ndir, nangmom, {1, 1, 8, 8});
    }
#endif
#ifdef HPX_HAVE_APEX
    apex::stop(reconstruct_timer);
#endif

    // Flux
    static_assert(q_inx3 % host_simd_t::size() == 0,
        "q_inx3 size is not evenly divisible by simd size! This simd width would require some "
        "further changes to the masking in the flux kernel!");
    static_assert(q_inx > host_simd_t::size(),
        "SIMD width is larger than one subgrid lane + 2. "
        "This does not work given the current implementation");
    const int blocks = NDIM * (q_inx3 / host_simd_t::size()) * 1;
    const host_buffer<bool>& masks = get_flux_host_masks<host_buffer<bool>>();

    aggregated_host_buffer<double, executor_t> amax(
        alloc_host_double, blocks * (1 + 2 * nf) * max_slices);
    aggregated_host_buffer<int, executor_t> amax_indices(alloc_host_int, blocks * max_slices);
    aggregated_host_buffer<int, executor_t> amax_d(alloc_host_int, blocks * max_slices);

#ifdef HPX_HAVE_APEX
    auto flux_timer = apex::start("kernel hydro_flux kokkos");
#endif
    flux_impl_teamless<host_simd_t, host_simd_mask_t>(agg_exec, q, combined_x, f,
        amax, amax_indices, amax_d, masks, omega, dx, A_, B_, nf, fgamma,
        de_switch_1, blocks, 1);
#ifdef HPX_HAVE_APEX
    apex::stop(flux_timer);
#endif
    sync_kokkos_host_kernel(agg_exec.get_underlying_executor());

    octotiger::hydro::hydro_kokkos_cpu_subgrids_processed++;
    if (slice_id == 0)
        octotiger::hydro::hydro_kokkos_cpu_aggregated_subgrids_launches++;


    const int amax_slice_offset = (1 + 2 * nf) * blocks * slice_id;
    const int max_indices_slice_offset = blocks * slice_id;

    // Find Maximum
    size_t current_max_slot = 0;
    for (size_t dim_i = 1; dim_i < blocks; dim_i++) {
        if (amax[dim_i + amax_slice_offset] > amax[current_max_slot + amax_slice_offset]) {
            current_max_slot = dim_i;
        } else if (amax[dim_i + amax_slice_offset] == amax[current_max_slot + amax_slice_offset]) {
            if (amax_indices[dim_i + max_indices_slice_offset] <
                amax_indices[current_max_slot + max_indices_slice_offset])
                current_max_slot = dim_i;
        }
    }

    // Create & Return timestep_t type
    std::vector<double> URs(nf), ULs(nf);
    const size_t current_max_index = amax_indices[current_max_slot + max_indices_slice_offset];
    /* const size_t current_d = amax_d[current_max_slot]; */
    const auto current_dim = current_max_slot / (blocks / NDIM);
    timestep_t ts;
    ts.a = amax[current_max_slot + amax_slice_offset];
    ts.x = combined_x[current_max_index + x_slice_offset * slice_id];
    ts.y = combined_x[current_max_index + q_inx3 + x_slice_offset * slice_id];
    ts.z = combined_x[current_max_index + 2 * q_inx3 + x_slice_offset * slice_id];
    const size_t current_i = current_max_slot;
    /* const auto flipped_dim = flip_dim(current_d, current_dim); */
    constexpr int compressedH_DN[3] = {q_inx2, q_inx, 1};
    for (int f = 0; f < nf; f++) {
        URs[f] = amax[blocks + current_i * 2 * nf + f + amax_slice_offset];
        ULs[f] = amax[blocks + current_i * 2 * nf + nf + f + amax_slice_offset];
    }
    ts.ul = std::move(URs);
    ts.ur = std::move(ULs);
    ts.dim = current_dim;
    return ts;
}

// Input U, X, omega, executor, device_id
// Output F
template <typename executor_t>
timestep_t launch_hydro_kokkos_kernels(const hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const std::vector<std::vector<safe_real>>& U, const std::vector<std::vector<safe_real>>& X,
    const double omega, const size_t n_species, 
    std::vector<hydro_state_t<std::vector<safe_real>>>& F) {
    static const cell_geometry<NDIM, INX> geo;

    // Some assumptions must be true for the kernel to work
    // Should always be true for 3D production scenarios...
    assert(NDIM == 3);
    assert(geo.NFACEDIR == 9);
    assert(geo.NDIR == 27);
    assert(hydro.get_nf() == opts().n_fields);
    assert(hydro.get_nf() == spc_i + opts().n_species);

    auto executor_slice_fut = hydro_kokkos_agg_executor_pool<executor_t>::request_executor_slice();
    auto ret_fut = executor_slice_fut.value().then(hpx::annotated_function([&](auto && fut) {
      typename Aggregated_Executor<executor_t>::Executor_Slice agg_exec = fut.get();
      // How many executor slices are working together and what's our ID?
      const size_t slice_id = agg_exec.id;
      const size_t number_slices = agg_exec.number_slices;
      const size_t max_slices = opts().max_kernels_fused;

      // Get allocators of all the executors working together
      // Allocator_Slice<double, kokkos_host_allocator<double>, executor_t>
      Allocator_Slice<double, kokkos_host_allocator<double>, executor_t> alloc_host_double =
          agg_exec
              .template make_allocator<double, kokkos_host_allocator<double>>();
      Allocator_Slice<int, kokkos_host_allocator<int>, executor_t> alloc_host_int =
          agg_exec
              .template make_allocator<int, kokkos_host_allocator<int>>();
      timestep_t max_lambda{};
      {

      // Host buffers
      aggregated_host_buffer<double, executor_t> combined_x(
          alloc_host_double, (NDIM * q_inx3 + padding) * max_slices);
      aggregated_host_buffer<double, executor_t> combined_large_x(
          alloc_host_double, (NDIM * H_N3 + padding) * max_slices);
      aggregated_host_buffer<double, executor_t> combined_u(
          alloc_host_double, (hydro.get_nf() * H_N3 + padding) *
          max_slices);
      aggregated_host_buffer<double, executor_t> combined_f(
          alloc_host_double, (NDIM * hydro.get_nf() * q_inx3 + padding) * max_slices);

      // Map aggregated host buffers to the current slice
      auto [combined_x_slice, combined_large_x_slice, combined_u_slice, combined_f_slice] =
          map_views_to_slice(agg_exec, combined_x, combined_large_x, combined_u, combined_f);

      // Convert input into current slice buffers
      convert_x_structure(X, combined_x_slice.data());
      for (int n = 0; n < NDIM; n++) {
          std::copy(X[n].begin(), X[n].end(),
              combined_large_x_slice.data() + n * H_N3);
      }
      for (int f = 0; f < hydro.get_nf(); f++) {
          std::copy(
              U[f].begin(), U[f].end(), combined_u_slice.data() + f * H_N3);
      }

      // Helper host buffers
      aggregated_host_buffer<int, executor_t> disc_detect(
          alloc_host_int, (hydro.get_nf()) * max_slices);
      aggregated_host_buffer<int, executor_t> smooth_field(
          alloc_host_int, (hydro.get_nf()) * max_slices);
      aggregated_host_buffer<double, executor_t> dx(
          alloc_host_double, max_slices);

      // Map aggregated host buffers to the current slice
      auto [disc_detect_slice, smooth_field_slice, dx_slice] =
          map_views_to_slice(agg_exec, disc_detect, smooth_field, dx);
      const auto& disc_detect_bool = hydro.get_disc_detect();
      const auto& smooth_bool = hydro.get_smooth_field();
      for (auto f = 0; f < hydro.get_nf(); f++) {
          disc_detect_slice[f] = disc_detect_bool[f];
          smooth_field_slice[f] = smooth_bool[f];
      }
      dx_slice[0] = X[0][geo.H_DNX] - X[0][0];

      // Either handles the launches on the CPU or on the GPU depending on the passed executor
      max_lambda = device_interface_kokkos_hydro(combined_x, combined_large_x,
          combined_u, disc_detect, smooth_field, combined_f, geo.NDIR, hydro.get_nf(),
          hydro.get_angmom_index() != -1, n_species, omega, hydro.get_angmom_index(), geo.NANGMOM,
          dx, physics<NDIM>::A_, physics<NDIM>::B_, physics<NDIM>::fgamma_,
          physics<NDIM>::de_switch_1, agg_exec, alloc_host_double, alloc_host_int);

      // Convert output and store in f result slice
      for (int dim = 0; dim < NDIM; dim++) {
          for (integer field = 0; field != opts().n_fields; ++field) {
              const auto dim_offset = dim * opts().n_fields * q_inx3 + field * q_inx3;
              for (integer i = 0; i <= INX; ++i) {
                  for (integer j = 0; j <= INX; ++j) {
                      for (integer k = 0; k <= INX; ++k) {
                          const auto i0 = findex(i, j, k);
                          const auto input_index =
                              (i + 1) * q_inx * q_inx + (j + 1) * q_inx + (k + 1);
                          F[dim][field][i0] = combined_f_slice[dim_offset + input_index];
                      }
                  }
              }
          }
      }
      }
      return max_lambda;
          }, "kokkos_hydro_solver"));
    return ret_fut.get();
}
#endif
