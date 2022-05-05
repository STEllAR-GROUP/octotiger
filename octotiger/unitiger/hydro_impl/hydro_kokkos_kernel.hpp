#pragma once

#include <Kokkos_View.hpp>
#include <aggregation_manager.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/kokkos/executors.hpp>
#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/common_kernel/kokkos_util.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"    // required for wrappers

#include "octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp"
#include "octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp"

static const char hydro_kokkos_kernel_identifier[] = "hydro_kernel_aggregator_kokkos";
template<typename executor_t>
using hydro_kokkos_agg_executor_pool = aggregation_pool<hydro_kokkos_kernel_identifier, executor_t,
                                       pool_strategy>;

constexpr int padding = 128;

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
const storage& get_flux_device_masks(executor_t& exec) {
    static storage masks(NDIM * q_inx3);
    static bool initialized = false;
    if (!initialized) {
        const storage_host& tmp_masks = get_flux_host_masks<storage_host>();
        Kokkos::deep_copy(exec.instance(), masks, tmp_masks);
        exec.instance().fence();
        initialized = true;
    }
    return masks;
}

template<typename Agg_executor_t, typename TargetView_t, typename SourceView_t>
void aggregated_deep_copy(
    Agg_executor_t &agg_exec,
    TargetView_t &target,
    SourceView_t &source) {
    if (agg_exec.sync_aggregation_slices()) {
        Kokkos::deep_copy(agg_exec.get_underlying_executor().instance(), target, source);
    }
}

template<typename executor_t, typename TargetView_t, typename SourceView_t>
hpx::lcos::shared_future<void> aggregrated_deep_copy_async(
    typename Aggregated_Executor<executor_t>::Executor_Slice &agg_exec,
    TargetView_t &target, SourceView_t &source) {
    auto launch_copy_lambda = [](TargetView_t &target, SourceView_t &source, executor_t &exec) -> hpx::lcos::shared_future<void> {
      return hpx::kokkos::deep_copy_async(exec.instance(), target, source);
    };
    return agg_exec.wrap_async(launch_copy_lambda, target, source, agg_exec.get_underlying_executor());
}

/// Team-less version of the kokkos flux impl
/** Meant to be run on host, though it can be used on both host and device.
 * Does not use any team utility as those cause problems in the kokkos host executions spaces
 * (Kokkos serial) when using one execution space per kernel execution (not thread-safe it appears).
 * This is a stop-gap solution until teams work properly on host as well.
 */
template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_int_buffer_t,
    typename kokkos_mask_t>
void flux_impl_teamless(hpx::kokkos::executor<kokkos_backend_t>& executor,
    typename Aggregated_Executor<hpx::kokkos::executor<kokkos_backend_t>>::Executor_Slice &agg_exec,
    const kokkos_buffer_t& q_combined, const kokkos_buffer_t& x_combined,
    kokkos_buffer_t& f_combined, kokkos_buffer_t& amax, kokkos_int_buffer_t& amax_indices,
    kokkos_int_buffer_t& amax_d, const kokkos_mask_t& masks, const double omega, const kokkos_buffer_t& dx,
    const double A_, const double B_, const int nf, const double fgamma, const double de_switch_1,
    const int number_blocks, const int team_size) {
    // Supported team_sizes need to be the power of two! Team size of 1 is a special case for usage
    // with the serial kokkos backend:
    assert((team_size == 1));
    if (agg_exec.sync_aggregation_slices()) {
        const int number_slices = agg_exec.number_slices;
        assert(number_slices == 1); // not implemented yet
        auto policy = Kokkos::Experimental::require(
            Kokkos::RangePolicy<decltype(executor.instance())>(
                agg_exec.get_underlying_executor().instance(), 0, number_blocks),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);

        Kokkos::parallel_for(
            "kernel hydro flux", policy, KOKKOS_LAMBDA(int idx) {
                // Index helpers:
                const int blocks_per_dim = number_blocks / NDIM;
                const int dim = (idx / blocks_per_dim);
                const int index = (idx % blocks_per_dim) * team_size;
                const int block_id = idx;
                const int slice_id = 0;

                // Default values for relevant buffers/variables:

                // Set during cmake step with
                // -DOCTOTIGER_WITH_MAX_NUMBER_FIELDS
                double local_f[OCTOTIGER_MAX_NUMBER_FIELDS];
                // assumes maximal number (given by cmake) of species in a simulation.  Not the most
                // elegant solution and rather old-fashion but one that works.  May be changed to a
                // more flexible sophisticated object.
                for (int f = 0; f < nf; f++) {
                    local_f[f] = 0.0;
                }
                double local_x[3] = {0.0, 0.0, 0.0};
                double local_vg[3] = {0.0, 0.0, 0.0};
                for (int f = 0; f < nf; f++) {
                    f_combined[dim * nf * q_inx3 + f * q_inx3 + index] = 0.0;
                }
                amax[block_id] = 0.0;
                amax_indices[block_id] = 0;
                amax_d[block_id] = 0;
                double current_amax = 0.0;
                int current_d = 0;

                // Calculate the flux:
                if (index > q_inx * q_inx + q_inx && index < q_inx3) {
                    double mask = masks[index + dim * dim_offset];
                    if (mask != 0.0) {
                        for (int fi = 0; fi < 9; fi++) {
                            double this_ap = 0.0, this_am = 0.0;
                            const int d = faces[dim][fi];
                            const int flipped_dim = flip_dim(d, dim);
                            for (int dim = 0; dim < 3; dim++) {
                                local_x[dim] =
                                    x_combined[dim * q_inx3 + index] + (0.5 * xloc[d][dim] * dx[slice_id]);
                            }
                            local_vg[0] = -omega * (x_combined[q_inx3 + index] + 0.5 * xloc[d][1] * dx[slice_id]);
                            local_vg[1] = +omega * (x_combined[index] + 0.5 * xloc[d][0] * dx[slice_id]);
                            local_vg[2] = 0.0;
                            // Call the actual compute method
                            cell_inner_flux_loop<double>(omega, nf, A_, B_, q_combined, local_f,
                                local_x, local_vg, this_ap, this_am, dim, d, dx[slice_id], fgamma,
                                de_switch_1, dim_offset * d + index,
                                dim_offset * flipped_dim - compressedH_DN[dim] + index,
                                face_offset);
                            // TODO Preparation for later SIMD masking (not
                            // supported yet)
                            this_ap *= mask;
                            this_am *= mask;
                            // Update maximum values
                            const double amax_tmp = max_wrapper(this_ap, (-this_am));
                            if (amax_tmp > current_amax) {
                                current_amax = amax_tmp;
                                current_d = d;
                            }
                            // Add results to the final flux buffer
                            for (int f = 1; f < nf; f++) {
                                f_combined[dim * nf * q_inx3 + f * q_inx3 + index] +=
                                    quad_weights[fi] * local_f[f];
                            }
                        }
                    }
                    for (int f = 10; f < nf; f++) {
                        f_combined[dim * nf * q_inx3 + index] +=
                            f_combined[dim * nf * q_inx3 + f * q_inx3 + index];
                    }
                }

                // Write Maximum of local team to amax:
                amax[block_id] = current_amax;
                amax_indices[block_id] = index;
                amax_d[block_id] = current_d;
                // Save face to the end of the amax buffer This avoids putting combined_q back on
                // the host side just to read those few values
                const int flipped_dim = flip_dim(amax_d[block_id], dim);
                for (int f = 0; f < nf; f++) {
                    amax[number_blocks + block_id * 2 * nf + f] =
                        q_combined[amax_indices[block_id] + f * face_offset +
                            dim_offset * amax_d[block_id]];
                    amax[number_blocks + block_id * 2 * nf + nf + f] =
                        q_combined[amax_indices[block_id] - compressedH_DN[dim] + f * face_offset +
                            dim_offset * flipped_dim];
                }
            });
    }
}

template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_int_buffer_t,
    typename kokkos_mask_t>
void flux_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
    typename Aggregated_Executor<hpx::kokkos::executor<kokkos_backend_t>>::Executor_Slice &agg_exec,
    const kokkos_buffer_t& q_combined,
    const kokkos_buffer_t& x_combined, kokkos_buffer_t& f_combined, kokkos_buffer_t& amax,
    kokkos_int_buffer_t& amax_indices, kokkos_int_buffer_t& amax_d, const kokkos_mask_t& masks,
    const double omega, const kokkos_buffer_t& dx, const double A_, const double B_, const int nf,
    const double fgamma, const double de_switch_1, const int number_blocks, const int team_size) {
    // Supported team_sizes need to be the power of two! Team size of 1 is a special case for usage
    // with the serial kokkos backend:
    assert((team_size == 256) || (team_size == 128) || (team_size == 64) || (team_size == 32) ||
        (team_size == 1));

    if (agg_exec.sync_aggregation_slices()) {
        const int number_slices = agg_exec.number_slices;
        // Set policy via executor and allocate enough scratch memory:
        using policytype = Kokkos::TeamPolicy<decltype(executor.instance())>;
        auto policy =
            policytype(agg_exec.get_underlying_executor().instance(), number_blocks * number_slices, team_size);
        using membertype = typename policytype::member_type;
        if (team_size > 1)
            policy.set_scratch_size(
                0, Kokkos::PerTeam(team_size * (sizeof(double) + sizeof(int) *
                    2)));


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

                // todo insert reconstruct index
                const int q_slice_offset = (nf * 27 * q_inx3 + padding) * slice_id;
                const int f_slice_offset = (NDIM * nf * q_inx3 + padding) * slice_id;
                const int x_slice_offset = (NDIM * q_inx3 + padding) * slice_id;
                const int amax_slice_offset = (1 + 2 * nf) * number_blocks * slice_id;
                const int max_indices_slice_offset = number_blocks * slice_id;
                // Default values for relevant buffers/variables:
                auto q_combined_slice = Kokkos::subview(
                    q_combined, std::make_pair(q_slice_offset, (nf * 27 * q_inx3 + padding) *
                    (slice_id + 1)));
                auto x_combined_slice = Kokkos::subview(
                    x_combined, std::make_pair(x_slice_offset, (NDIM * q_inx3 + padding) *
                    (slice_id + 1)));
                auto f_combined_slice = Kokkos::subview(
                    f_combined, std::make_pair(f_slice_offset, (NDIM * nf * q_inx3 + padding) *
                    (slice_id + 1)));

                // Set during cmake step with
                // -DOCTOTIGER_WITH_MAX_NUMBER_FIELDS
                double local_f[OCTOTIGER_MAX_NUMBER_FIELDS];
                // assumes maximal number (given by cmake) of species in a simulation.  Not the most
                // elegant solution and rather old-fashion but one that works.  May be changed to a
                // more flexible sophisticated object.
                for (int f = 0; f < nf; f++) {
                    local_f[f] = 0.0;
                }
                double local_x[3] = {0.0, 0.0, 0.0};
                double local_vg[3] = {0.0, 0.0, 0.0};
                for (int f = 0; f < nf; f++) {
                    f_combined_slice[dim * nf * q_inx3 + f * q_inx3 + index] = 0.0;
                }
                if (tid == 0) {
                    amax[block_id + amax_slice_offset] = 0.0;
                    amax_indices[block_id + max_indices_slice_offset] = 0;
                    amax_d[block_id + max_indices_slice_offset] = 0;
                }
                double current_amax = 0.0;
                int current_d = 0;

                // Calculate the flux:
                if (index > q_inx * q_inx + q_inx && index < q_inx3) {
                    const double mask = masks[index + dim * dim_offset];
                    if (mask != 0.0) {
                        for (int fi = 0; fi < 9; fi++) {            
                            double this_ap = 0.0, this_am = 0.0;   
                            const int d = faces[dim][fi];
                            const int flipped_dim = flip_dim(d, dim);
                            for (int dim = 0; dim < 3; dim++) {
                                local_x[dim] =
                                    x_combined_slice[dim * q_inx3 + index] + (0.5 * xloc[d][dim] * dx[slice_id]);
                            }
                            local_vg[0] = -omega * (x_combined_slice[q_inx3 + index] + 0.5 * xloc[d][1] * dx[slice_id]);
                            local_vg[1] = +omega * (x_combined_slice[index] + 0.5 * xloc[d][0] * dx[slice_id]);
                            local_vg[2] = 0.0;
                            // Call the actual compute method
                            cell_inner_flux_loop<double>(omega, nf, A_, B_, q_combined_slice, local_f,
                                local_x, local_vg, this_ap, this_am, dim, d, dx[slice_id], fgamma,
                                de_switch_1, dim_offset * d + index,
                                dim_offset * flipped_dim - compressedH_DN[dim] + index,
                                face_offset);
                            // TODO Preparation for later SIMD masking (not
                            // supported yet)
                            this_ap *= mask;
                            this_am *= mask;
                            // Update maximum values
                            const double amax_tmp = max_wrapper(this_ap, (-this_am));
                            if (amax_tmp > current_amax) {
                                current_amax = amax_tmp;
                                current_d = d;
                            }
                            // Add results to the final flux buffer
                            for (int f = 1; f < nf; f++) {
                                f_combined_slice[dim * nf * q_inx3 + f * q_inx3 + index] +=
                                    quad_weights[fi] * local_f[f];
                            }
                        }
                    }
                    for (int f = 10; f < nf; f++) {
                        f_combined_slice[dim * nf * q_inx3 + index] +=
                            f_combined_slice[dim * nf * q_inx3 + f * q_inx3 + index];
                    }
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
                    sm_i[tid] = index;
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
                        amax[block_id + amax_slice_offset] = sm_amax[0];
                        amax_indices[block_id + max_indices_slice_offset] = sm_i[0];
                        amax_d[block_id + max_indices_slice_offset] = sm_d[0];
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
                        amax[block_id + amax_slice_offset] = current_amax;
                        amax_indices[block_id + max_indices_slice_offset] = index;
                        amax_d[block_id + max_indices_slice_offset + max_indices_slice_offset] = current_d;
                    }
                    // Save face to the end of the amax buffer This avoids putting combined_q back
                    // on the host side just to read those few values
                    const int flipped_dim = flip_dim(amax_d[block_id + max_indices_slice_offset], dim);
                    for (int f = 0; f < nf; f++) {
                        amax[number_blocks + block_id * 2 * nf + f + amax_slice_offset] =
                            q_combined_slice[amax_indices[block_id + max_indices_slice_offset] + f * face_offset +
                                dim_offset * amax_d[block_id + max_indices_slice_offset]];
                        amax[number_blocks + block_id * 2 * nf + nf + f + amax_slice_offset] =
                            q_combined_slice[amax_indices[block_id + max_indices_slice_offset] -
                            compressedH_DN[dim] +
                                f * face_offset + dim_offset * flipped_dim];
                    }
                }
            });
    }
}

/// Reconstruct with or without am
template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_int_buffer_t>
void reconstruct_impl(hpx::kokkos::executor<kokkos_backend_t>& executor, 
    typename Aggregated_Executor<hpx::kokkos::executor<kokkos_backend_t>>::Executor_Slice &agg_exec,
    const double omega,
    const int nf_, const int angmom_index_, const kokkos_int_buffer_t& smooth_field_,
    const kokkos_int_buffer_t& disc_detect_, kokkos_buffer_t& combined_q,
    const kokkos_buffer_t& combined_x, kokkos_buffer_t& combined_u, kokkos_buffer_t& AM,
    const kokkos_buffer_t& dx, const kokkos_buffer_t& cdiscs, const int n_species_, const int ndir,
    const int nangmom, const Kokkos::Array<long, 4>&& tiling_config) {
    const int blocks = q_inx3 / 64 + 1;
    const int number_slices = agg_exec.number_slices;
    if (agg_exec.sync_aggregation_slices()) {
        auto policy = Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<4>>(
                agg_exec.get_underlying_executor().instance(), {0, 0, 0, 0}, {number_slices, blocks, 8, 8},
                tiling_config),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        Kokkos::parallel_for(
            "kernel hydro reconstruct", policy, KOKKOS_LAMBDA(int slice_id, int idx, int idy, int idz) {
                const int sx_i = angmom_index_;
                const int zx_i = sx_i + NDIM;
                const int q_i = (idx) * 64 + (idy) * 8 + (idz);
                const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
                    (((q_i % q_inx2) / q_inx) + 2) * inx_large + (((q_i % q_inx2) % q_inx) + 2);

                const int u_slice_offset = (nf_ * H_N3 + padding) * slice_id;
                const int am_slice_offset = (NDIM * q_inx3 + padding) * slice_id;
                if (q_i < q_inx3) {
                    for (int n = 0; n < nangmom; n++) {
                        AM[n * am_offset + q_i + am_slice_offset] =
                            combined_u[(zx_i + n) * u_face_offset + i + u_slice_offset] * combined_u[i + u_slice_offset];
                    }
                    for (int d = 0; d < ndir; d++) {
                        cell_reconstruct_inner_loop_p1(nf_, angmom_index_, smooth_field_,
                            disc_detect_, combined_q, combined_u, AM, dx[slice_id], cdiscs, d, i, q_i, ndir,
                            nangmom, slice_id);
                    }
                    // Phase 2
                    for (int d = 0; d < ndir; d++) {
                        cell_reconstruct_inner_loop_p2(omega, angmom_index_, combined_q, combined_x,
                            combined_u, AM, dx[slice_id], d, i, q_i, ndir, nangmom, n_species_, nf_, slice_id);
                    }
                }
            });
    }
}

/// Optimized for reconstruct without am correction
template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_int_buffer_t>
void reconstruct_no_amc_impl(hpx::kokkos::executor<kokkos_backend_t>& executor, 
    typename Aggregated_Executor<hpx::kokkos::executor<kokkos_backend_t>>::Executor_Slice &agg_exec,
    const double omega,
    const int nf_, const int angmom_index_, const kokkos_int_buffer_t& smooth_field_,
    const kokkos_int_buffer_t& disc_detect_, kokkos_buffer_t& combined_q,
    const kokkos_buffer_t& combined_x, kokkos_buffer_t& combined_u, kokkos_buffer_t& AM,
    const kokkos_buffer_t& dx, const kokkos_buffer_t& cdiscs, const int n_species_, const int ndir,
    const int nangmom, const Kokkos::Array<long, 4>&& tiling_config) {
    const int blocks = q_inx3 / 64 + 1;
    const int number_slices = agg_exec.number_slices;
    if (agg_exec.sync_aggregation_slices()) {
        auto policy = Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<decltype(agg_exec.get_underlying_executor().instance()), Kokkos::Rank<4>>(
                agg_exec.get_underlying_executor().instance(), {0, 0, 0, 0}, {number_slices, blocks, 8, 8}, tiling_config),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        Kokkos::parallel_for(
            "kernel hydro reconstruct", policy, KOKKOS_LAMBDA(int slice_id, int idx, int idy, int idz) {
                const int q_i = (idx) *64 + (idy) *8 + (idz);
                const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
                    (((q_i % q_inx2) / q_inx) + 2) * inx_large + (((q_i % q_inx2) % q_inx) + 2);
                if (q_i < q_inx3) {
                    for (int d = 0; d < ndir; d++) {
                        cell_reconstruct_inner_loop_p1(nf_, angmom_index_, smooth_field_,
                            disc_detect_, combined_q, combined_u, AM, dx[slice_id], cdiscs, d, i, q_i, ndir,
                            nangmom, slice_id);
                    }
                    // Phase 2
                    for (int d = 0; d < ndir; d++) {
                        cell_reconstruct_inner_loop_p2(omega, angmom_index_, combined_q, combined_x,
                            combined_u, AM, dx[slice_id], d, i, q_i, ndir, nangmom, n_species_, nf_, slice_id);
                    }
                }
            });
    }
}

template <typename kokkos_backend_t, typename kokkos_buffer_t>
void hydro_pre_recon_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
    typename Aggregated_Executor<hpx::kokkos::executor<kokkos_backend_t>>::Executor_Slice &agg_exec,
    const kokkos_buffer_t& large_x, const double omega, const bool angmom, kokkos_buffer_t& u,
    const int nf, const int n_species, const Kokkos::Array<long, 4>&& tiling_config) {
    const int number_slices = agg_exec.number_slices;

    if (agg_exec.sync_aggregation_slices()) {
      auto policy = Kokkos::Experimental::require(
          Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<4>>(
              agg_exec.get_underlying_executor().instance(), {0, 0, 0, 0}, {number_slices, inx_large, inx_large, inx_large}),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight);
      Kokkos::parallel_for(
          "kernel hydro pre recon", policy, KOKKOS_LAMBDA(int slice_id, int idx, int idy, int idz) {
              cell_hydro_pre_recon(large_x, omega, angmom, u, nf, n_species, idx, idy, idz, slice_id); // TODO add slice id
        });
    }
}

template <typename kokkos_backend_t, typename kokkos_buffer_t>
void find_contact_discs_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
    typename Aggregated_Executor<hpx::kokkos::executor<kokkos_backend_t>>::Executor_Slice &agg_exec,
    const kokkos_buffer_t& u, kokkos_buffer_t& P, kokkos_buffer_t& disc, const double A_,
    const double B_, const double fgamma_, const double de_switch_1, const size_t ndir, const size_t nf,
    const Kokkos::Array<long, 4>&& tiling_config_phase1,
    const Kokkos::Array<long, 4>&& tiling_config_phase2) {
    const int number_slices = agg_exec.number_slices;
    if (agg_exec.sync_aggregation_slices()) {
        auto policy_phase_1 = Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<4>>(
                agg_exec.get_underlying_executor().instance(), {0, 0, 0, 0},
                {number_slices, inx_normal, inx_normal, inx_normal}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        Kokkos::parallel_for(
            "kernel find contact discs 1", policy_phase_1,
            KOKKOS_LAMBDA(int slice_id, int idx, int idy, int idz) {
                cell_find_contact_discs_phase1(
                    P, u, A_, B_, fgamma_, de_switch_1, nf, idx, idy, idz, slice_id);
            });

        auto policy_phase_2 = Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<4>>(
                agg_exec.get_underlying_executor().instance(), {0, 0, 0, 0}, {number_slices, q_inx, q_inx, q_inx}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        Kokkos::parallel_for(
            "kernel find contact discs 2", policy_phase_2,
            KOKKOS_LAMBDA(int slice_id, int idx, int idy, int idz) {
                cell_find_contact_discs_phase2(disc, P, fgamma_, ndir, idx, idy, idz, slice_id);
            });
    }
}

template <typename executor_t,
    std::enable_if_t<is_kokkos_device_executor<executor_t>::value, int> = 0>
timestep_t device_interface_kokkos_hydro(executor_t& exec,
    const aggregated_host_buffer<double, executor_t>& combined_x,
    const aggregated_host_buffer<double, executor_t>& combined_large_x,
    aggregated_host_buffer<double, executor_t>& combined_u,
    const aggregated_host_buffer<int, executor_t>& disc_detect,
    const aggregated_host_buffer<int, executor_t>& smooth_field,
    aggregated_host_buffer<double, executor_t>& host_f, const size_t ndir, const size_t nf,
    const bool angmom, const size_t n_species, const double omega, const int angmom_index,
    const int nangmom, const aggregated_host_buffer<double, executor_t>& dx, const double A_, const double B_, const double fgamma,
    const double de_switch_1, typename
    Aggregated_Executor<executor_t>::Executor_Slice& agg_exec,
    Allocator_Slice<double, kokkos_host_allocator<double>, executor_t>& alloc_host_double,
    Allocator_Slice<int, kokkos_host_allocator<int>, executor_t>& alloc_host_int) {
    // How many executor slices are working together and what's our ID?
    const size_t slice_id = agg_exec.id;
    const size_t number_slices = agg_exec.number_slices;
    const size_t max_slices = opts().max_executor_slices;

    // Slice offsets
    const int u_slice_offset = nf * H_N3 + padding;
    constexpr int x_slice_offset = NDIM * q_inx3 + padding;
    const int disc_detect_slice_offset = nf;
    const int smooth_slice_offset = nf;
    constexpr int large_x_slice_offset = (H_N3 * NDIM + padding); 
    //const int q_slice_offset = (nf_ * 27 * H_N3 + padding) 
    const int f_slice_offset = (NDIM* nf *  q_inx3 + padding);
    const int disc_offset = ndir / 2 * H_N3 + padding;

    auto alloc_device_int =
        agg_exec
            .template make_allocator<int, kokkos_device_allocator<int>>();
    auto alloc_device_double =
        agg_exec
            .template make_allocator<double, kokkos_device_allocator<double>>();

    // Find contact discs
    aggregated_device_buffer<double, executor_t> u(
        alloc_device_double, (nf * H_N3 + padding) * max_slices);
    aggregated_deep_copy(agg_exec, u, combined_u);
    aggregated_device_buffer<double, executor_t> P(
        alloc_device_double, (H_N3 + padding) * max_slices);
    aggregated_device_buffer<double, executor_t> disc(
        alloc_device_double, (ndir / 2 * H_N3 + padding) * max_slices);
    find_contact_discs_impl(exec, agg_exec, u, P, disc, physics<NDIM>::A_, physics<NDIM>::B_,
        physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1, ndir, nf, {1, 1, 12,
        12}, {1, 1, 10, 10});

    // Pre recon
    aggregated_device_buffer<double, executor_t> large_x(
        alloc_device_double, (NDIM * H_N3 + padding) * max_slices);
    aggregated_deep_copy(agg_exec, large_x, combined_large_x);
    hydro_pre_recon_impl(exec, agg_exec, large_x, omega, angmom, u, nf, n_species, {1, 1,
        14, 14});

    // Reconstruct
    aggregated_device_buffer<double, executor_t> x(
        alloc_device_double, (NDIM * q_inx3 + padding) * max_slices);
    aggregated_deep_copy(agg_exec, x, combined_x);
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
        reconstruct_impl(exec, agg_exec, omega, nf, angmom_index, device_smooth_field, device_disc_detect, q,
            x, u, AM, dx_device, disc, n_species, ndir, nangmom, {1, 1, 8, 8});
    } else {
        reconstruct_no_amc_impl(exec, agg_exec, omega, nf, angmom_index, device_smooth_field,
            device_disc_detect, q, x, u, AM, dx_device, disc, n_species, ndir, nangmom, {1, 1, 8, 8});
    }

    // Flux
    const device_buffer<bool>& masks =
        get_flux_device_masks<device_buffer<bool>, host_buffer<bool>, executor_t>(agg_exec.get_underlying_executor());
    const int number_blocks = (q_inx3 / 128 + 1) * 1;
    aggregated_device_buffer<double, executor_t> amax(
        alloc_device_double, number_blocks * NDIM * (1 + 2 * nf) * max_slices);
    aggregated_device_buffer<int, executor_t> amax_indices(
        alloc_device_int, number_blocks * NDIM * max_slices);
    aggregated_device_buffer<int, executor_t> amax_d(
        alloc_device_int, number_blocks * NDIM * max_slices);
    aggregated_device_buffer<double, executor_t> f(
        alloc_device_double, (NDIM * nf * q_inx3 + padding) * max_slices);
    flux_impl(exec, agg_exec, q, x, f, amax, amax_indices, amax_d, masks, omega, dx_device, A_, B_, nf, fgamma,
        de_switch_1, NDIM * number_blocks, 128);
    aggregated_host_buffer<double, executor_t> host_amax(
        alloc_host_double, number_blocks * NDIM * (1 + 2 * nf) * max_slices);
    aggregated_host_buffer<int, executor_t> host_amax_indices(
        alloc_host_int, number_blocks * NDIM * max_slices);
    aggregated_host_buffer<int, executor_t> host_amax_d(
        alloc_host_int, number_blocks * NDIM * max_slices);

    aggregated_deep_copy(agg_exec, host_amax, amax);
    aggregated_deep_copy(agg_exec, host_amax_indices, amax_indices);
    aggregated_deep_copy(agg_exec, host_amax_d, amax_d);

    auto fut = aggregrated_deep_copy_async<executor_t>(agg_exec, host_f, f);
    fut.get();

    const int amax_slice_offset = NDIM * (1 + 2 * nf) * number_blocks * slice_id;
    const int max_indices_slice_offset = NDIM * number_blocks * slice_id;

    // Find Maximum
    size_t current_max_slot = 0;
    for (size_t dim_i = 1; dim_i < number_blocks * NDIM; dim_i++) {
        if (host_amax[dim_i + amax_slice_offset] >
            host_amax[current_max_slot + amax_slice_offset]) {
            current_max_slot = dim_i;
        } else if (host_amax[dim_i + amax_slice_offset] ==
            host_amax[current_max_slot + amax_slice_offset]) {
            if (host_amax_indices[dim_i + max_indices_slice_offset] <
                host_amax_indices[current_max_slot + max_indices_slice_offset])
                current_max_slot = dim_i;
        }
    }

    // Create & Return timestep_t type
    std::vector<double> URs(nf), ULs(nf);
    const size_t current_max_index = host_amax_indices[current_max_slot + max_indices_slice_offset];
    /* const size_t current_d = host_amax_d[current_max_slot]; */
    timestep_t ts;
    ts.a = host_amax[current_max_slot + amax_slice_offset];
    ts.x = combined_x[current_max_index + x_slice_offset * slice_id];
    ts.y = combined_x[current_max_index + q_inx3 + x_slice_offset * slice_id];
    ts.z = combined_x[current_max_index + 2 * q_inx3+ x_slice_offset * slice_id];
    ts.z = combined_x[current_max_index + 2 * q_inx3 + x_slice_offset * slice_id];
    const size_t current_i = current_max_slot;
    const size_t current_dim = current_max_slot / number_blocks;
    /* const auto flipped_dim = flip_dim(current_d, current_dim); */
    constexpr int compressedH_DN[3] = {q_inx2, q_inx, 1};
    for (int f = 0; f < nf; f++) {
        URs[f] = host_amax[NDIM * number_blocks + current_i * 2 * nf + f + amax_slice_offset];
        ULs[f] = host_amax[NDIM * number_blocks + current_i * 2 * nf + nf + f + amax_slice_offset];
    }
    ts.ul = std::move(URs);
    ts.ur = std::move(ULs);
    ts.dim = current_dim;
    return ts;
}

template <typename executor_t,
    std::enable_if_t<is_kokkos_host_executor<executor_t>::value, int> = 0>
timestep_t device_interface_kokkos_hydro(executor_t& exec, const aggregated_host_buffer<double, executor_t>& combined_x,
    const aggregated_host_buffer<double, executor_t>& combined_large_x,aggregated_host_buffer<double, executor_t>& combined_u,
    const aggregated_host_buffer<int, executor_t>& disc_detect, const aggregated_host_buffer<int, executor_t>& smooth_field,
    aggregated_host_buffer<double, executor_t>& f, const size_t ndir, const size_t nf, const bool angmom,
    const size_t n_species, const double omega, const int angmom_index, const int nangmom,
    const aggregated_host_buffer<double, executor_t>& dx, const double A_, const double B_, const double fgamma,
    const double de_switch_1,
    typename Aggregated_Executor<executor_t>::Executor_Slice &agg_exec,
    Allocator_Slice<double, kokkos_host_allocator<double>, executor_t> &alloc_host_double,
    Allocator_Slice<int, kokkos_host_allocator<int>, executor_t> &alloc_host_int) {

    // How many executor slices are working together and what's our ID?
    const size_t slice_id = agg_exec.id;
    const size_t number_slices = agg_exec.number_slices;
    const size_t max_slices = opts().max_executor_slices;

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
    find_contact_discs_impl(exec, agg_exec, combined_u, P, disc, physics<NDIM>::A_, physics<NDIM>::B_,
        physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1, ndir, nf,
        {1, inx_normal, inx_normal, inx_normal}, {1, q_inx, q_inx, q_inx});

    // Pre recon
    hydro_pre_recon_impl(exec, agg_exec, combined_large_x, omega, angmom, combined_u, nf, n_species,
        {1, inx_large, inx_large, inx_large});

    // Reconstruct
    aggregated_host_buffer<double, executor_t> q(
        alloc_host_double, (nf * 27 * q_inx3 + padding) * max_slices);
    aggregated_host_buffer<double, executor_t> AM(
        alloc_host_double, (NDIM * q_inx3 + padding) * max_slices);
    reconstruct_impl(exec, agg_exec, omega, nf, angmom_index, smooth_field, disc_detect, q, combined_x,
        combined_u, AM, dx, disc, n_species, ndir, nangmom, {1, INX, INX, INX});

    // Flux
    const int blocks = NDIM * (q_inx3 / 128 + 1) * 128;
    const host_buffer<bool>& masks = get_flux_host_masks<host_buffer<bool>>();

    aggregated_host_buffer<double, executor_t> amax(
        alloc_host_double, blocks * (1 + 2 * nf) * max_slices);
    aggregated_host_buffer<int, executor_t> amax_indices(alloc_host_int, blocks * max_slices);
    aggregated_host_buffer<int, executor_t> amax_d(alloc_host_int, blocks * max_slices);
    flux_impl_teamless(exec, agg_exec, q, combined_x, f, amax, amax_indices, amax_d, masks, omega, dx, A_, B_,
        nf, fgamma, de_switch_1, blocks, 1);

    sync_kokkos_host_kernel(exec);

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
    const double omega, const size_t n_species, executor_t& executor,
    std::vector<hydro_state_t<std::vector<safe_real>>>& F) {
    static const cell_geometry<NDIM, INX> geo;

    auto executor_slice_fut = hydro_kokkos_agg_executor_pool<executor_t>::request_executor_slice();
    auto ret_fut = executor_slice_fut.value().then(hpx::util::annotated_function([&](auto && fut) {
      typename Aggregated_Executor<executor_t>::Executor_Slice agg_exec = fut.get();
      // How many executor slices are working together and what's our ID?
      const size_t slice_id = agg_exec.id;
      const size_t number_slices = agg_exec.number_slices;
      const size_t max_slices = opts().max_executor_slices;

      // Slice offsets
      const int u_slice_offset = hydro.get_nf() * H_N3 + padding;
      constexpr int x_slice_offset = NDIM * q_inx3 + padding;
      const int disc_detect_slice_offset = hydro.get_nf();
      const int smooth_slice_offset = hydro.get_nf();
      constexpr int large_x_slice_offset = (H_N3 * NDIM + padding); 
      //const int q_slice_offset = (nf_ * 27 * H_N3 + padding) 
      const int f_slice_offset = (NDIM * hydro.get_nf()   *  q_inx3 + padding);
      const int disc_offset = geo.NDIR / 2 * H_N3 + padding;

      // Get allocators of all the executors working together
      // Allocator_Slice<double, kokkos_host_allocator<double>, executor_t>
      Allocator_Slice<double, kokkos_host_allocator<double>, executor_t> alloc_host_double =
          agg_exec
              .template make_allocator<double, kokkos_host_allocator<double>>();
      Allocator_Slice<int, kokkos_host_allocator<int>, executor_t> alloc_host_int =
          agg_exec
              .template make_allocator<int, kokkos_host_allocator<int>>();

      // Host buffers
      aggregated_host_buffer<double, executor_t> combined_x(
          alloc_host_double, (NDIM * q_inx3 + padding) * max_slices);
      aggregated_host_buffer<double, executor_t> combined_large_x(
          alloc_host_double, (NDIM * H_N3 + padding) * max_slices);
      aggregated_host_buffer<double, executor_t> combined_u(
          alloc_host_double, (hydro.get_nf() * H_N3 + padding) *
          max_slices);
      aggregated_host_buffer<int, executor_t> disc_detect(
          alloc_host_int, (hydro.get_nf()) * max_slices);
      aggregated_host_buffer<int, executor_t> smooth_field(
          alloc_host_int, (hydro.get_nf()) * max_slices);
      aggregated_host_buffer<double, executor_t> f(
          alloc_host_double, (NDIM * hydro.get_nf() * q_inx3 + padding) * max_slices);


      // Convert input
      convert_x_structure(X, combined_x.data() + x_slice_offset * slice_id);
      for (int n = 0; n < NDIM; n++) {
          std::copy(X[n].begin(), X[n].end(), combined_large_x.data() + n * H_N3 + large_x_slice_offset * slice_id);
      }
      for (int f = 0; f < hydro.get_nf(); f++) {
          std::copy(U[f].begin(), U[f].end(), combined_u.data() + f * H_N3 + u_slice_offset * slice_id);
      }
      const auto& disc_detect_bool = hydro.get_disc_detect();
      const auto& smooth_bool = hydro.get_smooth_field();
      for (auto f = 0; f < hydro.get_nf(); f++) {
          disc_detect[f + disc_detect_slice_offset * slice_id] = disc_detect_bool[f];
          smooth_field[f + smooth_slice_offset * slice_id] = smooth_bool[f];
      }

      aggregated_host_buffer<double, executor_t> dx(
          alloc_host_double, max_slices);
      dx[slice_id] = X[0][geo.H_DNX] - X[0][0];

      // Either handles the launches on the CPU or on the GPU depending on the passed executor
      auto max_lambda = device_interface_kokkos_hydro(executor, combined_x, combined_large_x,
          combined_u, disc_detect, smooth_field, f, geo.NDIR, hydro.get_nf(),
          hydro.get_angmom_index() != -1, n_species, omega, hydro.get_angmom_index(), geo.NANGMOM,
          dx, physics<NDIM>::A_, physics<NDIM>::B_, physics<NDIM>::fgamma_,
          physics<NDIM>::de_switch_1, agg_exec, alloc_host_double, alloc_host_int);

      // Convert output
      for (int dim = 0; dim < NDIM; dim++) {
          for (integer field = 0; field != opts().n_fields; ++field) {
              const auto dim_offset = dim * opts().n_fields * q_inx3 + field * q_inx3;
              for (integer i = 0; i <= INX; ++i) {
                  for (integer j = 0; j <= INX; ++j) {
                      for (integer k = 0; k <= INX; ++k) {
                          const auto i0 = findex(i, j, k);
                          const auto input_index =
                              (i + 1) * q_inx * q_inx + (j + 1) * q_inx + (k + 1);
                          F[dim][field][i0] = f[dim_offset + input_index + f_slice_offset * slice_id];
                      }
                  }
              }
          }
      }
      // std::cout << max_lambda.a << std::endl;
      return max_lambda;
          }));
    return ret_fut.get();
}
#endif
