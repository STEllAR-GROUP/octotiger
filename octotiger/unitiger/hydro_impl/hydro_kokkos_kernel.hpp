#pragma once

#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/common_kernel/kokkos_util.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"    // required for wrappers

#include "octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp"
#include "octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp"

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

template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_int_buffer_t,
    typename kokkos_mask_t>
void flux_impl(hpx::kokkos::executor<kokkos_backend_t>& executor, const kokkos_buffer_t& q_combined,
    const kokkos_buffer_t& x_combined, kokkos_buffer_t& f_combined, kokkos_buffer_t& amax,
    kokkos_int_buffer_t& amax_indices, kokkos_int_buffer_t& amax_d, const kokkos_mask_t& masks,
    const double omega, const double dx, const double A_, const double B_, const int nf,
    const double fgamma, const double de_switch_1, const int number_blocks, const int team_size) {
    using policytype = Kokkos::TeamPolicy<decltype(executor.instance())>;
    assert(
        (team_size == 128) || (team_size == 1));
    auto policy = policytype(executor.instance(), number_blocks, team_size);
    using membertype = typename policytype::member_type;
    policy.set_scratch_size(0, Kokkos::PerTeam(team_size * (sizeof(double) + sizeof(int) * 2)));
    Kokkos::parallel_for(
        "kernel hydro flux", policy, KOKKOS_LAMBDA(const membertype& team_handle) {
            // Set during cmake step with -DOCTOTIGER_WITH_MAX_NUMBER_FIELDS
            double local_f[OCTOTIGER_MAX_NUMBER_FIELDS];
            // assumes maximal number (given by cmake) of species in a simulation.
            // Not the most elegant solution and rather old-fashion but one that works.
            // May be changed to a more flexible sophisticated object.
            for (int f = 0; f < nf; f++) {
                local_f[f] = 0.0;
            }
            double local_x[3] = {0.0, 0.0, 0.0};
            double local_vg[3] = {0.0, 0.0, 0.0};

            const int blocks_per_dim = number_blocks / NDIM;
            double current_amax = 0.0;
            int current_d = 0;

            // 3 dim 1000 i workitems
            const int dim = (team_handle.league_rank() / blocks_per_dim);    // blockIdx.z;
            const int tid =
                team_handle.team_rank();    // threadIdx.x * 64 + threadIdx.y * 8 + threadIdx.z;
            // const int index = (team_handle.league_rank() % blocks_per_dim) * team_size + tid +
            //     q_inx2;    //  blockIdx.y * 128 + tid + 104;
            const int index = (team_handle.league_rank() % blocks_per_dim) * team_size + tid;
            const int block_id =
                (team_handle.league_rank() % blocks_per_dim) + dim * blocks_per_dim;
            for (int f = 0; f < nf; f++) {
                f_combined[dim * nf * q_inx3 + f * q_inx3 + index] = 0.0;
            }
            if (tid == 0) {
                amax[block_id] = 0.0;
                amax_indices[block_id] = 0;
                amax_d[block_id] = 0;
            }

            Kokkos::View<double*,
                typename policytype::execution_space::scratch_memory_space>
                sm_amax(team_handle.team_scratch(0), team_size);
            Kokkos::View<int*, typename policytype::execution_space::scratch_memory_space>
                sm_i(team_handle.team_scratch(0), team_size);
            Kokkos::View<int*, typename policytype::execution_space::scratch_memory_space>
                sm_d(team_handle.team_scratch(0), team_size);
            // Get around scratch memory limitation using the serial backend and only access it for team_sizes > 1
            if (team_size > 1) {
                sm_amax[tid] = 0;
                sm_d[tid] = 0;
                sm_i[tid] = 0;
            }

            if (index < q_inx3) {
                double mask = masks[index + dim * dim_offset];
                if (mask != 0.0) {
                    for (int fi = 0; fi < 9; fi++) {            // 9
                        double this_ap = 0.0, this_am = 0.0;    // tmps
                        const int d = faces[dim][fi];
                        const int flipped_dim = flip_dim(d, dim);
                        for (int dim = 0; dim < 3; dim++) {
                            local_x[dim] =
                                x_combined[dim * q_inx3 + index] + (0.5 * xloc[d][dim] * dx);
                        }
                        local_vg[0] = -omega * (x_combined[q_inx3 + index] + 0.5 * xloc[d][1] * dx);
                        local_vg[1] = +omega * (x_combined[index] + 0.5 * xloc[d][0] * dx);
                        local_vg[2] = 0.0;
                        cell_inner_flux_loop<double>(omega, nf, A_, B_, q_combined, local_f,
                            local_x, local_vg, this_ap, this_am, dim, d, dx, fgamma, de_switch_1,
                            dim_offset * d + index,
                            dim_offset * flipped_dim - compressedH_DN[dim] + index, face_offset);
                        this_ap *= mask;
                        this_am *= mask;
                        const double amax_tmp = max_wrapper(this_ap, (-this_am));
                        if (amax_tmp > current_amax) {
                            current_amax = amax_tmp;
                            current_d = d;
                        }
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
                // Parallel maximum search within workgroup
                if (team_handle.team_size() == 128) {
                    sm_amax[tid] = current_amax;
                    sm_d[tid] = current_d;
                    sm_i[tid] = index;
                    team_handle.team_barrier();
                    // Max reduction with multiple warps
                    for (int tid_border = 64; tid_border >= 32; tid_border /= 2) {
                        if (tid < tid_border) {
                            if (sm_amax[tid + tid_border] > sm_amax[tid]) {
                                sm_amax[tid] = sm_amax[tid + tid_border];
                                sm_d[tid] = sm_d[tid + tid_border];
                                sm_i[tid] = sm_i[tid + tid_border];
                            }
                        }
                        team_handle.team_barrier();
                    }
                    // Max reduction within one warps
                    if (tid == 0) {
                        for (int tid_border = 16; tid_border >= 1; tid_border /= 2) {
                            if (tid < tid_border) {
                                if (sm_amax[tid + tid_border] > sm_amax[tid]) {
                                    sm_amax[tid] = sm_amax[tid + tid_border];
                                    sm_d[tid] = sm_d[tid + tid_border];
                                    sm_i[tid] = sm_i[tid + tid_border];
                                }
                            }
                        }
                        amax[block_id] = sm_amax[0];
                        amax_indices[block_id] = sm_i[0];
                        amax_d[block_id] = sm_d[0];
                    }
                }

                // Write Maximum of local team to amax
                if (tid == 0) {
                    if (team_handle.team_size() == 1) {
                        amax[block_id] = current_amax;
                        amax_indices[block_id] = index;
                        amax_d[block_id] = current_d;
                    }

                    // Save face to the end of the amax buffer
                    // This avoids putting combined_q back on the host side just to read
                    // those few values
                    const int flipped_dim = flip_dim(amax_d[block_id], dim);
                    // if (block_id == 137)
                    //     printf("%i %i %i %f \n", block_id, amax_d[block_id], amax_indices[block_id], amax[block_id]);
                    for (int f = 0; f < nf; f++) {
                        amax[number_blocks + block_id * 2 * nf + f] =
                            q_combined[amax_indices[block_id] + f * face_offset +
                                dim_offset * amax_d[block_id]];
                        amax[number_blocks + block_id * 2 * nf + nf + f] =
                            q_combined[amax_indices[block_id] - compressedH_DN[dim] +
                                f * face_offset + dim_offset * flipped_dim];
                    }
                }
            }
        });
}

template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_int_buffer_t>
void reconstruct_impl(hpx::kokkos::executor<kokkos_backend_t>& executor, const double omega,
    const int nf_, const int angmom_index_, const kokkos_int_buffer_t& smooth_field_,
    const kokkos_int_buffer_t& disc_detect_, kokkos_buffer_t& combined_q,
    const kokkos_buffer_t& combined_x, kokkos_buffer_t& combined_u, kokkos_buffer_t& AM,
    const double dx, const kokkos_buffer_t& cdiscs, const int n_species_, const int ndir,
    const int nangmom, const Kokkos::Array<long, 3>&& tiling_config) {
    const int blocks = q_inx3 / 64 + 1; 
    auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>>(
            executor.instance(), {0, 0, 0}, {blocks, 8, 8}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_for(
        "kernel hydro reconstruct", policy, KOKKOS_LAMBDA(int idx, int idy, int idz) {
            const int sx_i = angmom_index_;
            const int zx_i = sx_i + NDIM;
            const int q_i = (idx) * 64 + (idy) *8 + (idz);
            const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large + (((q_i % q_inx2) / q_inx) + 2) * inx_large +
                (((q_i % q_inx2) % q_inx) + 2);
            if (q_i < q_inx3) {
                for (int n = 0; n < nangmom; n++) {
                    AM[n * am_offset + q_i] =
                        combined_u[(zx_i + n) * u_face_offset + i] * combined_u[i];
                }
                for (int d = 0; d < ndir; d++) {
                    cell_reconstruct_inner_loop_p1(nf_, angmom_index_, smooth_field_, disc_detect_,
                        combined_q, combined_u, AM, dx, cdiscs, d, i, q_i, ndir, nangmom);
                }
                // Phase 2
                for (int d = 0; d < ndir; d++) {
                    cell_reconstruct_inner_loop_p2(omega, angmom_index_, combined_q, combined_x,
                        combined_u, AM, dx, d, i, q_i, ndir, nangmom, n_species_);
                }
            }
        });
}

template <typename kokkos_backend_t, typename kokkos_buffer_t>
void hydro_pre_recon_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
    const kokkos_buffer_t& large_x, const double omega, const bool angmom, kokkos_buffer_t& u,
    const int nf, const int n_species, const Kokkos::Array<long, 3>&& tiling_config) {
    auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>>(
            executor.instance(), {0, 0, 0}, {inx_large, inx_large, inx_large}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_for(
        "kernel hydro pre recon", policy, KOKKOS_LAMBDA(int idx, int idy, int idz) {
            cell_hydro_pre_recon(large_x, omega, angmom, u, nf, n_species, idx, idy, idz);
        });
}

template <typename kokkos_backend_t, typename kokkos_buffer_t>
void find_contact_discs_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
    const kokkos_buffer_t& u, kokkos_buffer_t& P, kokkos_buffer_t& disc, const double A_,
    const double B_, const double fgamma_, const double de_switch_1, const size_t ndir,
    const Kokkos::Array<long, 3>&& tiling_config_phase1,
    const Kokkos::Array<long, 3>&& tiling_config_phase2) {
    auto policy_phase_1 = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>>(
            executor.instance(), {0, 0, 0}, {inx_normal, inx_normal, inx_normal}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_for(
        "kernel find contact discs 1", policy_phase_1, KOKKOS_LAMBDA(int idx, int idy, int idz) {
            cell_find_contact_discs_phase1(P, u, A_, B_, fgamma_, de_switch_1, idx, idy, idz);
        });

    auto policy_phase_2 = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>>(
            executor.instance(), {0, 0, 0}, {q_inx, q_inx, q_inx}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_for(
        "kernel find contact discs 2", policy_phase_2, KOKKOS_LAMBDA(int idx, int idy, int idz) {
            cell_find_contact_discs_phase2(disc, P, fgamma_, ndir, idx, idy, idz);
        });
}

template <typename executor_t,
    std::enable_if_t<is_kokkos_device_executor<executor_t>::value, int> = 0>
timestep_t device_interface_kokkos_hydro(executor_t& exec, const host_buffer<double>& combined_x,
    const host_buffer<double>& combined_large_x, host_buffer<double>& combined_u,
    const host_buffer<int>& disc_detect, const host_buffer<int>& smooth_field,
    host_buffer<double>& host_f, const size_t ndir, const size_t nf, const bool angmom,
    const size_t n_species, const double omega, const int angmom_index, const int nangmom,
    const double dx, const double A_, const double B_, const double fgamma,
    const double de_switch_1) {
    // Find contact discs
    device_buffer<double> u(nf * H_N3 + padding);
    Kokkos::deep_copy(exec.instance(), u, combined_u);
    device_buffer<double> P(H_N3 + padding);
    device_buffer<double> disc(ndir / 2 * H_N3 + padding);
    find_contact_discs_impl(exec, u, P, disc, physics<NDIM>::A_, physics<NDIM>::B_,
        physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1, ndir, {1, 12, 12}, {1, 10, 10});

    // Pre recon
    device_buffer<double> large_x(NDIM * H_N3 + padding);
    Kokkos::deep_copy(exec.instance(), large_x, combined_large_x);
    hydro_pre_recon_impl(exec, large_x, omega, angmom, u, nf, n_species, {1, 14, 14});

    // Reconstruct
    device_buffer<double> x(NDIM * q_inx3 + padding);
    Kokkos::deep_copy(exec.instance(), x, combined_x);
    device_buffer<int> device_disc_detect(nf);
    Kokkos::deep_copy(exec.instance(), device_disc_detect, disc_detect);
    device_buffer<int> device_smooth_field(nf);
    Kokkos::deep_copy(exec.instance(), device_smooth_field, smooth_field);
    device_buffer<double> q(nf * 27 * q_inx3 + padding);
    device_buffer<double> AM(NDIM * q_inx3 + padding);
    reconstruct_impl(exec, omega, nf, angmom_index, device_smooth_field, device_disc_detect, q, x,
        u, AM, dx, disc, n_species, ndir, nangmom, {1, 8, 8});

    // Flux
    const device_buffer<bool>& masks =
        get_flux_device_masks<device_buffer<bool>, host_buffer<bool>, executor_t>(exec);
    const int number_blocks = (q_inx3 / 128 + 1);
    device_buffer<double> amax(number_blocks * NDIM * (1 + 2 * nf));
    device_buffer<int> amax_indices(number_blocks * NDIM);
    device_buffer<int> amax_d(number_blocks * NDIM);
    device_buffer<double> f(NDIM * nf * q_inx3 + padding);
    flux_impl(exec, q, x, f, amax, amax_indices, amax_d, masks, omega, dx, A_, B_, nf, fgamma,
        de_switch_1, NDIM* number_blocks, 128);
    host_buffer<double> host_amax(number_blocks * NDIM * (1 + 2 * nf));
    host_buffer<int> host_amax_indices(number_blocks * NDIM);
    host_buffer<int> host_amax_d(number_blocks * NDIM);
    Kokkos::deep_copy(exec.instance(), host_amax, amax);
    Kokkos::deep_copy(exec.instance(), host_amax_indices, amax_indices);
    Kokkos::deep_copy(exec.instance(), host_amax_d, amax_d);

    auto fut = hpx::kokkos::deep_copy_async(exec.instance(), host_f, f);
    fut.get();

    // TODO create Maximum method

    // Find Maximum
    size_t current_max_slot = 0;
    for (size_t dim_i = 1; dim_i < number_blocks * NDIM; dim_i++) {
        if (host_amax[dim_i] > host_amax[current_max_slot]) {
            current_max_slot = dim_i;
        }
    }

    // Create & Return timestep_t type
    std::vector<double> URs(nf), ULs(nf);
    const size_t current_max_index = host_amax_indices[current_max_slot];
    const size_t current_d = host_amax_d[current_max_slot];
    timestep_t ts;
    ts.a = host_amax[current_max_slot];
    ts.x = combined_x[current_max_index];
    ts.y = combined_x[current_max_index + q_inx3];
    ts.z = combined_x[current_max_index + 2 * q_inx3];
    const size_t current_i = current_max_slot;
    const size_t current_dim = current_max_slot / number_blocks;
    // TODO is this flip_dim call correct?
    const auto flipped_dim = flip_dim(current_d, current_dim);
    constexpr int compressedH_DN[3] = {q_inx2, q_inx, 1};
    for (int f = 0; f < nf; f++) {
        URs[f] = host_amax[NDIM * number_blocks + current_i * 2 * nf + f];
        ULs[f] = host_amax[NDIM * number_blocks + current_i * 2 * nf + nf + f];
    }
    ts.ul = std::move(ULs);
    ts.ur = std::move(URs);
    ts.dim = current_dim;
    return ts;
}

template <typename executor_t,
    std::enable_if_t<is_kokkos_host_executor<executor_t>::value, int> = 0>
timestep_t device_interface_kokkos_hydro(executor_t& exec, const host_buffer<double>& combined_x,
    const host_buffer<double>& combined_large_x, host_buffer<double>& combined_u,
    const host_buffer<int>& disc_detect, const host_buffer<int>& smooth_field,
    host_buffer<double>& f, const size_t ndir, const size_t nf, const bool angmom,
    const size_t n_species, const double omega, const int angmom_index, const int nangmom,
    const double dx, const double A_, const double B_, const double fgamma,
    const double de_switch_1) {
    // Find contact discs
    host_buffer<double> P(H_N3 + padding);
    host_buffer<double> disc(ndir / 2 * H_N3 + padding);
    find_contact_discs_impl(exec, combined_u, P, disc, physics<NDIM>::A_, physics<NDIM>::B_,
        physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1, ndir, {inx_normal, inx_normal, inx_normal}, {q_inx, q_inx, q_inx});

    // Pre recon
    hydro_pre_recon_impl(
        exec, combined_large_x, omega, angmom, combined_u, nf, n_species, {inx_large, inx_large, inx_large});

    // Reconstruct
    host_buffer<double> q(nf * 27 * q_inx3 + padding);
    host_buffer<double> AM(NDIM * q_inx3 + padding);
    reconstruct_impl(exec, omega, nf, angmom_index, smooth_field, disc_detect, q, combined_x,
        combined_u, AM, dx, disc, n_species, ndir, nangmom, {INX, INX, INX});

    // Flux
    const int blocks = NDIM * (q_inx3 / 128 + 1) * 128;
    const host_buffer<bool>& masks = get_flux_host_masks<host_buffer<bool>>();
    host_buffer<double> amax(blocks * (1 + 2 * nf));
    host_buffer<int> amax_indices(blocks);
    host_buffer<int> amax_d(blocks);
    flux_impl(exec, q, combined_x, f, amax, amax_indices, amax_d, masks, omega, dx, A_, B_, nf,
       fgamma, de_switch_1, blocks, 1);

    sync_kokkos_host_kernel(exec);

    // Find Maximum
    size_t current_max_slot = 0;
    for (size_t dim_i = 1; dim_i < blocks; dim_i++) {
        if (amax[dim_i] > amax[current_max_slot]) {
            current_max_slot = dim_i;
        }
    }

    // Create & Return timestep_t type
    std::vector<double> URs(nf), ULs(nf);
    const size_t current_max_index = amax_indices[current_max_slot];
    const size_t current_d = amax_d[current_max_slot];
    timestep_t ts;
    ts.a = amax[current_max_slot];
    ts.x = combined_x[current_max_index];
    ts.y = combined_x[current_max_index + q_inx3];
    ts.z = combined_x[current_max_index + 2 * q_inx3];
    const size_t current_i = current_max_slot;
    const auto current_dim = current_max_slot / (blocks / NDIM);
    const auto flipped_dim = flip_dim(current_d, current_dim);
    constexpr int compressedH_DN[3] = {q_inx2, q_inx, 1};
    for (int f = 0; f < nf; f++) {
        URs[f] = amax[blocks + current_i * 2 * nf + f];
        ULs[f] = amax[blocks + current_i * 2 * nf + nf + f];
    }
    ts.ul = std::move(ULs);
    ts.ur = std::move(URs);
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

    // Host buffers
    host_buffer<double> combined_x(NDIM * q_inx3 + padding);
    host_buffer<double> combined_large_x(NDIM * H_N3 + padding);
    host_buffer<double> combined_u(hydro.get_nf() * H_N3 + padding);
    host_buffer<int> disc_detect(hydro.get_nf());
    host_buffer<int> smooth_field(hydro.get_nf());
    host_buffer<double> f(NDIM * hydro.get_nf() * q_inx3 + padding);

    // Convert input
    convert_x_structure(X, combined_x.data());
    for (int n = 0; n < NDIM; n++) {
        std::copy(X[n].begin(), X[n].end(), combined_large_x.data() + n * H_N3);
    }
    for (int f = 0; f < hydro.get_nf(); f++) {
        std::copy(U[f].begin(), U[f].end(), combined_u.data() + f * H_N3);
    }
    const auto& disc_detect_bool = hydro.get_disc_detect();
    const auto& smooth_bool = hydro.get_smooth_field();
    for (auto f = 0; f < hydro.get_nf(); f++) {
        disc_detect[f] = disc_detect_bool[f];
        smooth_field[f] = smooth_bool[f];
    }

    // Either handles the launches on the CPU or on the GPU depending on the passed executor
    auto max_lambda = device_interface_kokkos_hydro(executor, combined_x, combined_large_x,
        combined_u, disc_detect, smooth_field, f, geo.NDIR, hydro.get_nf(),
        hydro.get_angmom_index() != -1, n_species, omega, hydro.get_angmom_index(), geo.NANGMOM,
        X[0][geo.H_DNX] - X[0][0], physics<NDIM>::A_, physics<NDIM>::B_, physics<NDIM>::fgamma_,
        physics<NDIM>::de_switch_1);

    // Convert output
    for (int dim = 0; dim < NDIM; dim++) {
        for (integer field = 0; field != opts().n_fields; ++field) {
            const auto dim_offset = dim * opts().n_fields * q_inx3 + field * q_inx3;
            for (integer i = 0; i <= INX; ++i) {
                for (integer j = 0; j <= INX; ++j) {
                    for (integer k = 0; k <= INX; ++k) {
                        const auto i0 = findex(i, j, k);
                        const auto input_index = (i + 1) * q_inx * q_inx + (j + 1) * q_inx + (k + 1);
                        F[dim][field][i0] = f[dim_offset + input_index];
                    }
                }
            }
        }
    }
    // std::cout << max_lambda.a << std::endl;
    return max_lambda;
}
#endif
