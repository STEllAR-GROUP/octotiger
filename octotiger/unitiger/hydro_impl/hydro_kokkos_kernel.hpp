#pragma once

#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp"
//#include "octotiger/grid.hpp"
//#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"
//#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"

#include "octotiger/common_kernel/kokkos_util.hpp"

template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_int_buffer_t>
void reconstruct_impl(hpx::kokkos::executor<kokkos_backend_t>& executor, const double omega, const int nf_, const int angmom_index_,
    const kokkos_int_buffer_t& smooth_field_, const kokkos_int_buffer_t& disc_detect_,
    kokkos_buffer_t& combined_q, const kokkos_buffer_t& combined_x, kokkos_buffer_t& combined_u,
    kokkos_buffer_t& AM, const double dx, const kokkos_buffer_t& cdiscs, const int n_species_,
    const int ndir, const int nangmom, const Kokkos::Array<long, 3>&& tiling_config) {
    auto policy = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>>(
            executor.instance(), {0, 0, 0}, {16, 8, 8}, tiling_config),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_for(
        "kernel hydro reconstruct", policy, KOKKOS_LAMBDA(int idx, int idy, int idz) {
            const int sx_i = angmom_index_;
            const int zx_i = sx_i + NDIM;
            const int q_i = (idx) *64 + (idy) *8 + (idz);
            const int i = ((q_i / 100) + 2) * 14 * 14 + (((q_i % 100) / 10) + 2) * 14 +
                (((q_i % 100) % 10) + 2);
            if (q_i < 1000) {
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
            executor.instance(), {0, 0, 0}, {14, 14, 14}, tiling_config),
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
            executor.instance(), {0, 0, 0}, {12, 12, 12}, tiling_config_phase1),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_for(
        "kernel find contact discs 1", policy_phase_1, KOKKOS_LAMBDA(int idx, int idy, int idz) {
            cell_find_contact_discs_phase1(P, u, A_, B_, fgamma_, de_switch_1, idx, idy, idz);
        });

    auto policy_phase_2 = Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>>(
            executor.instance(), {0, 0, 0}, {10, 10, 10}, tiling_config_phase2),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_for(
        "kernel find contact discs 2", policy_phase_1, KOKKOS_LAMBDA(int idx, int idy, int idz) {
            cell_find_contact_discs_phase2(disc, P, fgamma_, ndir, idx, idy, idz);
        });
}

template <typename executor_t,
    std::enable_if_t<is_kokkos_device_executor<executor_t>::value, int> = 0>
void device_interface_kokkos_hydro(executor_t& exec, const host_buffer<double>& combined_x,
    const host_buffer<double>& combined_large_x, host_buffer<double>& combined_u,
    const host_buffer<int>& disc_detect, const host_buffer<int>& smooth_field,
    host_buffer<double>& f, const size_t ndir, const size_t nf, const bool angmom,
    const size_t n_species, const double omega, const int angmom_index, const int nangmom, const double dx) {
    // Find contact discs
    device_buffer<double> u(NDIM * nf * 1000 + 32);
    Kokkos::deep_copy(exec.instance(), u, combined_u);
    device_buffer<double> P(H_N3 + 32);
    device_buffer<double> disc(ndir / 2 * H_N3 + 32);
    find_contact_discs_impl(exec, u, P, disc, physics<NDIM>::A_, physics<NDIM>::B_,
        physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1, ndir, {1, 12, 12}, {1, 10, 10});

    // Pre recon
    device_buffer<double> large_x(NDIM * H_N3 + 32);
    Kokkos::deep_copy(exec.instance(), large_x, combined_large_x);
    hydro_pre_recon_impl(exec, large_x, omega, angmom, u, nf, n_species, {1, 14, 14});


    // Reconstruct
    device_buffer<double> x(NDIM * 1000 + 32);
    Kokkos::deep_copy(exec.instance(), x, combined_x);
    device_buffer<int> device_disc_detect(nf);
    Kokkos::deep_copy(exec.instance(), device_disc_detect, disc_detect);
    device_buffer<int> device_smooth_field(nf);
    Kokkos::deep_copy(exec.instance(), device_smooth_field, smooth_field);
    device_buffer<double> q(
        nf * 27 * 10 * 10 * 10 + 32);
    device_buffer<double> AM(NDIM * 10 * 10 * 10 + 32);
    reconstruct_impl(
        exec, omega, nf, angmom_index, device_smooth_field, device_disc_detect, q, x, u,
        AM, dx, disc, n_species, ndir, nangmom, {1, 8, 8});
}

template <typename executor_t,
    std::enable_if_t<is_kokkos_host_executor<executor_t>::value, int> = 0>
void device_interface_kokkos_hydro(executor_t& exec, const host_buffer<double>& combined_x,
    const host_buffer<double>& combined_large_x, host_buffer<double>& combined_u,
    const host_buffer<int>& disc_detect, const host_buffer<int>& smooth_field,
    host_buffer<double>& f, const size_t ndir, const size_t nf, const bool angmom,
    const size_t n_species, const double omega, const int angmom_index, const int nangmom, const double dx) {
    // Find contact discs
    host_buffer<double> P(H_N3 + 32);
    host_buffer<double> disc(ndir / 2 * H_N3 + 32);
    find_contact_discs_impl(exec, combined_u, P, disc, physics<NDIM>::A_, physics<NDIM>::B_,
        physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1, ndir, {6, 12, 12}, {5, 10, 10});

    // Pre recon
    hydro_pre_recon_impl(
        exec, combined_large_x, omega, angmom, combined_u, nf, n_species, {7, 14, 14});

    // Reconstruct
    host_buffer<double> q(
        nf * 27 * 10 * 10 * 10 + 32);
    host_buffer<double> AM(NDIM * 10 * 10 * 10 + 32);
    reconstruct_impl(
        exec, omega, nf, angmom_index, smooth_field, disc_detect, q, combined_x, combined_u,
        AM, dx, disc, n_species, ndir, nangmom, {8, 8, 8});
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
    host_buffer<double> combined_x(NDIM * 1000 + 32);
    host_buffer<double> combined_large_x(NDIM * H_N3 + 32);
    host_buffer<double> combined_u(hydro.get_nf() * H_N3 + 32);
    host_buffer<int> disc_detect(hydro.get_nf());
    host_buffer<int> smooth_field(hydro.get_nf());
    host_buffer<double> f(NDIM * hydro.get_nf() * 1000 + 32);

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
    device_interface_kokkos_hydro(executor, combined_x, combined_large_x, combined_u, disc_detect,
        smooth_field, f, geo.NDIR, hydro.get_nf(), hydro.get_angmom_index() != -1, n_species,
        omega, hydro.get_angmom_index(), geo.NANGMOM, X[0][geo.H_DNX] - X[0][0]);

    return timestep_t{};
}
#endif
