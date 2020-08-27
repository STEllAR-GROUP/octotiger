
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/multipole_interactions/util/calculate_stencil.hpp"

#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/common_kernel/kokkos_util.hpp"
#endif

// --------------------------------------- Stencil interface

template <typename storage>
const storage& get_host_masks(bool indicators) {
    static storage stencil_masks(octotiger::fmm::FULL_STENCIL_SIZE);
    static storage stencil_indicators(octotiger::fmm::FULL_STENCIL_SIZE);
    static bool initialized = false;
    if (!initialized) {
        auto superimposed_stencil = octotiger::fmm::multipole_interactions::calculate_stencil();
        for (auto i = 0; i < octotiger::fmm::FULL_STENCIL_SIZE; i++) {
            stencil_masks[i] = false;
            stencil_indicators[i] = false;
        }
        auto inner_index = 0;
        for (auto stencil_element : superimposed_stencil.stencil_elements) {
            const int x = stencil_element.x + octotiger::fmm::STENCIL_MAX;
            const int y = stencil_element.y + octotiger::fmm::STENCIL_MAX;
            const int z = stencil_element.z + octotiger::fmm::STENCIL_MAX;
            size_t index = x * octotiger::fmm::STENCIL_INX * octotiger::fmm::STENCIL_INX +
                y * octotiger::fmm::STENCIL_INX + z;
            stencil_masks[index] = true;
            if (superimposed_stencil.stencil_phase_indicator[inner_index])
                stencil_indicators[index] = true;
            inner_index++;
        }
        initialized = true;
    }
    if (indicators)
        return stencil_indicators;
    else
        return stencil_masks;
}

template <typename storage, typename storage_host, typename executor_t>
const storage& get_device_masks(executor_t& exec, bool indicators) {
    static storage stencil_masks(octotiger::fmm::FULL_STENCIL_SIZE);
    static storage stencil_indicators(octotiger::fmm::FULL_STENCIL_SIZE);
    static bool initialized = false;
    if (!initialized) {
        const storage_host& tmp_masks = get_host_masks<storage_host>(false);
        Kokkos::deep_copy(exec.instance(), stencil_masks, tmp_masks);
        const storage_host& tmp_indicators = get_host_masks<storage_host>(true);
        Kokkos::deep_copy(exec.instance(), stencil_indicators, tmp_indicators);
        exec.instance().fence();
        initialized = true;
    }
    if (indicators)
        return stencil_indicators;
    else
        return stencil_masks;
}

// --------------------------------------- Kernel rho implementations

// TODO(daissgr) Implement rho kokkos kernel
template <typename executor_t, typename kokkos_buffer_t, typename kokkos_mask_t>
void multipole_kernel_rho_impl(executor_t& exec, kokkos_buffer_t& monopoles,
    kokkos_buffer_t& centers_of_mass, kokkos_buffer_t& multipoles,
    kokkos_buffer_t& potential_expansions, kokkos_buffer_t& angular_corrections, double theta,
    const kokkos_mask_t& masks, const kokkos_mask_t& indicators) {
    static_assert(always_false<executor_t>::value,
        "Multipole Rho Kernel not implemented for this kind of executor!");
}

template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_mask_t>
void multipole_kernel_rho_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
    kokkos_buffer_t& monopoles, kokkos_buffer_t& centers_of_mass, kokkos_buffer_t& multipoles,
    kokkos_buffer_t& potential_expansions, kokkos_buffer_t& angular_corrections, double theta,
    const kokkos_mask_t& masks, const kokkos_mask_t& indicators) {
    using namespace octotiger::fmm;

    Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>> policy_1(
        executor.instance(), {0, 0, 0}, {INX, INX, INX});
}
// --------------------------------------- Kernel non rho implementations

template <typename executor_t, typename kokkos_buffer_t, typename kokkos_mask_t>
void multipole_kernel_non_rho_impl(executor_t& exec, kokkos_buffer_t& monopoles,
    kokkos_buffer_t& centers_of_mass, kokkos_buffer_t& multipoles,
    kokkos_buffer_t& potential_expansions, double theta, const kokkos_mask_t& masks,
    const kokkos_mask_t& indicators) {
    static_assert(always_false<executor_t>::value,
        "Mutlipole Non-Rho Kernel not implemented for this kind of executor!");
}

// TODO(daissgr) Implement non-rho kokkos kernel
template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_mask_t>
void multipole_kernel_non_rho_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
    kokkos_buffer_t& monopoles, kokkos_buffer_t& centers_of_mass, kokkos_buffer_t& multipoles,
    kokkos_buffer_t& potential_expansions, double theta, const kokkos_mask_t& masks,
    const kokkos_mask_t& indicators) {
    using namespace octotiger::fmm;

    Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>> policy_1(
        executor.instance(), {0, 0, 0}, {INX, INX, INX});
}

// --------------------------------------- Launch Interface implementations

template <typename executor_t>
void launch_interface(executor_t& exec, host_buffer<double>& monopoles,
    host_buffer<double>& centers_of_mass, host_buffer<double>& multipoles,
    host_buffer<double>& potential_expansions, host_buffer<double>& angular_corrections,
    double theta, gsolve_type type) {
    static_assert(always_false<executor_t>::value,
        "Multipole launch interface implemented for this kind of executor!");
}

template <typename kokkos_backend_t>
void launch_interface(hpx::kokkos::executor<kokkos_backend_t>& exec, host_buffer<double>& monopoles,
    host_buffer<double>& centers_of_mass, host_buffer<double>& multipoles,
    host_buffer<double>& potential_expansions, host_buffer<double>& angular_corrections,
    double theta, gsolve_type type) {
    const device_buffer<int>& device_masks = get_device_masks<device_buffer<int>, host_buffer<int>,
        hpx::kokkos::executor<kokkos_backend_t>>(exec, false);
    const device_buffer<int>& device_indicators = get_device_masks<device_buffer<int>, host_buffer<int>,
        hpx::kokkos::executor<kokkos_backend_t>>(exec, true);
    // input buffers
    device_buffer<double> device_monopoles(octotiger::fmm::NUMBER_LOCAL_MONOPOLE_VALUES);
    Kokkos::deep_copy(exec.instance(), device_monopoles, monopoles);
    device_buffer<double> device_multipoles(octotiger::fmm::NUMBER_LOCAL_EXPANSION_VALUES);
    Kokkos::deep_copy(exec.instance(), device_multipoles, multipoles);
    device_buffer<double> device_centers(octotiger::fmm::NUMBER_MASS_VALUES);
    Kokkos::deep_copy(exec.instance(), device_centers, centers_of_mass);
    // result buffers
    device_buffer<double> device_expansions(octotiger::fmm::NUMBER_POT_EXPANSIONS);
    device_buffer<double> device_corrections(octotiger::fmm::NUMBER_ANG_CORRECTIONS);
    if (type == RHO) {
        // Launch kernel with angular corrections
        multipole_kernel_rho_impl(exec, device_monopoles, device_centers, device_multipoles,
            device_expansions, device_corrections, theta, device_masks, device_indicators);
        // Copy back angular cocrection results
        Kokkos::deep_copy(exec.instance(), angular_corrections, device_corrections);
    } else {
        // Launch kernel without angular corrections
        multipole_kernel_non_rho_impl(exec, device_monopoles, device_centers, device_multipoles,
            device_expansions, theta, device_masks, device_indicators);
    }
    // Copy back potential expansions results and sync
    auto fut =
        hpx::kokkos::deep_copy_async(exec.instance(), potential_expansions, device_expansions);
    fut.get();
}
template <>
void launch_interface(hpx::kokkos::executor<Kokkos::Serial>& exec, host_buffer<double>& monopoles,
    host_buffer<double>& centers_of_mass, host_buffer<double>& multipoles,
    host_buffer<double>& potential_expansions, host_buffer<double>& angular_corrections,
    double theta, gsolve_type type) {
    const host_buffer<int> &host_masks = get_host_masks<host_buffer<int>>(false);
    const host_buffer<int> &host_indicators = get_host_masks<host_buffer<int>>(true);
    if (type == RHO) {
        // Launch kernel with angular corrections
        multipole_kernel_rho_impl<Kokkos::Serial, host_buffer<double>, host_buffer<int>>(exec,
            monopoles, centers_of_mass, multipoles, potential_expansions, angular_corrections,
            theta, host_masks, host_indicators);
    } else {
        // Launch kernel without angular corrections
        multipole_kernel_non_rho_impl<Kokkos::Serial, host_buffer<double>, host_buffer<int>>(exec,
            monopoles, centers_of_mass, multipoles, potential_expansions, theta, host_masks,
            host_indicators);
    }
    // Sync
    exec.instance().fence();
}
template <>
void launch_interface(hpx::kokkos::executor<Kokkos::Experimental::HPX>& exec,
    host_buffer<double>& monopoles, host_buffer<double>& centers_of_mass,
    host_buffer<double>& multipoles, host_buffer<double>& potential_expansions,
    host_buffer<double>& angular_corrections, double theta, gsolve_type type) {
    const host_buffer<int> &host_masks = get_host_masks<host_buffer<int>>(false);
    const host_buffer<int> &host_indicators = get_host_masks<host_buffer<int>>(true);
    if (type == RHO) {
        // Launch kernel with angular corrections
        multipole_kernel_rho_impl<Kokkos::Experimental::HPX, host_buffer<double>, host_buffer<int>>(
            exec, monopoles, centers_of_mass, multipoles, potential_expansions, angular_corrections,
            theta, host_masks, host_indicators);
    } else {
        // Launch kernel without angular corrections
        multipole_kernel_non_rho_impl<Kokkos::Experimental::HPX, host_buffer<double>,
            host_buffer<int>>(exec, monopoles, centers_of_mass, multipoles, potential_expansions,
            theta, host_masks, host_indicators);
    }
    // Sync
    exec.instance().fence();
}

// --------------------------------------- Kernel interface

template <typename executor_t>
void multipole_kernel(executor_t& exec, std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
    std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
    std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
    std::array<bool, geo::direction::count()>& is_direction_empty, std::array<real, NDIM> xbase,
    std::shared_ptr<grid> grid) {
    // input buffers
    host_buffer<double> host_monopoles(octotiger::fmm::NUMBER_LOCAL_MONOPOLE_VALUES);
    host_buffer<double> host_multipoles(octotiger::fmm::NUMBER_LOCAL_EXPANSION_VALUES);
    host_buffer<double> host_masses(octotiger::fmm::NUMBER_MASS_VALUES);
    // result buffers
    host_buffer<double> host_expansions(octotiger::fmm::NUMBER_POT_EXPANSIONS);
    host_buffer<double> host_corrections(octotiger::fmm::NUMBER_ANG_CORRECTIONS);
    // convert input AoS into SoA input buffers
    octotiger::fmm::multipole_interactions::update_input(monopoles, M_ptr, com_ptr, neighbors, type,
        dx, xbase, host_monopoles, host_multipoles, host_masses, grid);
    // launch kernel (and copy data to device if necessary)
    launch_interface(exec, host_monopoles, host_masses, host_multipoles, host_expansions,
        host_corrections, dx, type);    // TODO(daissgr) this needs theta, not dx
    // TODO (daissgr) Copy results back
}
