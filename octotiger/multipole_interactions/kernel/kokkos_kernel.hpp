
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/multipole_interactions/util/calculate_stencil.hpp"

#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/common_kernel/kokkos_util.hpp"
#endif

// TODO(daissgr) Change to correct parameter lists

// --------------------------------------- Stencil interface

// TODO(daissgr) Return stencils

// --------------------------------------- Kernel rho implementations

// TODO(daissgr) Implement rho kokkos kernel
template <typename executor_t, typename kokkos_buffer_t, typename kokkos_mask_t>
void multipole_kernel_rho_impl(executor_t& exec, kokkos_buffer_t& monopoles,
    kokkos_buffer_t& centers_of_mass, kokkos_buffer_t& multipoles,
    kokkos_buffer_t& potential_expansions, kokkos_buffer_t& angular_corrections, 
    double theta) {
    static_assert(always_false<executor_t>::value,
        "Multipole Rho Kernel not implemented for this kind of executor!");
}

template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_mask_t>
void multipole_kernel_rho_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
    kokkos_buffer_t& monopoles, kokkos_buffer_t& centers_of_mass, kokkos_buffer_t& multipoles,
    kokkos_buffer_t& potential_expansions, kokkos_buffer_t& angular_corrections, 
    double theta) {
    using namespace octotiger::fmm;

    Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>> policy_1(
        executor.instance(), {0, 0, 0}, {INX, INX, INX});
}
// --------------------------------------- Kernel non rho implementations

template <typename executor_t, typename kokkos_buffer_t, typename kokkos_mask_t>
void multipole_kernel_non_rho_impl(executor_t& exec,
    kokkos_buffer_t& monopoles, kokkos_buffer_t& centers_of_mass, kokkos_buffer_t& multipoles,
    kokkos_buffer_t& potential_expansions, double theta) {
    static_assert(always_false<executor_t>::value,
        "Mutlipole Non-Rho Kernel not implemented for this kind of executor!");
}

// TODO(daissgr) Implement non-rho kokkos kernel
template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_mask_t>
void multipole_kernel_non_rho_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
    kokkos_buffer_t& monopoles, kokkos_buffer_t& centers_of_mass, kokkos_buffer_t& multipoles,
    kokkos_buffer_t& potential_expansions, double theta) {
    using namespace octotiger::fmm;

    Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>> policy_1(
        executor.instance(), {0, 0, 0}, {INX, INX, INX});
}

// --------------------------------------- Launch Interface implementations

template <typename executor_t>
void launch_interface(executor_t& exec, host_buffer<double>& monopoles,
    host_buffer<double>& centers_of_mass, host_buffer<double>& multipoles,
    host_buffer<double>& potential_expansions, host_buffer<double>& angular_corrections,
    double theta) {
    static_assert(always_false<executor_t>::value,
        "Multipole launch interface implemented for this kind of executor!");
}

template <typename kokkos_backend_t>
void launch_interface(hpx::kokkos::executor<kokkos_backend_t>& exec, host_buffer<double>& monopoles,
    host_buffer<double>& centers_of_mass, host_buffer<double>& multipoles,
    host_buffer<double>& potential_expansions, host_buffer<double>& angular_corrections,
    double theta) {
    // TODO(daissgr) Implement generic kokkos launch interface
}
template <>
void launch_interface(hpx::kokkos::executor<Kokkos::Serial>& exec, host_buffer<double>& monopoles,
    host_buffer<double>& centers_of_mass, host_buffer<double>& multipoles,
    host_buffer<double>& potential_expansions, host_buffer<double>& angular_corrections,
    double theta) {
    // TODO(daissgr) Implement serial kokkos launch interface
}
template <>
void launch_interface(hpx::kokkos::executor<Kokkos::Experimental::HPX>& exec,
    host_buffer<double>& monopoles, host_buffer<double>& centers_of_mass,
    host_buffer<double>& multipoles, host_buffer<double>& potential_expansions,
    host_buffer<double>& angular_corrections, double theta) {
    // TODO(daissgr) Implement hpx kokkos launch interface
}

// --------------------------------------- Kernel interface

template <typename executor_t>
void multipole_kernel(executor_t& exec, std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
    std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
    std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
    std::array<bool, geo::direction::count()>& is_direction_empty, std::array<real, NDIM> xbase,
    std::shared_ptr<grid> grid) {
    // TODO (daissgr) Create required host buffers
    // TODO (daissgr) Call launch interface
    // TODO (daissgr) Copy results back
}
