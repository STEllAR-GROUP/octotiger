
#include "octotiger/common_kernel/interaction_constants.hpp"

#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/common_kernel/kokkos_util.hpp"

template <typename T>
using host_buffer = recycled_pinned_view<T>;
template <typename T>
using device_buffer = recycled_device_view<T>;
#endif

// --------------------------------------- Kernel implementations

template <typename executor_t, typename buffer_t, typename mask_t>    
void p2p_kernel_impl(executor_t& exec, buffer_t& deviceView,
    mask_t& deviceMasks, buffer_t& deviceResultView, double dx,
    double theta) {
    static_assert(always_false<executor_t>::value, "P2P Kernel not implemented for this kind of executor!");
}

template <typename kokkos_backend_t,
    typename kokkos_buffer_t, typename kokkos_mask_t>    
void p2p_kernel_impl(
    hpx::kokkos::executor<kokkos_backend_t>& executor, kokkos_buffer_t& deviceView,
    kokkos_mask_t& devicemasks, kokkos_buffer_t& deviceResultView, double dx,
    double theta) {
    using namespace octotiger::fmm;

    Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>> policy_1(executor.instance(), {0, 0, 0}, {8, 8, 8});
    
    Kokkos::parallel_for("kernel p2p", policy_1,
        [deviceView, deviceResultView, devicemasks, dx, theta] CUDA_GLOBAL_METHOD(
            int idx, int idy, int idz) {
            // helper variables
            const size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
            const octotiger::fmm::multiindex<> cell_index(idx + INNER_CELLS_PADDING_DEPTH,
                idy + INNER_CELLS_PADDING_DEPTH, idz + INNER_CELLS_PADDING_DEPTH);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(idx, idy, idz);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);
            const double theta_rec_squared = (1.0 / theta) * (1.0 / theta);
            const double d_components[2] = {1.0 / dx, -1.0 / dx};
            double tmpstore[4] = {0.0, 0.0, 0.0, 0.0};

            // Go through all possible stance elements for the two cells this thread
            // is responsible for
            for (int stencil_x = STENCIL_MIN; stencil_x <= STENCIL_MAX; stencil_x++) {
                int x = stencil_x - STENCIL_MIN;
                for (int stencil_y = STENCIL_MIN; stencil_y <= STENCIL_MAX; stencil_y++) {
                    int y = stencil_y - STENCIL_MIN;
                    for (int stencil_z = STENCIL_MIN; stencil_z <= STENCIL_MAX; stencil_z++) {
                        const size_t index = x * STENCIL_INX * STENCIL_INX + y * STENCIL_INX +
                            (stencil_z - STENCIL_MIN);
                        // Skip stuff that is too far away
                        if (!devicemasks[index]) {
                            continue;
                        }
                        // Interaction helpers
                        const multiindex<> partner_index1(cell_index.x + stencil_x,
                            cell_index.y + stencil_y, cell_index.z + stencil_z);
                        const size_t partner_flat_index1 = to_flat_index_padded(partner_index1);
                        multiindex<> partner_index_coarse1(partner_index1);
                        partner_index_coarse1.transform_coarse();
                        const double theta_c_rec_squared = static_cast<double>(
                            distance_squared_reciprocal(cell_index_coarse, partner_index_coarse1));
                        const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                        const double mask = mask_b ? 1.0 : 0.0;

                        // Interaction calculation
                        const double r = std::sqrt(static_cast<double>(
                            stencil_x * stencil_x + stencil_y * stencil_y + stencil_z * stencil_z));
                        const double r3 = r * r * r;
                        const double four[4] = {
                            -1.0 / r, stencil_x / r3, stencil_y / r3, stencil_z / r3};
                        const double monopole =
                            deviceView[partner_flat_index1] * mask * d_components[0];
                        // Calculate the actual interactions
                        tmpstore[0] = tmpstore[0] + four[0] * monopole;
                        tmpstore[1] = tmpstore[1] + four[1] * monopole * d_components[1];
                        tmpstore[2] = tmpstore[2] + four[2] * monopole * d_components[1];
                        tmpstore[3] = tmpstore[3] + four[3] * monopole * d_components[1];
                    }
                }
            }
            deviceResultView[cell_flat_index_unpadded] = tmpstore[0];
            deviceResultView[1 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[1];
            deviceResultView[2 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[2];
            deviceResultView[3 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[3];
        });
}

// --------------------------------------- Launch Interface implementations

template <typename executor_t>
void launch_interface(executor_t& exec, host_buffer<double>& monopoles,
    host_buffer<int>& masks, host_buffer<double>& results, double dx, double theta) {
    static_assert(always_false<executor_t>::value, "P2P launch interface implemented for this kind of executor!");
}

template <typename kokkos_backend_t>
void launch_interface(
    hpx::kokkos::executor<kokkos_backend_t>& exec, host_buffer<double>& monopoles,
    host_buffer<int>& masks, host_buffer<double>& results, double dx, double theta) {

    // create device buffers
    device_buffer<int> device_masks(octotiger::fmm::FULL_STENCIL_SIZE);
    device_buffer<double> device_monopoles(octotiger::fmm::NUMBER_LOCAL_MONOPOLE_VALUES);
    device_buffer<double> device_results(octotiger::fmm::NUMBER_POT_EXPANSIONS_SMALL);

    // move device buffers
    Kokkos::deep_copy(exec.instance(), device_monopoles, monopoles);
    Kokkos::deep_copy(exec.instance(), device_masks, masks);

    // call kernel
    p2p_kernel_impl(exec, device_monopoles, device_masks, device_results, dx, theta);

    auto fut = hpx::kokkos::deep_copy_async(exec.instance(), results, device_results);
    fut.get();
}
template <>
void launch_interface(
    hpx::kokkos::executor<Kokkos::Experimental::HPX>& exec, host_buffer<double>& monopoles,
    host_buffer<int>& masks, host_buffer<double>& results, double dx, double theta) {
    // call kernel
    p2p_kernel_impl(exec, monopoles, masks, results, dx, theta);

    auto fut = exec.instance().impl_get_future();
    fut.get();
}



// --------------------------------------- Kernel interface

template <typename executor_t>
void p2p_kernel(executor_t& exec, std::vector<real>& monopoles,
    std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
    std::array<bool, geo::direction::count()>& is_direction_empty, std::shared_ptr<grid> grid_ptr) {
    // Create host buffers
    host_buffer<int> host_masks(octotiger::fmm::FULL_STENCIL_SIZE);
    host_buffer<double> host_monopoles(octotiger::fmm::NUMBER_LOCAL_MONOPOLE_VALUES);
    host_buffer<double> host_results(octotiger::fmm::NUMBER_POT_EXPANSIONS_SMALL);
    std::vector<bool> neighbor_empty_monopoles(27);    // TODO(daissgr) Get rid of this one
    // Fill input buffers with converted (AoS->SoA) data
    octotiger::fmm::monopole_interactions::update_input(
        monopoles, neighbors, type, host_monopoles, neighbor_empty_monopoles, grid_ptr);

    launch_interface(exec, host_monopoles, host_masks, host_results, dx, dx);
}
