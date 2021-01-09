
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/monopole_interactions/kernel/monopole_kernel_templates.hpp"
#include "octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/monopole_interaction_interface.hpp"

#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/common_kernel/kokkos_util.hpp"

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        // --------------------------------------- Stencil interface

        template <typename storage>
        const storage& get_host_masks() {
            static storage stencil_masks(FULL_STENCIL_SIZE);
            static bool initialized = false;
            if (!initialized) {
                auto superimposed_stencil = monopole_interactions::calculate_stencil().first;
                for (auto i = 0; i < FULL_STENCIL_SIZE; i++) {
                    stencil_masks[i] = false;
                }
                for (auto stencil_element : superimposed_stencil) {
                    const int x = stencil_element.x + STENCIL_MAX;
                    const int y = stencil_element.y + STENCIL_MAX;
                    const int z = stencil_element.z + STENCIL_MAX;
                    size_t index = x * STENCIL_INX * STENCIL_INX + y * STENCIL_INX + z;
                    stencil_masks[index] = true;
                }
                initialized = true;
            }
            return stencil_masks;
        }

        template <typename storage, typename storage_host, typename executor_t>
        const storage& get_device_masks(executor_t& exec) {
            static storage stencil_masks(FULL_STENCIL_SIZE);
            static bool initialized = false;
            if (!initialized) {
                const storage_host& tmp = get_host_masks<storage_host>();
                Kokkos::deep_copy(exec.instance(), stencil_masks, tmp);
                exec.instance().fence();
                initialized = true;
            }
            return stencil_masks;
        }

        // --------------------------------------- P2P Kernel implementations

        template <typename simd_t, typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_mask_t>
        void p2p_kernel_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
            const kokkos_buffer_t& monopoles, const kokkos_mask_t& devicemasks,
            kokkos_buffer_t& potential_expansions, const double dx, const double theta) {

            auto policy_1 = Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>>(
                    executor.instance(), {0, 0, 0}, {INX, INX, INX / simd_t::size()}),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);


            // Kokkos::parallel_for("kernel p2p", policy_1,
            //   [monopoles, potential_expansions, devicemasks, dx, theta] CUDA_GLOBAL_METHOD(
            //       int idx, int idy, int idz) {
            Kokkos::parallel_for(
                "kernel p2p", policy_1, KOKKOS_LAMBDA(int idx, int idy, int idz) {
                    // helper variables
                    constexpr size_t simd_length = simd_t::size();
                    const size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
                    const multiindex<> cell_index(idx + INNER_CELLS_PADDING_DEPTH,
                        idy + INNER_CELLS_PADDING_DEPTH, idz * simd_length + INNER_CELLS_PADDING_DEPTH);
                    const size_t cell_flat_index = to_flat_index_padded(cell_index);
                    multiindex<> cell_index_unpadded(idx, idy, idz * simd_length );
                    const size_t cell_flat_index_unpadded =
                        to_inner_flat_index_not_padded(cell_index_unpadded);

                    const int32_t cell_index_coarse_x = ((cell_index.x + INX) >> 1) - (INX / 2);
                    const int32_t cell_index_coarse_y = ((cell_index.y + INX) >> 1) - (INX / 2);
                    int32_t cell_index_coarse_z[simd_length];
                    for (int i = 0; i < simd_length; i++) {
                        cell_index_coarse_z[i] = ((cell_index.z + i + INX) >> 1) - (INX / 2);
                    }

                    const simd_t theta_rec_squared((1.0 / theta) * (1.0 / theta));
                    simd_t theta_c_rec_squared;
                    double theta_c_rec_squared_array[simd_length];

                    const double d_components[2] = {1.0 / dx, -1.0 / dx};
                    simd_t tmpstore[4] = {simd_t(0.0), simd_t(0.0), simd_t(0.0), simd_t(0.0)};
                    multiindex<> partner_index;
                    // Go through all possible stance elements for the two cells this thread
                    // is responsible for
                    for (int stencil_x = STENCIL_MIN; stencil_x <= STENCIL_MAX; stencil_x++) {
                        partner_index.x = cell_index.x + stencil_x;
                        const int32_t partner_index_coarse_x = ((partner_index.x + INX) >> 1) - (INX / 2);
                        const int32_t distance_x = (cell_index_coarse_x - partner_index_coarse_x) * 
                          (cell_index_coarse_x - partner_index_coarse_x);
                        const int x = stencil_x - STENCIL_MIN;
                        for (int stencil_y = STENCIL_MIN; stencil_y <= STENCIL_MAX; stencil_y++) {
                            partner_index.y = cell_index.y + stencil_y;
                            const int32_t partner_index_coarse_y = ((partner_index.y + INX) >> 1) - (INX / 2);
                            const int32_t distance_y = (cell_index_coarse_y - partner_index_coarse_y) * 
                              (cell_index_coarse_y - partner_index_coarse_y);
                            const int y = stencil_y - STENCIL_MIN;
                            for (int stencil_z = STENCIL_MIN; stencil_z <= STENCIL_MAX;
                                 stencil_z++) {
                                const size_t index = x * STENCIL_INX * STENCIL_INX +
                                    y * STENCIL_INX + (stencil_z - STENCIL_MIN);
                                // Skip stuff that is too far away
                                if (!devicemasks[index]) {
                                    continue;
                                }
                                partner_index.z = cell_index.z + stencil_z;
                                for (int i = 0; i < simd_length; i++) {
                                  const int32_t partner_index_coarse_z = ((partner_index.z + i + INX) >> 1) - (INX / 2);
                                  theta_c_rec_squared_array[i] = static_cast<double>(distance_x + distance_y +
                                      (cell_index_coarse_z[i] - partner_index_coarse_z) * 
                                      (cell_index_coarse_z[i] - partner_index_coarse_z));
                                }
                                theta_c_rec_squared.copy_from(theta_c_rec_squared_array, SIMD_NAMESPACE::element_aligned_tag{}); 

                                const auto mask = theta_c_rec_squared < theta_rec_squared;
                                if (!SIMD_NAMESPACE::any_of(mask)) {
                                    continue;
                                }

                                const size_t partner_flat_index =
                                    to_flat_index_padded(partner_index);
                                simd_t monopole(monopoles.data() + partner_flat_index, SIMD_NAMESPACE::element_aligned_tag{});
                                monopole = SIMD_NAMESPACE::choose(mask, monopole, simd_t(0.0));
                                monopole = monopole * d_components[0];

                                const double r =
                                    std::sqrt(static_cast<double>(stencil_x * stencil_x +
                                        stencil_y * stencil_y + stencil_z * stencil_z));
                                const double r3 = r * r * r;
                                const double four[4] = {
                                    -1.0 / r, stencil_x / r3, stencil_y / r3, stencil_z / r3};
                                tmpstore[0] += four[0] * monopole;
                                tmpstore[1] += four[1] * monopole * d_components[1];
                                tmpstore[2] += four[2] * monopole * d_components[1];
                                tmpstore[3] += four[3] * monopole * d_components[1];

                            }
                        }
                    }
                    tmpstore[0].copy_to(potential_expansions.data() + cell_flat_index_unpadded,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    tmpstore[1].copy_to(potential_expansions.data() + 1 * component_length_unpadded + cell_flat_index_unpadded,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    tmpstore[2].copy_to(potential_expansions.data() + 2 * component_length_unpadded + cell_flat_index_unpadded,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    tmpstore[3].copy_to(potential_expansions.data() + 3 * component_length_unpadded + cell_flat_index_unpadded,
                        SIMD_NAMESPACE::element_aligned_tag{});
                });
        }
        // --------------------------------------- P2M Kernel implementations

        template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_mask_t>
        void p2m_kernel_impl_non_rho(hpx::kokkos::executor<kokkos_backend_t>& executor,
            const kokkos_buffer_t& expansions_neighbors_soa,
            const kokkos_buffer_t& center_of_mass_neighbor_soa,
            const kokkos_buffer_t& center_of_mass_cells_soa, kokkos_buffer_t& potential_expansions,
            const multiindex<> neighbor_size, const multiindex<> start_index,
            const multiindex<> end_index, const multiindex<> dir, const double theta,
            const multiindex<> cells_start, const multiindex<> cells_end,
            const kokkos_mask_t& devicemasks) {
            auto policy_1 = Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>>(
                    executor.instance(), {0, 0, 0},
                    {cells_end.x - cells_start.x, cells_end.y - cells_start.y,
                        cells_end.z - cells_start.z}),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);

            // Kokkos::parallel_for("kernel p2p", policy_1,
            //   [monopoles, potential_expansions, devicemasks, dx, theta] CUDA_GLOBAL_METHOD(
            //       int idx, int idy, int idz) {
            Kokkos::parallel_for(
                "kernel p2m non-rho", policy_1, KOKKOS_LAMBDA(int idx, int idy, int idz) {
                    const int component_length_neighbor =
                        neighbor_size.x * neighbor_size.y * neighbor_size.z + SOA_PADDING;
                    const size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
                    // Set cell indices
                    const multiindex<> cell_index(idx + INNER_CELLS_PADDING_DEPTH + cells_start.x,
                        idy + INNER_CELLS_PADDING_DEPTH + cells_start.y,
                        idz + INNER_CELLS_PADDING_DEPTH + cells_start.z);
                    multiindex<> cell_index_coarse(cell_index);
                    cell_index_coarse.transform_coarse();
                    const size_t cell_flat_index = to_flat_index_padded(cell_index);
                    multiindex<> cell_index_unpadded(
                        idx + cells_start.x, idy + cells_start.y, idz + cells_start.z);
                    const size_t cell_flat_index_unpadded =
                        to_inner_flat_index_not_padded(cell_index_unpadded);

                    const double theta_rec_squared = sqr(1.0 / theta);

                    double X[NDIM];
                    X[0] = center_of_mass_cells_soa[cell_flat_index_unpadded];
                    X[1] = center_of_mass_cells_soa[1 * component_length_unpadded +
                        cell_flat_index_unpadded];
                    X[2] = center_of_mass_cells_soa[2 * component_length_unpadded +
                        cell_flat_index_unpadded];

                    // Create and set result arrays
                    double tmpstore[4];
                    for (size_t i = 0; i < 4; ++i)
                        tmpstore[i] = 0.0;
                    double m_partner[20];
                    double Y[NDIM];
                    for (size_t x = start_index.x; x < end_index.x; x++) {
                        for (size_t y = start_index.y; y < end_index.y; y++) {
                            for (size_t z = start_index.z; z < end_index.z; z++) {
                                // Global index (regarding inner cells + all neighbors)
                                // Used to figure out which stencil mask to use
                                const multiindex<> interaction_partner_index(
                                    INNER_CELLS_PADDING_DEPTH + dir.x * INNER_CELLS_PADDING_DEPTH +
                                        x,
                                    INNER_CELLS_PADDING_DEPTH + dir.y * INNER_CELLS_PADDING_DEPTH +
                                        y,
                                    INNER_CELLS_PADDING_DEPTH + dir.z * INNER_CELLS_PADDING_DEPTH +
                                        z);

                                // Get stencil mask and skip if necessary
                                multiindex<> stencil_element(
                                    interaction_partner_index.x - cell_index.x - STENCIL_MIN,
                                    interaction_partner_index.y - cell_index.y - STENCIL_MIN,
                                    interaction_partner_index.z - cell_index.z - STENCIL_MIN);
                                const size_t stencil_flat_index =
                                    stencil_element.x * STENCIL_INX * STENCIL_INX +
                                    stencil_element.y * STENCIL_INX + stencil_element.z;
                                if (!devicemasks[stencil_flat_index])
                                    continue;

                                multiindex<> partner_index_coarse(interaction_partner_index);
                                partner_index_coarse.transform_coarse();
                                const double theta_c_rec_squared =
                                    static_cast<double>(distance_squared_reciprocal(
                                        cell_index_coarse, partner_index_coarse));
                                const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                                double mask = mask_b ? 1.0 : 0.0;
                                if (!mask_b)
                                    continue;

                                // Local index
                                // Used to figure out which data element to use
                                const multiindex<> interaction_partner_data_index(
                                    x - start_index.x, y - start_index.y, z - start_index.z);
                                const size_t interaction_partner_flat_index =
                                    interaction_partner_data_index.x *
                                        (neighbor_size.y * neighbor_size.z) +
                                    interaction_partner_data_index.y * neighbor_size.z +
                                    interaction_partner_data_index.z;

                                // Load data of interaction partner
                                Y[0] = center_of_mass_neighbor_soa[interaction_partner_flat_index];
                                Y[1] = center_of_mass_neighbor_soa[1 * component_length_neighbor +
                                    interaction_partner_flat_index];
                                Y[2] = center_of_mass_neighbor_soa[2 * component_length_neighbor +
                                    interaction_partner_flat_index];
                                for (size_t i = 0; i < 20; ++i)
                                    m_partner[i] =
                                        expansions_neighbors_soa[i * component_length_neighbor +
                                            interaction_partner_flat_index] *
                                        mask;

                                // run templated interaction method instanced with double type
                                monopole_interactions::compute_kernel_p2m_non_rho(X, Y, m_partner,
                                    tmpstore, [](const double& one, const double& two) -> double {
                                        return max(one, two);
                                    });
                            }
                        }
                    }
                    // Store results in output arrays
                    for (size_t i = 0; i < 4; ++i)
                        potential_expansions[i * component_length_unpadded +
                            cell_flat_index_unpadded] += tmpstore[i];
                });
        }

        template <typename kokkos_backend_t, typename kokkos_buffer_t, typename kokkos_mask_t>
        void p2m_kernel_impl_rho(hpx::kokkos::executor<kokkos_backend_t>& executor,
            const kokkos_buffer_t& expansions_neighbors_soa,
            const kokkos_buffer_t& center_of_mass_neighbor_soa,
            const kokkos_buffer_t& center_of_mass_cells_soa, kokkos_buffer_t& potential_expansions,
            kokkos_buffer_t& angular_corrections, const multiindex<> neighbor_size,
            const multiindex<> start_index, const multiindex<> end_index, const multiindex<> dir,
            const double theta, const multiindex<> cells_start, const multiindex<> cells_end,
            const kokkos_mask_t& devicemasks, const bool reset_ang_corrs) {
            // TODO(daissgr) Is there no memset async equivalent in Kokkos? That would be better
            if (reset_ang_corrs) {
                auto policy_reset = Kokkos::Experimental::require(
                    Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>>(
                        executor.instance(), {0, 0, 0}, {INX, INX, INX}),
                    Kokkos::Experimental::WorkItemProperty::HintLightWeight);
                Kokkos::parallel_for(
                    "kernel p2m corrections reset", policy_reset,
                    KOKKOS_LAMBDA(int idx, int idy, int idz) {
                        const size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;

                        multiindex<> cell_index_unpadded(idx, idy, idz);
                        const size_t cell_flat_index_unpadded =
                            to_inner_flat_index_not_padded(cell_index_unpadded);
                        angular_corrections[cell_flat_index_unpadded] = 0.0;
                        angular_corrections[1 * component_length_unpadded +
                            cell_flat_index_unpadded] = 0.0;
                        angular_corrections[2 * component_length_unpadded +
                            cell_flat_index_unpadded] = 0.0;
                    });
            }
            auto policy_1 = Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<3>>(
                    executor.instance(), {0, 0, 0},
                    {cells_end.x - cells_start.x, cells_end.y - cells_start.y,
                        cells_end.z - cells_start.z}),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);

            Kokkos::parallel_for(
                "kernel p2m rho", policy_1, KOKKOS_LAMBDA(int idx, int idy, int idz) {
                    const int component_length_neighbor =
                        neighbor_size.x * neighbor_size.y * neighbor_size.z + SOA_PADDING;
                    const size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
                    // Set cell indices
                    const multiindex<> cell_index(idx + INNER_CELLS_PADDING_DEPTH + cells_start.x,
                        idy + INNER_CELLS_PADDING_DEPTH + cells_start.y,
                        idz + INNER_CELLS_PADDING_DEPTH + cells_start.z);
                    multiindex<> cell_index_coarse(cell_index);
                    cell_index_coarse.transform_coarse();
                    const size_t cell_flat_index = to_flat_index_padded(cell_index);
                    multiindex<> cell_index_unpadded(
                        idx + cells_start.x, idy + cells_start.y, idz + cells_start.z);
                    const size_t cell_flat_index_unpadded =
                        to_inner_flat_index_not_padded(cell_index_unpadded);

                    const double theta_rec_squared = sqr(1.0 / theta);

                    double X[NDIM];
                    X[0] = center_of_mass_cells_soa[cell_flat_index_unpadded];
                    X[1] = center_of_mass_cells_soa[1 * component_length_unpadded +
                        cell_flat_index_unpadded];
                    X[2] = center_of_mass_cells_soa[2 * component_length_unpadded +
                        cell_flat_index_unpadded];

                    // Create and set result arrays
                    double tmpstore[4];
                    for (size_t i = 0; i < 4; ++i)
                        tmpstore[i] = 0.0;
                    double tmp_corrections[3];
                    for (size_t i = 0; i < 3; ++i)
                        tmp_corrections[i] = 0.0;
                    double m_partner[20];
                    double Y[NDIM];
                    for (size_t x = start_index.x; x < end_index.x; x++) {
                        for (size_t y = start_index.y; y < end_index.y; y++) {
                            for (size_t z = start_index.z; z < end_index.z; z++) {
                                // Global index (regarding inner cells + all neighbors)
                                // Used to figure out which stencil mask to use
                                const multiindex<> interaction_partner_index(
                                    INNER_CELLS_PADDING_DEPTH + dir.x * INNER_CELLS_PADDING_DEPTH +
                                        x,
                                    INNER_CELLS_PADDING_DEPTH + dir.y * INNER_CELLS_PADDING_DEPTH +
                                        y,
                                    INNER_CELLS_PADDING_DEPTH + dir.z * INNER_CELLS_PADDING_DEPTH +
                                        z);

                                // Get stencil mask and skip if necessary
                                multiindex<> stencil_element(
                                    interaction_partner_index.x - cell_index.x - STENCIL_MIN,
                                    interaction_partner_index.y - cell_index.y - STENCIL_MIN,
                                    interaction_partner_index.z - cell_index.z - STENCIL_MIN);
                                const size_t stencil_flat_index =
                                    stencil_element.x * STENCIL_INX * STENCIL_INX +
                                    stencil_element.y * STENCIL_INX + stencil_element.z;
                                if (!devicemasks[stencil_flat_index])
                                    continue;

                                multiindex<> partner_index_coarse(interaction_partner_index);
                                partner_index_coarse.transform_coarse();
                                const double theta_c_rec_squared =
                                    static_cast<double>(distance_squared_reciprocal(
                                        cell_index_coarse, partner_index_coarse));
                                const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                                double mask = mask_b ? 1.0 : 0.0;
                                if (!mask_b)
                                    continue;

                                // Local index
                                // Used to figure out which data element to use
                                const multiindex<> interaction_partner_data_index(
                                    x - start_index.x, y - start_index.y, z - start_index.z);
                                const size_t interaction_partner_flat_index =
                                    interaction_partner_data_index.x *
                                        (neighbor_size.y * neighbor_size.z) +
                                    interaction_partner_data_index.y * neighbor_size.z +
                                    interaction_partner_data_index.z;

                                // Load data of interaction partner
                                Y[0] = center_of_mass_neighbor_soa[interaction_partner_flat_index];
                                Y[1] = center_of_mass_neighbor_soa[1 * component_length_neighbor +
                                    interaction_partner_flat_index];
                                Y[2] = center_of_mass_neighbor_soa[2 * component_length_neighbor +
                                    interaction_partner_flat_index];
                                for (size_t i = 0; i < 20; ++i)
                                    m_partner[i] =
                                        expansions_neighbors_soa[i * component_length_neighbor +
                                            interaction_partner_flat_index] *
                                        mask;

                                // run templated interaction method instanced with double type
                                monopole_interactions::compute_kernel_p2m_rho(X, Y, m_partner,
                                    tmpstore, tmp_corrections,
                                    [](const double& one, const double& two) -> double {
                                        return max(one, two);
                                    });
                            }
                        }
                    }
                    // Store results in output arrays
                    for (size_t i = 0; i < 4; ++i)
                        potential_expansions[i * component_length_unpadded +
                            cell_flat_index_unpadded] += tmpstore[i];

                    angular_corrections[cell_flat_index_unpadded] += tmp_corrections[0];
                    angular_corrections[1 * component_length_unpadded + cell_flat_index_unpadded] +=
                        tmp_corrections[1];
                    angular_corrections[2 * component_length_unpadded + cell_flat_index_unpadded] +=
                        tmp_corrections[2];
                });
        }

        // --------------------------------------- P2P Launch Interface implementations

        template <typename executor_t,
            std::enable_if_t<is_kokkos_device_executor<executor_t>::value, int> = 0>
        void launch_interface_p2p(executor_t& exec, host_buffer<double>& monopoles,
            host_buffer<double>& results, double dx, double theta) {
            // create device buffers
            const device_buffer<int>& device_masks =
                get_device_masks<device_buffer<int>, host_buffer<int>, executor_t>(exec);
            device_buffer<double> device_monopoles(NUMBER_LOCAL_MONOPOLE_VALUES);
            device_buffer<double> device_results(NUMBER_POT_EXPANSIONS_SMALL);

            // move device buffers
            Kokkos::deep_copy(exec.instance(), device_monopoles, monopoles);

            // call kernel
            p2p_kernel_impl<device_simd_t>(exec, device_monopoles, device_masks, device_results, dx, theta);

            auto fut = hpx::kokkos::deep_copy_async(exec.instance(), results, device_results);
            fut.get();
        }
        template <typename executor_t,
            std::enable_if_t<is_kokkos_host_executor<executor_t>::value, int> = 0>
        void launch_interface_p2p(executor_t& exec, host_buffer<double>& monopoles,
            host_buffer<double>& results, double dx, double theta) {
            const host_buffer<int>& host_masks = get_host_masks<host_buffer<int>>();
            // call kernel
            p2p_kernel_impl<host_simd_t>(exec, monopoles, host_masks, results, dx, theta);

            sync_kokkos_host_kernel(exec);
        }

        // --------------------------------------- P2P / P2M Launch Interface implementations

        template <typename executor_t,
            std::enable_if_t<is_kokkos_device_executor<executor_t>::value, int> = 0>
        void launch_interface_p2p_p2m(executor_t& exec, host_buffer<double>& monopoles,
            host_buffer<double>& results, host_buffer<double>& ang_corr_results,
            host_buffer<double>& center_of_masses_inner_cells,
            std::vector<host_buffer<double>>& local_expansions,
            std::vector<host_buffer<double>>& center_of_masses, double dx, double theta,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            const size_t number_p2m_kernels) {
            // create device buffers
            const device_buffer<int>& device_masks =
                get_device_masks<device_buffer<int>, host_buffer<int>, executor_t>(exec);
            device_buffer<double> device_monopoles(NUMBER_LOCAL_MONOPOLE_VALUES);
            device_buffer<double> device_results(NUMBER_POT_EXPANSIONS_SMALL);

            // move device buffers
            Kokkos::deep_copy(exec.instance(), device_monopoles, monopoles);

            // call p2p kernel
            p2p_kernel_impl<device_simd_t>(exec, device_monopoles, device_masks, device_results, dx, theta);

            device_buffer<double> device_center_of_masses_inner_cells(
                (INNER_CELLS + SOA_PADDING) * 3);
            Kokkos::deep_copy(
                exec.instance(), device_center_of_masses_inner_cells, center_of_masses_inner_cells);

            std::vector<device_buffer<double>> device_center_of_masses_neighbors;
            std::vector<device_buffer<double>> device_local_expansions_neighbors;

            device_buffer<double> device_corrections(NUMBER_ANG_CORRECTIONS);

            // - Launch Kernel
            size_t counter_kernel = 0;
            bool reset_ang_corrs = true;
            for (const geo::direction& dir : geo::direction::full_set()) {
                neighbor_gravity_type& neighbor = neighbors[dir];
                if (!neighbor.is_monopole && neighbor.data.M) {
                    int size = 1;
                    for (int i = 0; i < 3; i++) {
                        if (dir[i] == 0)
                            size *= INX;
                        else
                            size *= STENCIL_MAX;
                    }
                    // Indices to address the interaction and stencil data
                    multiindex<> start_index = get_padding_start_indices(dir);
                    multiindex<> end_index = get_padding_end_indices(dir);
                    multiindex<> neighbor_size = get_padding_real_size(dir);
                    multiindex<> dir_index;
                    dir_index.x = dir[0];
                    dir_index.y = dir[1];
                    dir_index.z = dir[2];
                    // Save Computation time by only considering cells that actually can change
                    // These are their start and stop indices which are used for the later kernel
                    // launch
                    multiindex<> cells_start(0, 0, 0);
                    multiindex<> cells_end(INX, INX, INX);
                    if (dir[0] == 1)
                        cells_start.x = INX - (STENCIL_MAX + 1);
                    if (dir[0] == -1)
                        cells_end.x = (STENCIL_MAX + 1);
                    if (dir[1] == 1)
                        cells_start.y = INX - (STENCIL_MAX + 1);
                    if (dir[1] == -1)
                        cells_end.y = (STENCIL_MAX + 1);
                    if (dir[2] == 1)
                        cells_start.z = INX - (STENCIL_MAX + 1);
                    if (dir[2] == -1)
                        cells_end.z = (STENCIL_MAX + 1);

                    device_local_expansions_neighbors.emplace_back((size + SOA_PADDING) * 20);
                    Kokkos::deep_copy(exec.instance(),
                        device_local_expansions_neighbors[counter_kernel],
                        local_expansions[counter_kernel]);
                    device_center_of_masses_neighbors.emplace_back((size + SOA_PADDING) * 3);
                    Kokkos::deep_copy(exec.instance(),
                        device_center_of_masses_neighbors[counter_kernel],
                        center_of_masses[counter_kernel]);

                    if (type == RHO) {
                        p2m_kernel_impl_rho(exec, device_local_expansions_neighbors[counter_kernel],
                            device_center_of_masses_neighbors[counter_kernel],
                            device_center_of_masses_inner_cells, device_results, device_corrections,
                            neighbor_size, start_index, end_index, dir_index, theta, cells_start,
                            cells_end, device_masks, reset_ang_corrs);
                        // only reset angular correction result buffer for the first run
                        reset_ang_corrs = false;
                    } else {
                        p2m_kernel_impl_non_rho(exec,
                            device_local_expansions_neighbors[counter_kernel],
                            device_center_of_masses_neighbors[counter_kernel],
                            device_center_of_masses_inner_cells, device_results, neighbor_size,
                            start_index, end_index, dir_index, theta, cells_start, cells_end,
                            device_masks);
                    }
                    counter_kernel++;
                }
            }
            if (type == RHO)
                Kokkos::deep_copy(exec.instance(), ang_corr_results, device_corrections);

            auto fut = hpx::kokkos::deep_copy_async(exec.instance(), results, device_results);
            fut.get();
        }

        template <typename executor_t,
            std::enable_if_t<is_kokkos_host_executor<executor_t>::value, int> = 0>
        void launch_interface_p2p_p2m(executor_t& exec, host_buffer<double>& monopoles,
            host_buffer<double>& results, host_buffer<double>& ang_corr_results,
            host_buffer<double>& center_of_masses_inner_cells,
            std::vector<host_buffer<double>>& local_expansions,
            std::vector<host_buffer<double>>& center_of_masses, double dx, double theta,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            const size_t number_p2m_kernels) {
            const host_buffer<int>& host_masks = get_host_masks<host_buffer<int>>();

            // call p2p kernel
            p2p_kernel_impl<host_simd_t>(exec, monopoles, host_masks, results, dx, theta);

            // - Launch Kernel
            size_t counter_kernel = 0;
            bool reset_ang_corrs = true;
            for (const geo::direction& dir : geo::direction::full_set()) {
                neighbor_gravity_type& neighbor = neighbors[dir];
                if (!neighbor.is_monopole && neighbor.data.M) {
                    int size = 1;
                    for (int i = 0; i < 3; i++) {
                        if (dir[i] == 0)
                            size *= INX;
                        else
                            size *= STENCIL_MAX;
                    }
                    // Indices to address the interaction and stencil data
                    multiindex<> start_index = get_padding_start_indices(dir);
                    multiindex<> end_index = get_padding_end_indices(dir);
                    multiindex<> neighbor_size = get_padding_real_size(dir);
                    multiindex<> dir_index;
                    dir_index.x = dir[0];
                    dir_index.y = dir[1];
                    dir_index.z = dir[2];
                    // Save Computation time by only considering cells that actually can change
                    // These are their start and stop indices which are used for the later kernel
                    // launch
                    multiindex<> cells_start(0, 0, 0);
                    multiindex<> cells_end(INX, INX, INX);
                    if (dir[0] == 1)
                        cells_start.x = INX - (STENCIL_MAX + 1);
                    if (dir[0] == -1)
                        cells_end.x = (STENCIL_MAX + 1);
                    if (dir[1] == 1)
                        cells_start.y = INX - (STENCIL_MAX + 1);
                    if (dir[1] == -1)
                        cells_end.y = (STENCIL_MAX + 1);
                    if (dir[2] == 1)
                        cells_start.z = INX - (STENCIL_MAX + 1);
                    if (dir[2] == -1)
                        cells_end.z = (STENCIL_MAX + 1);

                    if (type == RHO) {
                        p2m_kernel_impl_rho(exec, local_expansions[counter_kernel],
                            center_of_masses[counter_kernel], center_of_masses_inner_cells, results,
                            ang_corr_results, neighbor_size, start_index, end_index, dir_index,
                            theta, cells_start, cells_end, host_masks, reset_ang_corrs);
                        // only reset angular correction result buffer for the first run
                        reset_ang_corrs = false;
                    } else {
                        p2m_kernel_impl_non_rho(exec, local_expansions[counter_kernel],
                            center_of_masses[counter_kernel], center_of_masses_inner_cells, results,
                            neighbor_size, start_index, end_index, dir_index, theta, cells_start,
                            cells_end, host_masks);
                    }
                    counter_kernel++;
                }
            }
            sync_kokkos_host_kernel(exec);
        }

        // --------------------------------------- Kernel interface

        template <typename executor_t>
        void monopole_kernel(executor_t& exec, std::vector<real>& monopoles,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx, real theta,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::shared_ptr<grid> grid_ptr, const bool contains_multipole_neighbor) {
            // Create host buffers
            host_buffer<double> host_monopoles(NUMBER_LOCAL_MONOPOLE_VALUES);
            host_buffer<double> host_results(NUMBER_POT_EXPANSIONS_SMALL);
            // Fill input buffers with converted (AoS->SoA) data
            monopole_interactions::update_input(
                monopoles, neighbors, type, host_monopoles, grid_ptr);

            if (contains_multipole_neighbor) {
                // Get center of masses inner cells
                std::vector<space_vector> const& com0 = *(com_ptr[0]);
                host_buffer<double> host_center_of_masses_inner_cells(
                    (INNER_CELLS + SOA_PADDING) * 3);
                iterate_inner_cells_padded(
                    [&host_center_of_masses_inner_cells, com0](const multiindex<>& i,
                        const size_t flat_index, const multiindex<>& i_unpadded,
                        const size_t flat_index_unpadded) {
                        monopole_interactions::set_AoS_value<INNER_CELLS + SOA_PADDING, 3>(
                            host_center_of_masses_inner_cells,
                            std::move(com0.at(flat_index_unpadded)), flat_index_unpadded);
                    });

                std::vector<host_buffer<double>> host_center_of_masses;
                std::vector<host_buffer<double>> host_local_expansions;
                host_buffer<double> host_corrections(NUMBER_ANG_CORRECTIONS);

                size_t number_kernels = 0;
                for (const geo::direction& dir : geo::direction::full_set()) {
                    neighbor_gravity_type& neighbor = neighbors[dir];
                    if (!neighbor.is_monopole && neighbor.data.M) {
                        int size = 1;
                        for (int i = 0; i < 3; i++) {
                            if (dir[i] == 0)
                                size *= INX;
                            else
                                size *= STENCIL_MAX;
                        }
                        host_local_expansions.emplace_back((size + SOA_PADDING) * 20);
                        host_center_of_masses.emplace_back((size + SOA_PADDING) * 3);
                        monopole_interactions::update_neighbor_input(dir, com_ptr, neighbors, type,
                            host_local_expansions[number_kernels],
                            host_center_of_masses[number_kernels], grid_ptr, size + SOA_PADDING);
                        number_kernels++;
                    }
                }
                launch_interface_p2p_p2m(exec, host_monopoles, host_results, host_corrections,
                    host_center_of_masses_inner_cells, host_local_expansions, host_center_of_masses,
                    dx, theta, neighbors, type, number_kernels);
                if (type == RHO) {
                    std::vector<space_vector>& corrections = grid_ptr->get_L_c();
                    for (size_t component = 0; component < 3; component++) {
                        for (size_t entry = 0; entry < INNER_CELLS; entry++) {
                            corrections[entry][component] =
                                host_corrections[component * (INNER_CELLS + SOA_PADDING) + entry];
                        }
                    }
                }
            } else {
                launch_interface_p2p(exec, host_monopoles, host_results, dx, theta);
            }

            // Add results back into non-SoA array
            std::vector<expansion>& org = grid_ptr->get_L();
            for (size_t component = 0; component < 4; component++) {
                for (size_t entry = 0; entry < INNER_CELLS; entry++) {
                    org[entry][component] +=
                        host_results[component * (INNER_CELLS + SOA_PADDING) + entry];
                }
            }
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
