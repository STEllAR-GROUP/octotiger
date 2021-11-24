
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp"
#include "octotiger/multipole_interactions/util/calculate_stencil.hpp"

#include <math.h>

#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/common_kernel/kokkos_util.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        // --------------------------------------- Stencil interface

        template <typename storage>
        const storage& get_host_masks(bool indicators) {
            static storage stencil_masks(FULL_STENCIL_SIZE);
            static storage stencil_indicators(FULL_STENCIL_SIZE);
            static bool initialized = false;
            if (!initialized) {
                auto superimposed_stencil = multipole_interactions::calculate_stencil();
                for (auto i = 0; i < FULL_STENCIL_SIZE; i++) {
                    stencil_masks[i] = false;
                    stencil_indicators[i] = true;
                }
                auto inner_index = 0;
                for (auto stencil_element : superimposed_stencil.stencil_elements) {
                    const int x = stencil_element.x + STENCIL_MAX;
                    const int y = stencil_element.y + STENCIL_MAX;
                    const int z = stencil_element.z + STENCIL_MAX;
                    size_t index = x * STENCIL_INX * STENCIL_INX + y * STENCIL_INX + z;
                    stencil_masks[index] = true;
                    if (!superimposed_stencil.stencil_phase_indicator[inner_index])
                        stencil_indicators[index] = false;
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
            static storage stencil_masks(FULL_STENCIL_SIZE);
            static storage stencil_indicators(FULL_STENCIL_SIZE);
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
        //
        template <typename kokkos_backend_t, typename kokkos_buffer_t>
        void sum_potential_expansions(hpx::kokkos::executor<kokkos_backend_t>& executor,
            const kokkos_buffer_t& tmp_potential_expansions, kokkos_buffer_t& potential_expansions,
            const long number_blocks, const Kokkos::Array<long, 4>&& tiling_config) {
            auto policy_sum = Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<4>>(
                    executor.instance(), {0, 0, 0, 0}, {20, INX, INX, INX}, tiling_config),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);
            Kokkos::parallel_for(
                "kernel sum potential expansions", policy_sum,
                KOKKOS_LAMBDA(int component, int idx, int idy, int idz) {
                    constexpr size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
                    octotiger::fmm::multiindex<> cell_index_unpadded(idx, idy, idz);
                    const size_t cell_flat_index_unpadded =
                        octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);
                    double tmpstore = 0.0;
                    for (int i = 0; i < number_blocks; i++) {
                        tmpstore = tmpstore +
                            tmp_potential_expansions[i * NUMBER_POT_EXPANSIONS +
                                component * component_length_unpadded + cell_flat_index_unpadded];
                    }
                    potential_expansions[component * component_length_unpadded +
                        cell_flat_index_unpadded] = tmpstore;
                });
        }
        template <typename kokkos_backend_t, typename kokkos_buffer_t>
        void sum_angular_corrections(hpx::kokkos::executor<kokkos_backend_t>& executor,
            kokkos_buffer_t& tmp_angular_corrections, kokkos_buffer_t& angular_corrections,
            const long number_blocks, const Kokkos::Array<long, 4>&& tiling_config) {
            auto policy_sum = Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<4>>(
                    executor.instance(), {0, 0, 0, 0}, {3, INX, INX, INX}, tiling_config),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);
            Kokkos::parallel_for(
                "kernel sum angular corrections", policy_sum,
                KOKKOS_LAMBDA(int component, int idx, int idy, int idz) {
                    constexpr size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
                    octotiger::fmm::multiindex<> cell_index_unpadded(idx, idy, idz);
                    const size_t cell_flat_index_unpadded =
                        octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);
                    double tmp_corrections = 0.0;
                    for (int i = 0; i < number_blocks; i++) {
                        tmp_corrections = tmp_corrections +
                            tmp_angular_corrections[i * NUMBER_ANG_CORRECTIONS +
                                component * component_length_unpadded + cell_flat_index_unpadded];
                    }
                    angular_corrections[component * component_length_unpadded +
                        cell_flat_index_unpadded] = tmp_corrections;
                });
        }

        template <typename simd_t, typename simd_mask_t, typename kokkos_backend_t,
            typename kokkos_buffer_t, typename kokkos_mask_t>
        void multipole_kernel_rho_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
            const kokkos_buffer_t& monopoles, const kokkos_buffer_t& centers_of_mass,
            const kokkos_buffer_t& multipoles, kokkos_buffer_t& potential_expansions,
            kokkos_buffer_t& angular_corrections, const double theta, const kokkos_mask_t& masks,
            const kokkos_mask_t& indicators, const long number_blocks_dim,
            const Kokkos::Array<long, 5>&& tiling_config) {
            //{1, INX / 2, INX / simd_t::size()}
            auto policy_1 = Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<5>>(
                    executor.instance(), {0, 0, 0, 0, 0},
                    {number_blocks_dim, number_blocks_dim, INX, INX, INX / simd_t::size()},
                    tiling_config),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);

            Kokkos::parallel_for(
                "kernel multipole rho", policy_1,
                KOKKOS_LAMBDA(int block_id_x, int block_id_y, int idx, int idy, int idz) {
                    constexpr size_t simd_length = simd_t::size();
                    constexpr size_t component_length = ENTRIES + SOA_PADDING;
                    constexpr size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
                    const size_t block_id = block_id_x * number_blocks_dim + block_id_y;

                    // Set cell indices
                    const multiindex<> cell_index(idx + INNER_CELLS_PADDING_DEPTH,
                        idy + INNER_CELLS_PADDING_DEPTH,
                        idz * simd_length + INNER_CELLS_PADDING_DEPTH);
                    const size_t cell_flat_index = to_flat_index_padded(cell_index);
                    multiindex<> cell_index_unpadded(idx, idy, idz * simd_length);
                    const size_t cell_flat_index_unpadded =
                        to_inner_flat_index_not_padded(cell_index_unpadded);

                    const int32_t cell_index_coarse_x = ((cell_index.x + INX) >> 1) - (INX / 2);
                    const int32_t cell_index_coarse_y = ((cell_index.y + INX) >> 1) - (INX / 2);
                    int32_t cell_index_coarse_z[simd_length];
                    for (int i = 0; i < simd_length; i++) {
                        cell_index_coarse_z[i] = ((cell_index.z + i + INX) >> 1) - (INX / 2);
                    }

                    // Load multipoles for this cell
                    simd_t m_cell[20];
                    for (int i = 0; i < 20; i++) {
                        m_cell[i].copy_from(
                            multipoles.data() + i * component_length + cell_flat_index,
                            SIMD_NAMESPACE::element_aligned_tag{});
                    }
                    simd_t X[NDIM];
                    X[0].copy_from(centers_of_mass.data() + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    X[1].copy_from(centers_of_mass.data() + 1 * component_length + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    X[2].copy_from(centers_of_mass.data() + 2 * component_length + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});

                    // Create and set result arrays
                    simd_t tmpstore[20];
                    simd_t tmp_corrections[3];
                    for (size_t i = 0; i < 20; ++i)
                        tmpstore[i] = simd_t(0.0);
                    for (size_t i = 0; i < 3; ++i)
                        tmp_corrections[i] = simd_t(0.0);
                    simd_t m_partner[20];
                    simd_t Y[NDIM];

                    // Required for mask
                    const simd_t theta_rec_squared((1.0 / theta) * (1.0 / theta));
                    simd_t theta_c_rec_squared;
                    double theta_c_rec_squared_array[simd_length];

                    multiindex<> partner_index;
                    // calculate interactions between this cell and each stencil element
                    for (int x_iteration = 0;
                         x_iteration <= NUMBER_MULTIPOLE_BLOCKS - number_blocks_dim;
                         x_iteration++) {
                        const int x = block_id_x + x_iteration;
                        const int stencil_x = x + STENCIL_MIN;
                        partner_index.x = cell_index.x + stencil_x;

                        const int32_t partner_index_coarse_x =
                            ((partner_index.x + INX) >> 1) - (INX / 2);
                        const int32_t distance_x = (cell_index_coarse_x - partner_index_coarse_x) *
                            (cell_index_coarse_x - partner_index_coarse_x);

                        for (int y_iteration = 0;
                             y_iteration <= NUMBER_MULTIPOLE_BLOCKS - number_blocks_dim;
                             y_iteration++) {
                            const int y = block_id_y + y_iteration;
                            const int stencil_y = y + STENCIL_MIN;
                            partner_index.y = cell_index.y + stencil_y;

                            const int32_t partner_index_coarse_y =
                                ((partner_index.y + INX) >> 1) - (INX / 2);
                            const int32_t distance_y =
                                (cell_index_coarse_y - partner_index_coarse_y) *
                                (cell_index_coarse_y - partner_index_coarse_y);
                            for (int stencil_z = STENCIL_MIN; stencil_z <= STENCIL_MAX;
                                 stencil_z++) {
                                const size_t index = x * STENCIL_INX * STENCIL_INX +
                                    y * STENCIL_INX + (stencil_z - STENCIL_MIN);
                                if (!masks[index]) {
                                    continue;
                                }

                                partner_index.z = cell_index.z + stencil_z;
                                for (int i = 0; i < simd_length; i++) {
                                    const int32_t partner_index_coarse_z =
                                        ((partner_index.z + i + INX) >> 1) - (INX / 2);
                                    theta_c_rec_squared_array[i] =
                                        static_cast<double>(distance_x + distance_y +
                                            (cell_index_coarse_z[i] - partner_index_coarse_z) *
                                                (cell_index_coarse_z[i] - partner_index_coarse_z));
                                }
                                theta_c_rec_squared.copy_from(theta_c_rec_squared_array,
                                    SIMD_NAMESPACE::element_aligned_tag{});

                                simd_mask_t mask = theta_c_rec_squared < theta_rec_squared;
                                if (!SIMD_NAMESPACE::any_of(mask)) {
                                    continue;
                                }
                                const double mask_phase_one = indicators[index];
                                const size_t partner_flat_index =
                                    to_flat_index_padded(partner_index);

                                // Load data of interaction partner
                                Y[0].copy_from(centers_of_mass.data() + partner_flat_index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                                Y[1].copy_from(centers_of_mass.data() + 1 * component_length +
                                        partner_flat_index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                                Y[2].copy_from(centers_of_mass.data() + 2 * component_length +
                                        partner_flat_index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                                m_partner[0].copy_from(monopoles.data() + partner_flat_index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                                m_partner[0] =
                                    SIMD_NAMESPACE::choose(mask, m_partner[0], simd_t(0.0));

                                // Checks whether we are currently handling monopoles (false) or
                                // multipole (true) as interaction partners
                                if (!indicators[index]) {
                                    mask = simd_mask_t(false);
                                }
                                m_partner[0] += SIMD_NAMESPACE::choose(mask,
                                    simd_t(multipoles.data() + partner_flat_index,
                                        SIMD_NAMESPACE::element_aligned_tag{}),
                                    simd_t(0.0));
                                for (size_t i = 1; i < 20; ++i) {
                                    m_partner[i].copy_from(multipoles.data() +
                                            i * component_length + partner_flat_index,
                                        SIMD_NAMESPACE::element_aligned_tag{});
                                    m_partner[i] =
                                        SIMD_NAMESPACE::choose(mask, m_partner[i], simd_t(0.0));
                                }

                                // Do the actual calculations
                                multipole_interactions::compute_kernel_rho(X, Y, m_partner,
                                    tmpstore, tmp_corrections, m_cell,
                                    [](const simd_t& one, const simd_t& two) -> simd_t {
                                        return SIMD_NAMESPACE::max(one, two);
                                    });
                            }
                        }
                    }

                    // Store results in output arrays
                    for (size_t i = 0; i < 20; ++i) {
                        tmpstore[i].copy_to(potential_expansions.data() +
                                block_id * NUMBER_POT_EXPANSIONS + i * component_length_unpadded +
                                cell_flat_index_unpadded,
                            SIMD_NAMESPACE::element_aligned_tag{});
                    }
                    tmp_corrections[0].copy_to(angular_corrections.data() +
                            block_id * NUMBER_ANG_CORRECTIONS + cell_flat_index_unpadded,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    tmp_corrections[1].copy_to(angular_corrections.data() +
                            1 * component_length_unpadded + block_id * NUMBER_ANG_CORRECTIONS +
                            cell_flat_index_unpadded,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    tmp_corrections[2].copy_to(angular_corrections.data() +
                            2 * component_length_unpadded + block_id * NUMBER_ANG_CORRECTIONS +
                            cell_flat_index_unpadded,
                        SIMD_NAMESPACE::element_aligned_tag{});
                });
        }
        // --------------------------------------- Kernel root rho implementations

        template <typename simd_t, typename simd_mask_t, typename kokkos_backend_t,
            typename kokkos_buffer_t, typename kokkos_mask_t>
        void multipole_kernel_root_rho_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
            const kokkos_buffer_t& centers_of_mass, const kokkos_buffer_t& multipoles,
            kokkos_buffer_t& potential_expansions, kokkos_buffer_t& angular_corrections,
            const kokkos_mask_t& indicators, const long number_blocks,
            const Kokkos::Array<long, 4>&& tiling_config) {
            auto policy_1 = Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<4>>(
                    executor.instance(), {0, 0, 0, 0},
                    {number_blocks, INX, INX, INX / simd_t::size()}, tiling_config),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);

            Kokkos::parallel_for(
                "kernel multipole root rho", policy_1,
                KOKKOS_LAMBDA(int block_id, int idx, int idy, int idz) {
                    constexpr size_t simd_length = simd_t::size();
                    constexpr size_t component_length = ENTRIES + SOA_PADDING;
                    constexpr size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;

                    // Set cell indices
                    const multiindex<> cell_index(idx + INNER_CELLS_PADDING_DEPTH,
                        idy + INNER_CELLS_PADDING_DEPTH,
                        idz * simd_length + INNER_CELLS_PADDING_DEPTH);
                    const size_t cell_flat_index = to_flat_index_padded(cell_index);
                    multiindex<> cell_index_unpadded(idx, idy, idz * simd_length);
                    const size_t cell_flat_index_unpadded =
                        to_inner_flat_index_not_padded(cell_index_unpadded);

                    // Load multipoles for this cell
                    simd_t m_cell[20];
                    for (int i = 0; i < 20; i++) {
                        m_cell[i].copy_from(
                            multipoles.data() + i * component_length + cell_flat_index,
                            SIMD_NAMESPACE::element_aligned_tag{});
                    }
                    simd_t X[NDIM];
                    X[0].copy_from(centers_of_mass.data() + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    X[1].copy_from(centers_of_mass.data() + 1 * component_length + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    X[2].copy_from(centers_of_mass.data() + 2 * component_length + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});

                    // Create and set result arrays
                    simd_t tmpstore[20];
                    simd_t tmp_corrections[3];
                    for (size_t i = 0; i < 20; ++i)
                        tmpstore[i] = simd_t(0.0);
                    for (size_t i = 0; i < 3; ++i)
                        tmp_corrections[i] = simd_t(0.0);

                    simd_t m_partner[20];
                    simd_t Y[NDIM];
                    for (int x_offset = 0; x_offset <= INX - number_blocks; x_offset++) {
                        const int x = block_id + x_offset;
                        const int stencil_x = x - cell_index_unpadded.x;
                        for (int y = 0; y < INX; y++) {
                            const int stencil_y = y - cell_index_unpadded.y;
                            for (int z = 0; z < INX; z++) {
                                const int stencil_z = z - cell_index_unpadded.z;
                                double mask_helper2_array[simd_length];
                                for (int i = 0; i < simd_length; i++) {
                                    mask_helper2_array[i] = 0.0;
                                }
                                const multiindex<> stencil_element(stencil_x, stencil_y, stencil_z);
                                if (stencil_x >= STENCIL_MIN && stencil_x <= STENCIL_MAX &&
                                    stencil_y >= STENCIL_MIN && stencil_y <= STENCIL_MAX) {
                                    for (int i = 0;
                                         i < simd_length && stencil_z - STENCIL_MIN - i >= 0; i++) {
                                        const size_t index =
                                            (stencil_x - STENCIL_MIN) * STENCIL_INX * STENCIL_INX +
                                            (stencil_y - STENCIL_MIN) * STENCIL_INX +
                                            (stencil_z - STENCIL_MIN - i);
                                        if (stencil_z - i <= STENCIL_MAX) {
                                            if (!indicators[index] ||
                                                (stencil_x == 0 && stencil_y == 0 &&
                                                    stencil_z - i == 0)) {
                                                mask_helper2_array[i] = 12.0;
                                            }
                                        }
                                    }
                                }
                                // Workaround to set the mask - usually I'd like to set it
                                // component-wise but kokkos-simd currently does not support this!
                                // hence the mask_helpers
                                const simd_t mask_helper1(1.0);
                                const simd_t mask_helper2(
                                    mask_helper2_array, SIMD_NAMESPACE::element_aligned_tag{});
                                simd_mask_t mask = mask_helper2 < mask_helper1;
                                if (!SIMD_NAMESPACE::any_of(mask)) {
                                    continue;
                                }

                                const multiindex<> partner_index(x + INX, y + INX, z + INX);
                                const size_t partner_flat_index =
                                    to_flat_index_padded(partner_index);

                                // Load data of interaction partner!
                                // NOTE: We only load ONE partner (the same one) for all SIMD cell
                                // indices (1 to n) This is unlike the non-root variant where we
                                // have one partner for each cell (n to n)
                                Y[0] = SIMD_NAMESPACE::choose(
                                    mask, simd_t(centers_of_mass[partner_flat_index]), simd_t(0.0));
                                Y[1] = SIMD_NAMESPACE::choose(mask,
                                    simd_t(
                                        centers_of_mass[partner_flat_index + 1 * component_length]),
                                    simd_t(0.0));
                                Y[2] = SIMD_NAMESPACE::choose(mask,
                                    simd_t(
                                        centers_of_mass[partner_flat_index + 2 * component_length]),
                                    simd_t(0.0));
                                for (size_t i = 0; i < 20; ++i) {
                                    m_partner[i] = SIMD_NAMESPACE::choose(mask,
                                        simd_t(
                                            multipoles[partner_flat_index + i * component_length]),
                                        simd_t(0.0));
                                }

                                // Do the actual calculations
                                multipole_interactions::compute_kernel_rho(X, Y, m_partner,
                                    tmpstore, tmp_corrections, m_cell,
                                    [](const simd_t& one, const simd_t& two) -> simd_t {
                                        return SIMD_NAMESPACE::max(one, two);
                                    });
                            }
                        }
                    }
                    // Store results in output arrays
                    for (size_t i = 0; i < 20; ++i) {
                        tmpstore[i].copy_to(potential_expansions.data() +
                                block_id * NUMBER_POT_EXPANSIONS + i * component_length_unpadded +
                                cell_flat_index_unpadded,
                            SIMD_NAMESPACE::element_aligned_tag{});
                    }
                    tmp_corrections[0].copy_to(angular_corrections.data() +
                            block_id * NUMBER_ANG_CORRECTIONS + cell_flat_index_unpadded,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    tmp_corrections[1].copy_to(angular_corrections.data() +
                            block_id * NUMBER_ANG_CORRECTIONS + 1 * component_length_unpadded +
                            cell_flat_index_unpadded,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    tmp_corrections[2].copy_to(angular_corrections.data() +
                            block_id * NUMBER_ANG_CORRECTIONS + 2 * component_length_unpadded +
                            cell_flat_index_unpadded,
                        SIMD_NAMESPACE::element_aligned_tag{});
                });
        }
        // --------------------------------------- Kernel non rho implementations

        template <typename simd_t, typename simd_mask_t, typename kokkos_backend_t,
            typename kokkos_buffer_t, typename kokkos_mask_t>
        void multipole_kernel_non_rho_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
            const kokkos_buffer_t& monopoles, const kokkos_buffer_t& centers_of_mass,
            const kokkos_buffer_t& multipoles, kokkos_buffer_t& potential_expansions,
            const double theta, const kokkos_mask_t& masks, const kokkos_mask_t& indicators,
            const long number_blocks_dim, const Kokkos::Array<long, 5>&& tiling_config) {
            auto policy_1 = Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<5>>(
                    executor.instance(), {0, 0, 0, 0, 0},
                    {number_blocks_dim, number_blocks_dim, INX, INX, INX / simd_t::size()},
                    tiling_config),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);

            Kokkos::parallel_for(
                "kernel multipole non-rho", policy_1,
                KOKKOS_LAMBDA(int block_id_x, int block_id_y, int idx, int idy, int idz) {
                    constexpr size_t simd_length = simd_t::size();
                    constexpr size_t component_length = ENTRIES + SOA_PADDING;
                    constexpr size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
                    const size_t block_id = block_id_x * number_blocks_dim + block_id_y;

                    // Set cell indices
                    const multiindex<> cell_index(idx + INNER_CELLS_PADDING_DEPTH,
                        idy + INNER_CELLS_PADDING_DEPTH,
                        idz * simd_length + INNER_CELLS_PADDING_DEPTH);
                    const size_t cell_flat_index = to_flat_index_padded(cell_index);
                    multiindex<> cell_index_unpadded(idx, idy, idz * simd_length);
                    const size_t cell_flat_index_unpadded =
                        to_inner_flat_index_not_padded(cell_index_unpadded);

                    const int32_t cell_index_coarse_x = ((cell_index.x + INX) >> 1) - (INX / 2);
                    const int32_t cell_index_coarse_y = ((cell_index.y + INX) >> 1) - (INX / 2);
                    int32_t cell_index_coarse_z[simd_length];
                    for (int i = 0; i < simd_length; i++) {
                        cell_index_coarse_z[i] = ((cell_index.z + i + INX) >> 1) - (INX / 2);
                    }

                    simd_t X[NDIM];
                    X[0].copy_from(centers_of_mass.data() + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    X[1].copy_from(centers_of_mass.data() + 1 * component_length + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    X[2].copy_from(centers_of_mass.data() + 2 * component_length + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});

                    // Create and set result arrays
                    simd_t tmpstore[20];
                    for (size_t i = 0; i < 20; ++i)
                        tmpstore[i] = simd_t(0.0);
                    simd_t m_partner[20];
                    simd_t Y[NDIM];

                    // Required for mask
                    const simd_t theta_rec_squared((1.0 / theta) * (1.0 / theta));
                    simd_t theta_c_rec_squared;
                    double theta_c_rec_squared_array[simd_length];

                    multiindex<> partner_index;
                    // calculate interactions between this cell and each stencil element
                    for (int x_iteration = 0;
                         x_iteration <= NUMBER_MULTIPOLE_BLOCKS - number_blocks_dim;
                         x_iteration++) {
                        const int x = block_id_x + x_iteration;
                        const int stencil_x = x + STENCIL_MIN;
                        partner_index.x = cell_index.x + stencil_x;

                        const int32_t partner_index_coarse_x =
                            ((partner_index.x + INX) >> 1) - (INX / 2);
                        const int32_t distance_x = (cell_index_coarse_x - partner_index_coarse_x) *
                            (cell_index_coarse_x - partner_index_coarse_x);

                        for (int y_iteration = 0;
                             y_iteration <= NUMBER_MULTIPOLE_BLOCKS - number_blocks_dim;
                             y_iteration++) {
                            const int y = block_id_y + y_iteration;
                            const int stencil_y = y + STENCIL_MIN;
                            partner_index.y = cell_index.y + stencil_y;

                            const int32_t partner_index_coarse_y =
                                ((partner_index.y + INX) >> 1) - (INX / 2);
                            const int32_t distance_y =
                                (cell_index_coarse_y - partner_index_coarse_y) *
                                (cell_index_coarse_y - partner_index_coarse_y);
                            for (int stencil_z = STENCIL_MIN; stencil_z <= STENCIL_MAX;
                                 stencil_z++) {
                                const size_t index = x * STENCIL_INX * STENCIL_INX +
                                    y * STENCIL_INX + (stencil_z - STENCIL_MIN);
                                // Skip stuff that is too far away
                                if (!masks[index]) {
                                    continue;
                                }
                                partner_index.z = cell_index.z + stencil_z;
                                for (int i = 0; i < simd_length; i++) {
                                    const int32_t partner_index_coarse_z =
                                        ((partner_index.z + i + INX) >> 1) - (INX / 2);
                                    theta_c_rec_squared_array[i] =
                                        static_cast<double>(distance_x + distance_y +
                                            (cell_index_coarse_z[i] - partner_index_coarse_z) *
                                                (cell_index_coarse_z[i] - partner_index_coarse_z));
                                }
                                theta_c_rec_squared.copy_from(theta_c_rec_squared_array,
                                    SIMD_NAMESPACE::element_aligned_tag{});

                                auto mask = theta_c_rec_squared < theta_rec_squared;
                                if (!SIMD_NAMESPACE::any_of(mask)) {
                                    continue;
                                }
                                const double mask_phase_one = indicators[index];
                                const size_t partner_flat_index =
                                    to_flat_index_padded(partner_index);

                                // Load data of interaction partner
                                Y[0].copy_from(centers_of_mass.data() + partner_flat_index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                                Y[1].copy_from(centers_of_mass.data() + 1 * component_length +
                                        partner_flat_index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                                Y[2].copy_from(centers_of_mass.data() + 2 * component_length +
                                        partner_flat_index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                                m_partner[0].copy_from(monopoles.data() + partner_flat_index,
                                    SIMD_NAMESPACE::element_aligned_tag{});
                                m_partner[0] =
                                    SIMD_NAMESPACE::choose(mask, m_partner[0], simd_t(0.0));

                                // Checks whether we are currently handling monopoles (false) or
                                // multipole (true) as interaction partners
                                if (!indicators[index]) {
                                    mask = simd_mask_t(false);
                                }
                                m_partner[0] += SIMD_NAMESPACE::choose(mask,
                                    simd_t(multipoles.data() + partner_flat_index,
                                        SIMD_NAMESPACE::element_aligned_tag{}),
                                    simd_t(0.0));
                                for (size_t i = 1; i < 20; ++i) {
                                    m_partner[i].copy_from(multipoles.data() +
                                            i * component_length + partner_flat_index,
                                        SIMD_NAMESPACE::element_aligned_tag{});
                                    m_partner[i] =
                                        SIMD_NAMESPACE::choose(mask, m_partner[i], simd_t(0.0));
                                }

                                // Do the actual calculations
                                multipole_interactions::compute_kernel_non_rho(X, Y, m_partner,
                                    tmpstore, [](const simd_t& one, const simd_t& two) -> simd_t {
                                        return SIMD_NAMESPACE::max(one, two);
                                    });
                            }
                        }
                    }
                    // Store results in output arrays
                    for (size_t i = 0; i < 20; ++i) {
                        tmpstore[i].copy_to(potential_expansions.data() +
                                block_id * NUMBER_POT_EXPANSIONS + i * component_length_unpadded +
                                cell_flat_index_unpadded,
                            SIMD_NAMESPACE::element_aligned_tag{});
                    }
                });
        }

        // --------------------------------------- Kernel non rho root implementations

        template <typename simd_t, typename simd_mask_t, typename kokkos_backend_t,
            typename kokkos_buffer_t, typename kokkos_mask_t>
        void multipole_kernel_root_non_rho_impl(hpx::kokkos::executor<kokkos_backend_t>& executor,
            const kokkos_buffer_t& centers_of_mass, const kokkos_buffer_t& multipoles,
            kokkos_buffer_t& potential_expansions, const kokkos_mask_t& indicators,
            const long number_blocks, const Kokkos::Array<long, 4>&& tiling_config) {
            auto policy_1 = Kokkos::Experimental::require(
                Kokkos::MDRangePolicy<decltype(executor.instance()), Kokkos::Rank<4>>(
                    executor.instance(), {0, 0, 0, 0},
                    {number_blocks, INX, INX, INX / simd_t::size()}, tiling_config),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight);

            Kokkos::parallel_for(
                "kernel multipole root non-rho", policy_1,
                KOKKOS_LAMBDA(int block_id, int idx, int idy, int idz) {
                    constexpr size_t simd_length = simd_t::size();
                    constexpr size_t component_length = ENTRIES + SOA_PADDING;
                    constexpr size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;

                    // Set cell indices
                    const multiindex<> cell_index(idx + INNER_CELLS_PADDING_DEPTH,
                        idy + INNER_CELLS_PADDING_DEPTH,
                        idz * simd_length + INNER_CELLS_PADDING_DEPTH);
                    const size_t cell_flat_index = to_flat_index_padded(cell_index);
                    multiindex<> cell_index_unpadded(idx, idy, idz * simd_length);
                    const size_t cell_flat_index_unpadded =
                        to_inner_flat_index_not_padded(cell_index_unpadded);

                    simd_t X[NDIM];
                    X[0].copy_from(centers_of_mass.data() + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    X[1].copy_from(centers_of_mass.data() + 1 * component_length + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    X[2].copy_from(centers_of_mass.data() + 2 * component_length + cell_flat_index,
                        SIMD_NAMESPACE::element_aligned_tag{});

                    // Create and set result arrays
                    simd_t tmpstore[20];
                    for (size_t i = 0; i < 20; ++i)
                        tmpstore[i] = simd_t(0.0);

                    simd_t m_partner[20];
                    simd_t Y[NDIM];

                    // calculate interactions between this cell and each stencil element
                    // for (int x = 0; x < INX; x++) {
                    for (int x_offset = 0; x_offset <= INX - number_blocks; x_offset++) {
                        const int x = block_id + x_offset;
                        const int stencil_x = x - cell_index_unpadded.x;
                        for (int y = 0; y < INX; y++) {
                            const int stencil_y = y - cell_index_unpadded.y;
                            for (int z = 0; z < INX; z++) {
                                const int stencil_z = z - cell_index_unpadded.z;
                                double mask_helper2_array[simd_length];
                                for (int i = 0; i < simd_length; i++) {
                                    mask_helper2_array[i] = 0.0;
                                }
                                const multiindex<> stencil_element(stencil_x, stencil_y, stencil_z);
                                if (stencil_x >= STENCIL_MIN && stencil_x <= STENCIL_MAX &&
                                    stencil_y >= STENCIL_MIN && stencil_y <= STENCIL_MAX) {
                                    for (int i = 0;
                                         i < simd_length && stencil_z - STENCIL_MIN - i >= 0; i++) {
                                        const size_t index =
                                            (stencil_x - STENCIL_MIN) * STENCIL_INX * STENCIL_INX +
                                            (stencil_y - STENCIL_MIN) * STENCIL_INX +
                                            (stencil_z - STENCIL_MIN - i);
                                        if (stencil_z - i <= STENCIL_MAX) {
                                            if (!indicators[index] ||
                                                (stencil_x == 0 && stencil_y == 0 &&
                                                    stencil_z - i == 0)) {
                                                mask_helper2_array[i] = 12.0;
                                            }
                                        }
                                    }
                                }
                                // Workaround to set the mask - usually I'd like to set it
                                // component-wise but kokkos-simd currently does not support this!
                                // hence the mask_helpers
                                const simd_t mask_helper1(1.0);
                                const simd_t mask_helper2(
                                    mask_helper2_array, SIMD_NAMESPACE::element_aligned_tag{});
                                simd_mask_t mask = mask_helper2 < mask_helper1;
                                if (!SIMD_NAMESPACE::any_of(mask)) {
                                    continue;
                                }

                                const multiindex<> partner_index(x + INX, y + INX, z + INX);
                                const size_t partner_flat_index =
                                    to_flat_index_padded(partner_index);

                                // Load data of interaction partner!
                                // NOTE: We only load ONE partner (the same one) for all SIMD cell
                                // indices (1 to n) This is unlike the non-root variant where we
                                // have one partner for each cell (n to n)
                                Y[0] = SIMD_NAMESPACE::choose(
                                    mask, simd_t(centers_of_mass[partner_flat_index]), simd_t(0.0));
                                Y[1] = SIMD_NAMESPACE::choose(mask,
                                    simd_t(
                                        centers_of_mass[partner_flat_index + 1 * component_length]),
                                    simd_t(0.0));
                                Y[2] = SIMD_NAMESPACE::choose(mask,
                                    simd_t(
                                        centers_of_mass[partner_flat_index + 2 * component_length]),
                                    simd_t(0.0));
                                for (size_t i = 0; i < 20; ++i) {
                                    m_partner[i] = SIMD_NAMESPACE::choose(mask,
                                        simd_t(
                                            multipoles[partner_flat_index + i * component_length]),
                                        simd_t(0.0));
                                }

                                // Do the actual calculations
                                multipole_interactions::compute_kernel_non_rho(X, Y, m_partner,
                                    tmpstore, [](const simd_t& one, const simd_t& two) -> simd_t {
                                        return SIMD_NAMESPACE::max(one, two);
                                    });
                            }
                        }
                    }
                    // Store results in output arrays
                    for (size_t i = 0; i < 20; ++i) {
                        tmpstore[i].copy_to(potential_expansions.data() +
                                block_id * NUMBER_POT_EXPANSIONS + i * component_length_unpadded +
                                cell_flat_index_unpadded,
                            SIMD_NAMESPACE::element_aligned_tag{});
                    }
                });
        }

        // --------------------------------------- Launch Interface implementations

        template <typename executor_t,
            std::enable_if_t<is_kokkos_device_executor<executor_t>::value, int> = 0>
        void launch_interface(executor_t& exec, const host_buffer<double>& monopoles,
            const host_buffer<double>& centers_of_mass, const host_buffer<double>& multipoles,
            host_buffer<double>& potential_expansions, host_buffer<double>& angular_corrections,
            const double theta, const gsolve_type type, const bool use_root_stencil) {
            const device_buffer<int>& device_masks =
                get_device_masks<device_buffer<int>, host_buffer<int>, executor_t>(exec, false);
            const device_buffer<int>& device_indicators =
                get_device_masks<device_buffer<int>, host_buffer<int>, executor_t>(exec, true);
            // input buffers
            // std::cout << "device buffer creation" << std::endl;
            device_buffer<double> device_monopoles(NUMBER_LOCAL_MONOPOLE_VALUES);
            if (!use_root_stencil) {
                // std::cout << "device buffer deep copy" << std::endl;
                Kokkos::deep_copy(exec.instance(), device_monopoles, monopoles);
            }
            // std::cout << "device buffer creation" << std::endl;
            device_buffer<double> device_multipoles(NUMBER_LOCAL_EXPANSION_VALUES);
            // std::cout << "device buffer deep copy" << std::endl;
            Kokkos::deep_copy(exec.instance(), device_multipoles, multipoles);
            // std::cout << "device buffer creation" << std::endl;
            device_buffer<double> device_centers(NUMBER_MASS_VALUES);
            // std::cout << "device buffer deep copy" << std::endl;
            Kokkos::deep_copy(exec.instance(), device_centers, centers_of_mass);
            // result buffers
            // std::cout << "device buffer creation" << std::endl;
            device_buffer<double> device_expansions(NUMBER_POT_EXPANSIONS);
            device_buffer<double> device_corrections(NUMBER_ANG_CORRECTIONS);

            int number_blocks = NUMBER_MULTIPOLE_BLOCKS * NUMBER_MULTIPOLE_BLOCKS;
            int number_blocks_dim = NUMBER_MULTIPOLE_BLOCKS;
            if (use_root_stencil) {
                number_blocks = INX;
                number_blocks_dim = INX;
            }

            if (use_root_stencil) {
                // Check kernel input invariant - root kernel does not support more than INX blocks
                assert(number_blocks >= 1 && number_blocks <= INX);
            } else {
                // Check kernel input invariant - normal kernel does not support more than
                // NUMBER_MULTIPOLE_BLOCKS blocks
                assert(number_blocks >= 1 &&
                    number_blocks <= NUMBER_MULTIPOLE_BLOCKS * NUMBER_MULTIPOLE_BLOCKS);
                assert(number_blocks_dim * number_blocks_dim == number_blocks);
            }

            device_buffer<double> tmp_device_corrections(number_blocks * NUMBER_ANG_CORRECTIONS);
            device_buffer<double> tmp_device_expansions(
                number_blocks * NUMBER_POT_EXPANSIONS);
            if (type == RHO) {
                // Launch kernel with angular corrections
                if (!use_root_stencil) {
                    multipole_kernel_rho_impl<device_simd_t, device_simd_mask_t>(exec,
                        device_monopoles, device_centers, device_multipoles, tmp_device_expansions,
                        tmp_device_corrections, theta, device_masks, device_indicators,
                        number_blocks_dim, {1, 1, 1, INX, INX / device_simd_t::size()});
                } else {
                    multipole_kernel_root_rho_impl<device_simd_t, device_simd_mask_t>(exec,
                        device_centers, device_multipoles, tmp_device_expansions,
                        tmp_device_corrections, device_indicators, number_blocks,
                        {1, 1, INX, INX / device_simd_t::size()});
                }
                sum_potential_expansions(exec, tmp_device_expansions, device_expansions,
                    number_blocks, {1, 1, INX, INX});
                sum_angular_corrections(exec, tmp_device_corrections, device_corrections,
                    number_blocks, {1, 1, INX, INX});
                // Copy back angular cocrection results
                // std::cout << "device buffer deep copy" << std::endl;
                Kokkos::deep_copy(exec.instance(), angular_corrections, device_corrections);
            } else {
                // Launch kernel without angular corrections
                if (!use_root_stencil) {
                    multipole_kernel_non_rho_impl<device_simd_t, device_simd_mask_t>(exec,
                        device_monopoles, device_centers, device_multipoles, tmp_device_expansions,
                        theta, device_masks, device_indicators, number_blocks_dim,
                        {1, 1, 1, INX, INX / device_simd_t::size()});
                } else {
                    multipole_kernel_root_non_rho_impl<device_simd_t, device_simd_mask_t>(exec,
                        device_centers, device_multipoles, tmp_device_expansions, device_indicators,
                        number_blocks, {1, 1, INX, INX / device_simd_t::size()});
                }
                sum_potential_expansions(exec, tmp_device_expansions, device_expansions,
                    number_blocks, {1, 1, INX, INX});
            }
            // Copy back potential expansions results and sync
            // std::cout << "device buffer deep copy" << std::endl;
            auto fut = hpx::kokkos::deep_copy_async(
                exec.instance(), potential_expansions, device_expansions);
            // std::cin.get();
            fut.get();
        }

        template <typename executor_t,
            std::enable_if_t<is_kokkos_host_executor<executor_t>::value, int> = 0>
        void launch_interface(executor_t& exec, const host_buffer<double>& monopoles,
            const host_buffer<double>& centers_of_mass, const host_buffer<double>& multipoles,
            host_buffer<double>& potential_expansions, host_buffer<double>& angular_corrections,
            const double theta, const gsolve_type type, const bool use_root_stencil) {
            const host_buffer<int>& host_masks = get_host_masks<host_buffer<int>>(false);
            const host_buffer<int>& host_indicators = get_host_masks<host_buffer<int>>(true);
            if (type == RHO) {
                // Launch kernel with angular corrections
                if (!use_root_stencil) {
                    multipole_kernel_rho_impl<host_simd_t, host_simd_mask_t>(exec, monopoles,
                        centers_of_mass, multipoles, potential_expansions, angular_corrections,
                        theta, host_masks, host_indicators, 1,
                        {1, 1, INX / 2, INX / 2, INX / host_simd_t::size()});
                } else {
                    multipole_kernel_root_rho_impl<host_simd_t, host_simd_mask_t>(exec,
                        centers_of_mass, multipoles, potential_expansions, angular_corrections,
                        host_indicators, 1, {INX / 2, INX / 2, INX / host_simd_t::size()});
                }
            } else {
                // Launch kernel without angular corrections
                if (!use_root_stencil) {
                    multipole_kernel_non_rho_impl<host_simd_t, host_simd_mask_t>(exec, monopoles,
                        centers_of_mass, multipoles, potential_expansions, theta, host_masks,
                        host_indicators, 1, {1, 1, INX / 2, INX / 2, INX / host_simd_t::size()});
                } else {
                    multipole_kernel_root_non_rho_impl<host_simd_t, host_simd_mask_t>(exec,
                        centers_of_mass, multipoles, potential_expansions, host_indicators, 1,
                        {1, INX / 2, INX / 2, INX / host_simd_t::size()});
                }
            }
            sync_kokkos_host_kernel(exec);
        }

        // --------------------------------------- Kernel interface

        template <typename executor_t>
        void multipole_kernel(executor_t& exec, std::vector<real>& monopoles,
            std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx, real theta,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::array<real, NDIM> xbase, std::shared_ptr<grid> grid, const bool use_root_stencil) {
            // input buffers
            host_buffer<double> host_monopoles(NUMBER_LOCAL_MONOPOLE_VALUES);
            host_buffer<double> host_multipoles(NUMBER_LOCAL_EXPANSION_VALUES);
            host_buffer<double> host_masses(NUMBER_MASS_VALUES);
            // result buffers
            host_buffer<double> host_expansions(NUMBER_POT_EXPANSIONS);
            host_buffer<double> host_corrections(NUMBER_ANG_CORRECTIONS);
            // convert input AoS into SoA input buffers
            multipole_interactions::update_input(monopoles, M_ptr, com_ptr, neighbors, type, dx,
                xbase, host_monopoles, host_multipoles, host_masses, grid, use_root_stencil);
            // launch kernel (and copy data to device if necessary)
            launch_interface(exec, host_monopoles, host_masses, host_multipoles, host_expansions,
                host_corrections, theta, type, use_root_stencil);
            // Add results back into non-SoA array
            std::vector<expansion>& org = grid->get_L();
            for (size_t component = 0; component < 20; component++) {
                for (size_t entry = 0; entry < INNER_CELLS; entry++) {
                    org[entry][component] +=
                        host_expansions[component * (INNER_CELLS + SOA_PADDING) + entry];
                }
            }
            // Copy angular corrections back into non-SoA
            if (type == RHO) {
                std::vector<space_vector>& corrections = grid->get_L_c();
                for (size_t component = 0; component < 3; component++) {
                    for (size_t entry = 0; entry < INNER_CELLS; entry++) {
                        corrections[entry][component] =
                            host_corrections[component * (INNER_CELLS + SOA_PADDING) + entry];
                    }
                }
            }
        }

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
