#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp"
#include "octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp"

#include <sstream>
#include <cstddef>

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

       __device__ __constant__ bool device_stencil_indicator_const[FULL_STENCIL_SIZE];
       __device__ __constant__ bool device_constant_stencil_masks[FULL_STENCIL_SIZE];
        void copy_stencil_to_m2m_constant_memory(const float *stencil_masks, const size_t full_stencil_size) {
            cudaError_t err = cudaMemcpyToSymbol(device_constant_stencil_masks, stencil_masks, full_stencil_size);
        }
        void copy_indicator_to_m2m_constant_memory(const float *indicator, const size_t indicator_size) {
            cudaError_t err = cudaMemcpyToSymbol(device_stencil_indicator_const, indicator, indicator_size);
        }

        __device__ const size_t component_length = ENTRIES + SOA_PADDING;
        __device__ const size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;

        __global__ void
        __launch_bounds__(INX * INX, 2)
        cuda_multipole_interactions_kernel_rho(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            double (&angular_corrections)[NUMBER_ANG_CORRECTIONS],
            const double theta, const bool computing_second_half) {
            int index_x = threadIdx.x + blockIdx.x;
            if (computing_second_half)
                index_x += 4;
            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(index_x + INNER_CELLS_PADDING_DEPTH,
                threadIdx.y + INNER_CELLS_PADDING_DEPTH, threadIdx.z + INNER_CELLS_PADDING_DEPTH);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(index_x,
                                                             threadIdx.y, threadIdx.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

            // Load multipoles for this cell
            double m_cell[20];
            #pragma unroll
            for (int i = 0; i < 20; i++)
                m_cell[i] = multipoles[i * component_length + cell_flat_index];
            double X[NDIM];
            X[0] = center_of_masses[cell_flat_index];
            X[1] = center_of_masses[1 * component_length + cell_flat_index];
            X[2] = center_of_masses[2 * component_length + cell_flat_index];

            // Create and set result arrays
            double tmpstore[20];
            double tmp_corrections[3];
            #pragma unroll
            for (size_t i = 0; i < 20; ++i)
                tmpstore[i] = 0.0;
            #pragma unroll
            for (size_t i = 0; i < 3; ++i)
                tmp_corrections[i] = 0.0;
            // Required for mask
            const double theta_rec_squared = sqr(1.0 / theta);
            double m_partner[20];
            double Y[NDIM];

            // calculate interactions between this cell and each stencil element
            for (int stencil_x = STENCIL_MIN; stencil_x <= STENCIL_MAX; stencil_x++) {
                int x = stencil_x - STENCIL_MIN;
                for (int stencil_y = STENCIL_MIN; stencil_y <= STENCIL_MAX; stencil_y++) {
                    int y = stencil_y - STENCIL_MIN;
                    for (int stencil_z = STENCIL_MIN; stencil_z <= STENCIL_MAX; stencil_z++) {
                        const size_t index = x * STENCIL_INX * STENCIL_INX +
                            y * STENCIL_INX + (stencil_z - STENCIL_MIN);
                        if (!device_constant_stencil_masks[index]) {
                            continue;
                        }
                        const double mask_phase_one = device_stencil_indicator_const[index];
                        const multiindex<> partner_index(cell_index.x + stencil_x,
                                                          cell_index.y + stencil_y,
                                                          cell_index.z + stencil_z);
                        const size_t partner_flat_index = to_flat_index_padded(partner_index);
                        multiindex<> partner_index_coarse(partner_index);
                        partner_index_coarse.transform_coarse();

                        // Create mask - TODO is this really necessay in the non-vectorized code..?
                        const double theta_c_rec_squared = static_cast<double>(
                            distance_squared_reciprocal(cell_index_coarse, partner_index_coarse));
                        const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                        double mask = mask_b ? 1.0 : 0.0;

                        // Load data of interaction partner
                        Y[0] = center_of_masses[partner_flat_index];
                        Y[1] = center_of_masses[1 * component_length + partner_flat_index];
                        Y[2] = center_of_masses[2 * component_length + partner_flat_index];
                        m_partner[0] = local_monopoles[partner_flat_index] * mask;
                        mask = mask * mask_phase_one;    // do not load multipoles outside the inner stencil
                        m_partner[0] += multipoles[partner_flat_index] * mask;
                        #pragma unroll
                        for (size_t i = 1; i < 20; ++i)
                            m_partner[i] = multipoles[i * component_length + partner_flat_index] * mask;

                        // Do the actual calculations
                        compute_kernel_rho(X, Y, m_partner, tmpstore, tmp_corrections, m_cell,
                            [] __device__(const double& one, const double& two) -> double {
                                return std::max(one, two);
                            });
                    }
                }
            }

            // Store results in output arrays
            #pragma unroll
            for (size_t i = 0; i < 20; ++i)
                potential_expansions[i * component_length_unpadded + cell_flat_index_unpadded] =
                    tmpstore[i];

            angular_corrections[cell_flat_index_unpadded] = tmp_corrections[0];
            angular_corrections[1 * component_length_unpadded + cell_flat_index_unpadded] =
                tmp_corrections[1];
            angular_corrections[2 * component_length_unpadded + cell_flat_index_unpadded] =
                tmp_corrections[2];
        }
        __global__ void
        __launch_bounds__(INX * INX, 2)
        cuda_multipole_interactions_kernel_root_rho(
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            double (&angular_corrections)[NUMBER_ANG_CORRECTIONS]) {
            int index_x = threadIdx.x + blockIdx.x;
            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(index_x + INNER_CELLS_PADDING_DEPTH,
                threadIdx.y + INNER_CELLS_PADDING_DEPTH, threadIdx.z + INNER_CELLS_PADDING_DEPTH);
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(index_x,
                                                             threadIdx.y, threadIdx.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

            // Load multipoles for this cell
            double m_cell[20];
            #pragma unroll
            for (int i = 0; i < 20; i++)
                m_cell[i] = multipoles[i * component_length + cell_flat_index];
            double X[NDIM];
            X[0] = center_of_masses[cell_flat_index];
            X[1] = center_of_masses[1 * component_length + cell_flat_index];
            X[2] = center_of_masses[2 * component_length + cell_flat_index];

            // Create and set result arrays
            double tmpstore[20];
            double tmp_corrections[3];
            #pragma unroll
            for (size_t i = 0; i < 20; ++i)
                tmpstore[i] = 0.0;
            #pragma unroll
            for (size_t i = 0; i < 3; ++i)
                tmp_corrections[i] = 0.0;

            double m_partner[20];
            double Y[NDIM];

            for (int x = 0; x < INX; x++) {
                const int stencil_x = x - cell_index_unpadded.x;
                for (int y = 0; y < INX; y++) {
                    const int stencil_y = y - cell_index_unpadded.y;
                    for (int z = 0; z < INX; z++) {
                    const int stencil_z = z - cell_index_unpadded.z;
                        const multiindex<> stencil_element(stencil_x, stencil_y, stencil_z);
                        if (stencil_x >= STENCIL_MIN && stencil_x <= STENCIL_MAX &&
                            stencil_y >= STENCIL_MIN && stencil_y <= STENCIL_MAX &&
                            stencil_z >= STENCIL_MIN && stencil_z <= STENCIL_MAX) {
                            const size_t index = (stencil_x - STENCIL_MIN)  * STENCIL_INX * STENCIL_INX +
                            (stencil_y - STENCIL_MIN) * STENCIL_INX + (stencil_z - STENCIL_MIN);
                            if (!device_stencil_indicator_const[index] ||
                                (stencil_x == 0 && stencil_y == 0 && stencil_z == 0)) {
                                continue;
                            } 
                        }
                        const multiindex<> partner_index(
                            x + INX, y + INX,
                            z + INX);
                        const size_t partner_flat_index =
                            to_flat_index_padded(partner_index);    

                        // Load data of interaction partner
                        Y[0] = center_of_masses[partner_flat_index];
                        Y[1] = center_of_masses[1 * component_length + partner_flat_index];
                        Y[2] = center_of_masses[2 * component_length + partner_flat_index];
                        #pragma unroll
                        for (size_t i = 0; i < 20; ++i)
                            m_partner[i] = multipoles[i * component_length + partner_flat_index];

                        // Do the actual calculations
                        compute_kernel_rho(X, Y, m_partner, tmpstore, tmp_corrections, m_cell,
                            [] __device__(const double& one, const double& two) -> double {
                                return std::max(one, two);
                            });
                    }
                }
            }

            // Store results in output arrays
            #pragma unroll
            for (size_t i = 0; i < 20; ++i)
                potential_expansions[i * component_length_unpadded + cell_flat_index_unpadded] =
                    tmpstore[i];

            angular_corrections[cell_flat_index_unpadded] = tmp_corrections[0];
            angular_corrections[1 * component_length_unpadded + cell_flat_index_unpadded] =
                tmp_corrections[1];
            angular_corrections[2 * component_length_unpadded + cell_flat_index_unpadded] =
                tmp_corrections[2];
        }

        __global__ void
        __launch_bounds__(INX * INX, 2)
        cuda_multipole_interactions_kernel_non_rho(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            const double theta, const bool computing_second_half) {
            int index_x = threadIdx.x + blockIdx.x;
            if (computing_second_half)
                index_x += 4;
            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(index_x + INNER_CELLS_PADDING_DEPTH,
                                                          threadIdx.y + INNER_CELLS_PADDING_DEPTH,
                                                          threadIdx.z + INNER_CELLS_PADDING_DEPTH);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(index_x,
                                                             threadIdx.y, threadIdx.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

            double X[NDIM];
            X[0] = center_of_masses[cell_flat_index];
            X[1] = center_of_masses[1 * component_length + cell_flat_index];
            X[2] = center_of_masses[2 * component_length + cell_flat_index];

            // Create and set result arrays
            double tmpstore[20];
            #pragma unroll
            for (size_t i = 0; i < 20; ++i)
                tmpstore[i] = 0.0;
            // Required for mask
            const double theta_rec_squared = sqr(1.0 / theta);
            double m_partner[20];
            double Y[NDIM];

            // calculate interactions between this cell and each stencil element
            for (int stencil_x = STENCIL_MIN; stencil_x <= STENCIL_MAX; stencil_x++) {
                int x = stencil_x - STENCIL_MIN;
                for (int stencil_y = STENCIL_MIN; stencil_y <= STENCIL_MAX; stencil_y++) {
                    int y = stencil_y - STENCIL_MIN;
                    for (int stencil_z = STENCIL_MIN; stencil_z <= STENCIL_MAX; stencil_z++) {
                        const size_t index = x * STENCIL_INX * STENCIL_INX +
                            y * STENCIL_INX + (stencil_z - STENCIL_MIN);
                        if (!device_constant_stencil_masks[index]) {
                            continue;
                        }
                        const double mask_phase_one = device_stencil_indicator_const[index];
                        const multiindex<> partner_index(cell_index.x + stencil_x,
                                                          cell_index.y + stencil_y,
                                                          cell_index.z + stencil_z);

                        const size_t partner_flat_index = to_flat_index_padded(partner_index);
                        multiindex<> partner_index_coarse(partner_index);
                        partner_index_coarse.transform_coarse();

                        // Create mask
                        const double theta_c_rec_squared = static_cast<double>(
                            distance_squared_reciprocal(cell_index_coarse, partner_index_coarse));
                        const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                        double mask = mask_b ? 1.0 : 0.0;

                        // Load data of interaction partner
                        Y[0] = center_of_masses[partner_flat_index];
                        Y[1] = center_of_masses[1 * component_length + partner_flat_index];
                        Y[2] = center_of_masses[2 * component_length + partner_flat_index];

                        m_partner[0] = local_monopoles[partner_flat_index] * mask;
                        mask = mask * mask_phase_one;    // do not load multipoles outside the inner stencil
                        m_partner[0] += multipoles[partner_flat_index] * mask;
                        #pragma unroll
                        for (size_t i = 1; i < 20; ++i)
                            m_partner[i] = multipoles[i * component_length + partner_flat_index] * mask;

                        // Do the actual calculations
                        compute_kernel_non_rho(X, Y, m_partner, tmpstore,
                            [] __device__(const double& one, const double& two) -> double {
                                return std::max(one, two);
                            });
                    }
                }
            }

            // Store results in output arrays
            #pragma unroll
            for (size_t i = 0; i < 20; ++i)
                potential_expansions[i * component_length_unpadded + cell_flat_index_unpadded] =
                    tmpstore[i];
        }

        __global__ void
        __launch_bounds__(INX * INX, 2)
        cuda_multipole_interactions_kernel_root_non_rho(
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS]) {
            int index_x = threadIdx.x + blockIdx.x;
            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(index_x + INNER_CELLS_PADDING_DEPTH,
                                                          threadIdx.y + INNER_CELLS_PADDING_DEPTH,
                                                          threadIdx.z + INNER_CELLS_PADDING_DEPTH);
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(index_x,
                                                             threadIdx.y, threadIdx.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

            double X[NDIM];
            X[0] = center_of_masses[cell_flat_index];
            X[1] = center_of_masses[1 * component_length + cell_flat_index];
            X[2] = center_of_masses[2 * component_length + cell_flat_index];

            // Create and set result arrays
            double tmpstore[20];
            #pragma unroll
            for (size_t i = 0; i < 20; ++i)
                tmpstore[i] = 0.0;
            // Required for mask
            double m_partner[20];
            double Y[NDIM];

            for (int x = 0; x < INX; x++) {
                const int stencil_x = x - cell_index_unpadded.x;
                for (int y = 0; y < INX; y++) {
                    const int stencil_y = y - cell_index_unpadded.y;
                    for (int z = 0; z < INX; z++) {
                    const int stencil_z = z - cell_index_unpadded.z;
                        double mask = 1.0;
                        const multiindex<> stencil_element(stencil_x, stencil_y, stencil_z);
                        if (stencil_x >= STENCIL_MIN && stencil_x <= STENCIL_MAX &&
                            stencil_y >= STENCIL_MIN && stencil_y <= STENCIL_MAX &&
                            stencil_z >= STENCIL_MIN && stencil_z <= STENCIL_MAX) {
                            const size_t index = (stencil_x - STENCIL_MIN)  * STENCIL_INX * STENCIL_INX +
                            (stencil_y - STENCIL_MIN) * STENCIL_INX + (stencil_z - STENCIL_MIN);
                            if (!device_stencil_indicator_const[index] ||
                                (stencil_x == 0 && stencil_y == 0 && stencil_z == 0)) {
                                continue;
                            } 
                        }
                        const multiindex<> partner_index(
                            x + INX, y + INX,
                            z + INX);
                        const size_t partner_flat_index =
                            to_flat_index_padded(partner_index);    
                        // Load data of interaction partner
                        Y[0] = center_of_masses[partner_flat_index];
                        Y[1] = center_of_masses[1 * component_length + partner_flat_index];
                        Y[2] = center_of_masses[2 * component_length + partner_flat_index];

                        #pragma unroll
                        for (size_t i = 0; i < 20; ++i)
                            m_partner[i] = multipoles[i * component_length + partner_flat_index] * mask;

                        // Do the actual calculations
                        compute_kernel_non_rho(X, Y, m_partner, tmpstore,
                            [] __device__(const double& one, const double& two) -> double {
                                return std::max(one, two);
                            });
                    }
                }
            }

            // Store results in output arrays
            #pragma unroll
            for (size_t i = 0; i < 20; ++i)
                potential_expansions[i * component_length_unpadded + cell_flat_index_unpadded] =
                    tmpstore[i];
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
