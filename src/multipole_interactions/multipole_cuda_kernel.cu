#ifdef OCTOTIGER_CUDA_ENABLED
#include <sstream>
#include "compute_kernel_templates.hpp"
#include "multipole_cuda_kernel.hpp"
namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        __constant__ double device_stencil_indicator_const[STENCIL_SIZE];
        __constant__ octotiger::fmm::multiindex<> device_stencil_const[STENCIL_SIZE];
        void copy_stencil_to_m2m_constant_memory(const multiindex<> *stencil, const size_t stencil_size) {
            cudaError_t err = cudaMemcpyToSymbol(device_stencil_const, stencil, stencil_size);
            if (err != cudaSuccess) {
                std::stringstream temp;
                temp << "Copy stencil to constant memory returned error code " << cudaGetErrorString(err);
                throw std::runtime_error(temp.str());
            }
        }
        void copy_indicator_to_m2m_constant_memory(const double *indicator, const size_t indicator_size) {
            cudaError_t err = cudaMemcpyToSymbol(device_stencil_indicator_const, indicator, indicator_size);
            if (err != cudaSuccess) {
                std::stringstream temp;
                temp << "Copy stencil indicator to constant memory returned error code " << cudaGetErrorString(err);
                throw std::runtime_error(temp.str());
            }
        }

        __device__ constexpr size_t component_length = ENTRIES + SOA_PADDING;
        __device__ constexpr size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;

        __global__ void
        __launch_bounds__(512, 1)
        cuda_multipole_interactions_kernel_rho(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            double (&angular_corrections)[NUMBER_ANG_CORRECTIONS],
            const double theta) {
            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(threadIdx.x + INNER_CELLS_PADDING_DEPTH,
                threadIdx.y + INNER_CELLS_PADDING_DEPTH, threadIdx.z + INNER_CELLS_PADDING_DEPTH);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(threadIdx.x, threadIdx.y, threadIdx.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

            // Load multipoles for this cell
            double m_cell[20];
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
            for (size_t stencil_index = 0; stencil_index < STENCIL_SIZE; stencil_index++) {
                // Get phase indicator (indicates whether multipole multipole interactions still
                // needs to be done)
                const double mask_phase_one = device_stencil_indicator_const[stencil_index];

                // Get interaction partner indices
                const multiindex<> partner_index(cell_index.x + device_stencil_const[stencil_index].x,
                                                 cell_index.y + device_stencil_const[stencil_index].y,
                                                 cell_index.z + device_stencil_const[stencil_index].z);
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
        __launch_bounds__(512, 1)
        cuda_multipole_interactions_kernel_non_rho(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            const double theta) {
            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(threadIdx.x + INNER_CELLS_PADDING_DEPTH,
                threadIdx.y + INNER_CELLS_PADDING_DEPTH, threadIdx.z + INNER_CELLS_PADDING_DEPTH);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(threadIdx.x, threadIdx.y, threadIdx.z);
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
            for (size_t stencil_index = 0; stencil_index < STENCIL_SIZE; stencil_index++) {
                // Get phase indicator (indicates whether multipole multipole interactions still
                // needs to be done)
                const double mask_phase_one = device_stencil_indicator_const[stencil_index];

                // Get interaction partner indices
                const multiindex<>& stencil_element = device_stencil_const[stencil_index];
                const multiindex<> partner_index(cell_index.x + stencil_element.x,
                                                 cell_index.y + stencil_element.y,
                                                 cell_index.z + stencil_element.z);
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
