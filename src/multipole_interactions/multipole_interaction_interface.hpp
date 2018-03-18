#pragma once

#include <memory>
#include <vector>

#include "../common_kernel/kernel_simd_types.hpp"

#include "geometry.hpp"
#include "grid.hpp"
// #include "node_server.hpp"
#include "interaction_types.hpp"
#include "taylor.hpp"

#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/multiindex.hpp"
#include "m2m_kernel.hpp"
#include "m2p_kernel.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        // for both local and multipole expansion
        // typedef taylor<4, real> expansion;

        class multipole_interaction_interface
        {
        public:
            multipole_interaction_interface(void);
            void compute_multipole_interactions(std::vector<real>& monopoles,
                std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::array<real, NDIM> xbase);
            void set_grid_ptr(std::shared_ptr<grid> ptr) {
                grid_ptr = ptr;
            }

        protected:
            void update_input(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<real, NDIM> xbase, std::vector<real> &local_monopoles,
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                    &local_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                    &center_of_masses_SoA);
            void compute_interactions(std::array<bool, geo::direction::count()>& is_direction_empty,
                std::vector<neighbor_gravity_type>& all_neighbor_interaction_data,
                const std::vector<real> &local_monopoles,
                const struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                    &local_expansions_SoA,
                const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                    &center_of_masses_SoA);

        protected:
            std::vector<real> local_monopoles_staging_area;
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING> local_expansions_staging_area;
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING> center_of_masses_staging_area;
            static thread_local const two_phase_stencil stencil;
            gsolve_type type;
            real dX;
            std::array<real, NDIM> xBase;
            interaction_kernel_type m2m_type;
            std::shared_ptr<grid> grid_ptr;
        };
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
