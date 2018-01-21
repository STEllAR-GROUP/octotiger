#pragma once

#include <memory>
#include <vector>

#include "../common_kernel/kernel_simd_types.hpp"

#include "geometry.hpp"
#include "grid.hpp"
// #include "grid.hpp"
// #include "node_server.hpp"
#include "interaction_types.hpp"
#include "taylor.hpp"

#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/multiindex.hpp"
#include "p2m_kernel.hpp"
#include "p2p_kernel.hpp"

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        /// Interface for the new monopole-multipole compute kernel
        /** Class takes an interaction partner m0 (should be a multipole out of the neighbor data /
         * mpoles) and
         * a monopole.
         */
        class monopole_interaction_interface
        {
        private:
            /*
             * logical structure of all arrays:
             * cube of 8x8x8 cells of configurable size,
             * input variables (local_expansions, center_of_masses) have
             * an additional 1-sized layer of cells around it for padding
             */

            std::vector<real> local_monopoles;
            // TODO delete these 2
            // std::vector<expansion> local_expansions;
            // std::vector<space_vector> center_of_masses;
            /// Expansions for all the multipoles the current monopole is neighboring
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING> local_expansions_SoA;
            /// com_ptr - Center of masses, required for the angular corrections
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING> center_of_masses_SoA;

            // multipole expansion on this cell (L)
            std::vector<expansion> potential_expansions;
            // angular momentum correction on this cell (L_c)
            std::vector<space_vector> angular_corrections;

            std::vector<bool> neighbor_empty_multipoles;
            std::vector<bool> neighbor_empty_monopoles;

            gsolve_type type;

            bool multipole_neighbors_exist;
            real dx;

            std::shared_ptr<grid> grid_ptr;

            p2m_kernel kernel;
            p2p_kernel kernel_monopoles;

            bool z_skip[3];
            bool y_skip[3][3];
            bool x_skip[3][3][3];

        public:
            /// The stencil is used to identify the neighbors?
            static std::vector<multiindex<>> stencil;
            static std::vector<std::array<real, 4>> four;

            /// Constructor for the boundary interactor between a monopole and its neighboring
            /// multipoles
            monopole_interaction_interface(void);
            void update_input(std::vector<real>& mons, std::vector<multipole>& multipoles,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx_arg);
            /// Computes the boundary interaction between the current monopole and the multipoles
            /** This function first converts all of the kernel input and output into the
             * struct_of_array datastructure for cache effiency. Afterwards a p2m_kernel is
             * created which will compute to monopole-multipole boundary interactions. The kernel
             * is started with the apply_stencil method. The output can be obtained with the
             * methods get_potential_expansions and get_center_of_masses
             */
            void compute_interactions(interaction_kernel_type p2p_type,
                interaction_kernel_type p2m_type,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::vector<neighbor_gravity_type>& all_neighbor_interaction_data);

            /// Returns compute kernel output regarding the potential expansions L
            std::vector<expansion>& get_potential_expansions();
            /// Returns compute kernel output regarding angular corrections L_c
            std::vector<space_vector>& get_angular_corrections();

            /// Print kernel output regarding the potential expansion
            void print_potential_expansions();
            /// Print kernel output regarding the angular corrections
            void print_angular_corrections();

            /// Add expansions onto the current kernel input expansions
            void add_to_potential_expansions(std::vector<expansion>& L);
            /// Add more center of masses to the current kernel input masses
            void add_to_center_of_masses(std::vector<space_vector>& L_c);

            void set_grid_ptr(std::shared_ptr<grid> ptr) {
                grid_ptr = ptr;
            }
        };
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
