#pragma once

#include <memory>
#include <vector>

#include "../common_kernel/kernel_simd_types.hpp"

#include "geometry.hpp"
// #include "grid.hpp"
// #include "node_server.hpp"
#include "interaction_types.hpp"
#include "taylor.hpp"

#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/multiindex.hpp"

namespace octotiger {
namespace fmm {
    namespace m2p_kernel {
        /// Interface for the new monopole-multipole compute kernel
        /** Class takes an interaction partner m0 (should be a multipole out of the neighbor data /
         * mpoles) and
         * a monopole.
         */
        class m2p_interactions
        {
        private:
            /*
             * logical structure of all arrays:
             * cube of 8x8x8 cells of configurable size,
             * input variables (local_expansions, center_of_masses) have
             * an additional 1-sized layer of cells around it for padding
             */

            /// Expansions for all the multipoles the current monopole is neighboring
            std::vector<real> local_expansions;
            std::vector<expansion> cell_expansions;

            /// com_ptr - Center of masses, required for the angular corrections
            std::vector<space_vector> center_of_masses;

            // monopole expansions
            std::vector<expansion> potential_expansions;
            // angular momentum correction on this cell (L_c)
            std::vector<space_vector> angular_corrections;

            std::vector<bool> neighbor_empty;

            gsolve_type type;

            bool monopole_neighbors_exist;

        public:
            /// The stencil is used to identify the neighbors?
            static std::vector<multiindex<>> stencil;
            /// Constructor for the boundary interactor between a monopole and its neighboring
            /// multipoles
            m2p_interactions(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type);
            /// Computes the boundary interaction between the current monopole and the multipoles
            /** This function first converts all of the kernel input and output into the
             * struct_of_array datastructure for cache effiency. Afterwards a m2p_kernel is
             * created which will compute to monopole-multipole boundary interactions. The kernel
             * is started with the apply_stencil method. The output can be obtained with the
             * methods get_potential_expansions and get_center_of_masses
             */
            void compute_interactions();
            /// Get the local expansion input of the compute kernel
            std::vector<real>& get_local_expansions();
            /// Get the center of mass input of the compute kernel
            std::vector<space_vector>& get_center_of_masses();
            /// Returns compute kernel output regarding the potential expansions L
            std::vector<expansion>& get_potential_expansions();
            /// Returns compute kernel output regarding angular corrections L_c
            std::vector<space_vector>& get_angular_corrections();

            /// Prints expansion kernel input
            void print_local_expansions();
            /// Print center of mass kernel input
            void print_center_of_masses();
            /// Print kernel output regarding the potential expansion
            void print_potential_expansions();
            /// Print kernel output regarding the angular corrections
            void print_angular_corrections();

            /// Add expansions onto the current kernel input expansions
            void add_to_potential_expansions(std::vector<expansion>& L);
            /// Add more center of masses to the current kernel input masses
            void add_to_center_of_masses(std::vector<space_vector>& L_c);
        };
    }    // namespace m2p_kernel
}    // namespace fmm
}    // namespace octotiger
