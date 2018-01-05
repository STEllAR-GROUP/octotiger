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
    namespace multipole_interactions {

        // for both local and multipole expansion
        // typedef taylor<4, real> expansion;

        class multipole_interaction_interface
        {
        private:
            /*
             * logical structure of all arrays:
             * cube of 8x8x8 cells of configurable size,
             * input variables (local_expansions, center_of_masses) have
             * an additional 1-sized layer of cells around it for padding
             */

            // M_ptr
            std::vector<expansion> local_expansions;
            std::vector<real> local_monopoles;

            // com0 = *(com_ptr[0])
            std::vector<space_vector> center_of_masses;

            // multipole expansion on this cell (L)
            std::vector<expansion> potential_expansions;
            // angular momentum correction on this cell (L_c)
            std::vector<space_vector> angular_corrections;
            std::vector<bool> interact;

            std::vector<bool> neighbor_empty_multipole;
            std::vector<bool> neighbor_empty_monopole;

            gsolve_type type;

            bool monopole_neighbors_exist;
            real dX;
            std::array<real, NDIM> xBase;

        public:
            static std::vector<multiindex<>> stencil_multipole_interactions;
            static std::vector<multiindex<>> stencil_mixed_interactions;

            // at this point, uses the old datamembers of the grid class as input
            // and converts them to the new data structure
            multipole_interaction_interface(void);
            void update_input(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<real, NDIM> xbase);

            void compute_interactions();

            // void get_converted_local_expansions(std::vector<multipole>& M_ptr);

            std::vector<expansion>& get_local_expansions();

            // void get_converted_center_of_masses(
            //     std::vector<std::shared_ptr<std::vector<space_vector>>> com_ptr);

            std::vector<space_vector>& get_center_of_masses();

            // void get_converted_potential_expansions(std::vector<expansion>& L);

            std::vector<expansion>& get_potential_expansions();

            // void get_converted_angular_corrections(std::vector<space_vector>& L_c);

            std::vector<space_vector>& get_angular_corrections();

            void print_potential_expansions();

            void print_angular_corrections();

            void print_local_expansions();

            void print_center_of_masses();

            void add_to_potential_expansions(std::vector<expansion>& L);

            void add_to_center_of_masses(std::vector<space_vector>& L_c);
        };
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
