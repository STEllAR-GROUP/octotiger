#pragma once

#include <memory>
#include <vector>

#include "m2m_simd_types.hpp"

#include "geometry.hpp"
// #include "grid.hpp"
// #include "node_server.hpp"
#include "interaction_types.hpp"
#include "taylor.hpp"

#include "interaction_constants.hpp"
#include "multiindex.hpp"

namespace octotiger {
namespace fmm {
    namespace p2p_kernel {

        // for both local and multipole expansion
        // typedef taylor<4, real> expansion;

        class p2p_interactions
        {
        private:
            /*
             * logical structure of all arrays:
             * cube of 8x8x8 cells of configurable size,
             * input variables (local_expansions, center_of_masses) have
             * an additional 1-sized layer of cells around it for padding
             */

            // M_ptr
            std::vector<real> local_expansions;

            // multipole expansion on this cell (L)
            std::vector<expansion> potential_expansions;

            std::vector<bool> neighbor_empty;

            gsolve_type type;

            real dx;

        public:
            static std::vector<multiindex<>> stencil;

            // at this point, uses the old datamembers of the grid class as input
            // and converts them to the new data structure
            p2p_interactions(std::vector<real>& mons, std::vector<neighbor_gravity_type>& neighbors,
                gsolve_type type, real dx);

            void compute_interactions();

            std::vector<real>& get_local_expansions();

            std::vector<expansion>& get_potential_expansions();

            void print_potential_expansions();

            void add_to_potential_expansions(std::vector<expansion>& L);
        };
    }    // namespace p2p_kernel
}    // namespace fmm
}    // namespace octotiger
