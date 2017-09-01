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

    // for both local and multipole expansion
    // typedef taylor<4, real> expansion;

    class m2m_interactions
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

        // com0 = *(com_ptr[0])
        std::vector<space_vector> center_of_masses;

        // multipole expansion on this cell (L)
        std::vector<expansion> potential_expansions;
        // angular momentum correction on this cell (L_c)
        std::vector<space_vector> angular_corrections;

        std::vector<bool> neighbor_empty;

        gsolve_type type;

    public:
        static std::vector<multiindex<>> stencil;

        // at this point, uses the old datamembers of the grid class as input
        // and converts them to the new data structure
        m2m_interactions(std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            // grid& g,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type);

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
}    // namespace fmm
}    // namespace octotiger
