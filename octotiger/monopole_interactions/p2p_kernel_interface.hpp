#include "octotiger/geometry.hpp"
#include "octotiger/grid.hpp"

#include <array>
#include <memory>
#include <vector>

void p2p_kernel_interface(std::vector<real>& monopoles,
    std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
    std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
    std::array<bool, geo::direction::count()>& is_direction_empty, std::shared_ptr<grid>& grid,
    const bool contains_multipole_neighbor);
