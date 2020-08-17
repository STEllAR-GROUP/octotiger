#include "octotiger/grid.hpp"
#include "octotiger/geometry.hpp"

#include <array>
#include <vector>
#include <memory>

void p2p_kernel_interface(std::vector<real>& monopoles,
    std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
    std::array<bool, geo::direction::count()>& is_direction_empty, std::shared_ptr<grid> grid);