
#include "octotiger/geometry.hpp"
#include "octotiger/grid.hpp"

#include <array>
#include <memory>
#include <vector>

void multipole_kernel_interface(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
    std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
    std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
    std::array<bool, geo::direction::count()>& is_direction_empty,
    std::array<real, NDIM> xbase, std::shared_ptr<grid> grid, const bool use_root_stencil);
