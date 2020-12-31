
#include "octotiger/geometry.hpp"
#include "octotiger/grid.hpp"

#include <array>
#include <memory>
#include <vector>

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        enum accelerator_kernel_type
        {
            OFF,
            DEVICE_CUDA,
            DEVICE_KOKKOS
        };
        enum host_kernel_type
        {
            LEGACY,
            HOST_VC,
            HOST_KOKKOS
        };

        void multipole_kernel_interface(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::array<real, NDIM> xbase, std::shared_ptr<grid> grid, const bool use_root_stencil);

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
