#include "octotiger/monopole_interactions/p2p_kernel_interface.hpp"
#include "octotiger/monopole_interactions/legacy/cuda_p2p_interaction_interface.hpp"
#include "octotiger/options.hpp"

#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/monopole_interactions/legacy/p2p_interaction_interface.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/options.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include <aligned_buffer_util.hpp>
#include <buffer_manager.hpp>

void p2p_kernel_interface(std::vector<real>& monopoles,
    std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
    std::array<bool, geo::direction::count()>& is_direction_empty,
    std::shared_ptr<grid> grid_ptr) {
    bool use_legacy = true;

    // Try accelerator implementation

    // Try host implementation

    // try legacy implemantation
#ifdef OCTOTIGER_HAVE_CUDA
    octotiger::fmm::monopole_interactions::cuda_p2p_interaction_interface p2p_interactor{};
#else
    octotiger::fmm::monopole_interactions::p2p_interaction_interface p2p_interactor{};
#endif
    p2p_interactor.set_grid_ptr(grid_ptr);
    p2p_interactor.compute_p2p_interactions(monopoles, neighbors, type, dx, is_direction_empty);

    return;
}