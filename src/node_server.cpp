//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/defs.hpp"
#include "octotiger/future.hpp"
#include "octotiger/node_registry.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/problem.hpp"
#include "octotiger/taylor.hpp"
#include "octotiger/util.hpp"
#include "octotiger/interaction_types.hpp"

#include "octotiger/monopole_interactions/monopole_kernel_interface.hpp"
#include "octotiger/multipole_interactions/multipole_kernel_interface.hpp"

#include "octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp"

#include <hpx/async_combinators/when_all.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>

#include <array>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <sys/stat.h>
#if !defined(_MSC_VER)
#include <unistd.h>
#endif

HPX_REGISTER_COMPONENT(hpx::components::managed_component<node_server>, node_server);

hpx::mutex node_server::node_count_mtx;
node_count_type node_server::cumulative_node_count;
bool node_server::static_initialized(false);
std::atomic<integer> node_server::static_initializing(0);

std::uint64_t node_server::cumulative_nodes_count(bool reset) {
	std::lock_guard<hpx::mutex> lock(node_count_mtx);
	if (reset) {
		cumulative_node_count.total = 0;
	}
	return cumulative_node_count.total;
}

std::uint64_t node_server::cumulative_leafs_count(bool reset) {
	std::lock_guard<hpx::mutex> lock(node_count_mtx);
	if (reset) {
		cumulative_node_count.leaf = 0;
	}
	return cumulative_node_count.leaf;
}

std::uint64_t node_server::cumulative_amrs_count(bool reset) {
	std::lock_guard<hpx::mutex> lock(node_count_mtx);
	if (reset) {
		cumulative_node_count.amr_bnd = 0;
	}
	return cumulative_node_count.amr_bnd;
}

void node_server::register_counters() {
	hpx::performance_counters::install_counter_type("/octotiger/subgrids", &cumulative_nodes_count, "total number of subgrids processed");
	hpx::performance_counters::install_counter_type("/octotiger/subgrid_leaves", &cumulative_leafs_count, "total number of subgrid leaves processed");
	hpx::performance_counters::install_counter_type("/octotiger/amr_bounds", &cumulative_amrs_count, "total number of amr bounds processed");
}

real node_server::get_rotation_count() const {
	if (opts().problem == DWD) {
		return rotational_time / (2.0 * M_PI);
	}
	return current_time;
}

future<void> node_server::exchange_flux_corrections() {
	const geo::octant ci = my_location.get_child_index();
	constexpr auto full_set = geo::face::full_set();
	for (auto &f : full_set) {
		const auto face_dim = f.get_dimension();
		auto const &this_aunt = aunts[f];
		if (!this_aunt.empty()) {
			std::array<integer, NDIM> lb, ub;
			lb[XDIM] = lb[YDIM] = lb[ZDIM] = 0;
			ub[XDIM] = ub[YDIM] = ub[ZDIM] = INX;
			if (f.get_side() == geo::MINUS) {
				lb[face_dim] = 0;
			} else {
				lb[face_dim] = INX;
			}
			ub[face_dim] = lb[face_dim] + 1;
			auto data = grid_ptr->get_flux_restrict(lb, ub, face_dim);
			this_aunt.send_hydro_flux_correct(std::move(data), f.flip(), ci);
		}
	}

	constexpr integer size = geo::face::count() * geo::quadrant::count();
	std::array<future<void>, size> futs;
	for (auto &f : futs) {
		f = hpx::make_ready_future();
	}
	integer index = 0;
	for (auto const &f : geo::face::full_set()) {
		if (this->nieces[f] == +1) {
			for (auto const &quadrant : geo::quadrant::full_set()) {
				futs[index++] = niece_hydro_channels[f][quadrant].get_future().then(
				hpx::util::annotated_function([this, f, quadrant](future<std::vector<real> > &&fdata) -> void {
					const auto face_dim = f.get_dimension();
					std::array<integer, NDIM> lb, ub;
					switch (face_dim) {
						case XDIM:
						lb[XDIM] = f.get_side() == geo::MINUS ? 0 : INX;
						lb[YDIM] = quadrant.get_side(0) * (INX / 2);
						lb[ZDIM] = quadrant.get_side(1) * (INX / 2);
						ub[XDIM] = lb[XDIM] + 1;
						ub[YDIM] = lb[YDIM] + (INX / 2);
						ub[ZDIM] = lb[ZDIM] + (INX / 2);
						break;
						case YDIM:
						lb[XDIM] = quadrant.get_side(0) * (INX / 2);
						lb[YDIM] = f.get_side() == geo::MINUS ? 0 : INX;
						lb[ZDIM] = quadrant.get_side(1) * (INX / 2);
						ub[XDIM] = lb[XDIM] + (INX / 2);
						ub[YDIM] = lb[YDIM] + 1;
						ub[ZDIM] = lb[ZDIM] + (INX / 2);
						break;
						case ZDIM:
						lb[XDIM] = quadrant.get_side(0) * (INX / 2);
						lb[YDIM] = quadrant.get_side(1) * (INX / 2);
						lb[ZDIM] = f.get_side() == geo::MINUS ? 0 : INX;
						ub[XDIM] = lb[XDIM] + (INX / 2);
						ub[YDIM] = lb[YDIM] + (INX / 2);
						ub[ZDIM] = lb[ZDIM] + 1;
						break;
					}
					grid_ptr->set_flux_restrict(GET(fdata), lb, ub, face_dim);
				}, "node_server::exchange_flux_corrections::set_flux_restrict"));
			}
		}
	}
	return hpx::when_all(std::move(futs)).then(
        hpx::util::annotated_function([](future<decltype(futs)> fout) {
		auto fin = GET(fout);
		for (auto &f : fin) {
			GET(f);
		}
	}, "node_server::exchange_flux_corrections::sync"));
}

void node_server::all_hydro_bounds() {
	exchange_interlevel_hydro_data(); // bottom up step
	collect_hydro_boundaries(); // interlevel step
	send_hydro_amr_boundaries(); // up-down step
	++hcycle;
}

void node_server::energy_hydro_bounds() {
	exchange_interlevel_hydro_data();
	collect_hydro_boundaries(true);
	send_hydro_amr_boundaries(true);
	++hcycle;
}

void node_server::exchange_interlevel_hydro_data() {
  hpx::util::annotated_function([&]() {
    if (is_refined) {
      std::vector<real> outflow(opts().n_fields, ZERO);
      for (auto const &ci : geo::octant::full_set()) {
        auto data = GET(child_hydro_channels[ci].get_future(hcycle));
        grid_ptr->set_restrict(data, ci);
        integer fi = 0;
        for (auto i = data.end() - opts().n_fields; i != data.end(); ++i) {
          outflow[fi] += *i;
          ++fi;
        }
      }
      grid_ptr->set_outflows(std::move(outflow));
    }
    auto data = grid_ptr->get_restrict();
    integer ci = my_location.get_child_index();
    if (my_location.level() != 0) {
      parent.send_hydro_children(std::move(data), ci, hcycle);
    }
  }, "all_hydro_bounds::exchange_interlevel_hydro_data")();
}

void node_server::collect_hydro_boundaries(bool energy_only) {
  hpx::util::annotated_function([&]() {
	grid_ptr->clear_amr();
  ready_for_hydro_exchange[hcycle%number_hydro_exchange_promises].set_value();
  const bool use_local_optimization = opts().optimize_local_communication;
  const bool use_local_amr_optimization = opts().optimize_local_communication;

	std::vector<hpx::lcos::shared_future<void>> neighbors_ready; 
  bool local_amr_handling = false;
	for (auto const &dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty() && neighbors[dir].is_local()) {
        /* std::vector<hpx::lcos::local::promise<void>> *neighbor_promises = neighbors[dir].hydro_ready_vec; */
        hpx::future<std::shared_ptr<node_server>> pf = hpx::get_ptr<node_server>(neighbors[dir].get_gid());
        auto direct_access = pf.get();
        neighbors_ready.emplace_back((
            direct_access->ready_for_hydro_exchange)[hcycle % number_hydro_exchange_promises]
                                         .get_shared_future());
        } else if (neighbors[dir].empty() && parent.is_local() && use_local_amr_optimization) {
            local_amr_handling = true;
        }
  }
  if (local_amr_handling && my_location.level() != 0) {
    /* std::vector<hpx::lcos::local::promise<void>> *parent_promise = parent.amr_hydro_ready_vec; */
    /* neighbors_ready.emplace_back((*parent_promise)[hcycle%number_hydro_exchange_promises].get_shared_future()); */
    hpx::future<std::shared_ptr<node_server>> pf = hpx::get_ptr<node_server>(parent.get_gid());
    auto direct_access = pf.get();
    neighbors_ready.emplace_back(
        (direct_access->ready_for_amr_hydro_exchange)[hcycle % number_hydro_exchange_promises]
            .get_shared_future());
  }
  auto get_neighbors = hpx::when_all(neighbors_ready);
  get_neighbors.get();

  std::vector<hpx::lcos::shared_future<void>> neighbors_finished_reading;
	for (auto const &dir : geo::direction::full_set()) {
    const integer width = H_BW;
    if (neighbors[dir].is_local() && use_local_optimization && !neighbors[dir].empty()) {
      /* auto fut = sibling_hydro_channels[dir].get_future(hcycle); */
      /* fut.get(); */
      hpx::future<std::shared_ptr<node_server>> pf = hpx::get_ptr<node_server>(neighbors[dir].get_gid());
      auto direct_access = pf.get();
      const auto &uneighbor = direct_access->grid_ptr->U;

      std::array<integer, NDIM> lb_orig, ub_orig;
      std::array<integer, NDIM> lb_target, ub_target;
      const auto& bw = energy_only ? grid_ptr->energy_bw : grid_ptr->field_bw;
      for (integer field = 0; field != opts().n_fields; ++field) {
        get_boundary_size(lb_orig, ub_orig, dir.flip(), INNER, INX, H_BW, bw[field]);
        get_boundary_size(lb_target, ub_target, dir, OUTER, INX, H_BW, bw[field]);
        for (integer i = 0; i < ub_target[XDIM] - lb_target[XDIM]; ++i) {
          const int i_orig = i + lb_orig[XDIM];
          const int i_target = i + lb_target[XDIM];
          for (integer j = 0; j < ub_target[YDIM] - lb_target[YDIM]; ++j) {
            const int j_orig = j + lb_orig[YDIM];
            const int j_target = j + lb_target[YDIM];
            const int k_orig = lb_orig[ZDIM];
            const int k_target = lb_target[ZDIM];
            std::copy(uneighbor[field].begin() + hindex(i_orig, j_orig, k_orig),
                uneighbor[field].begin() + hindex(i_orig, j_orig, k_orig) + ub_target[ZDIM] - lb_target[ZDIM],
                (grid_ptr->U)[field].begin() + hindex(i_target, j_target, k_target));

            /* for (integer k = 0; k < ub_target[ZDIM] - lb_target[ZDIM]; ++k) { */
            /*   const int k_orig = k + lb_orig[ZDIM]; */
            /*   const int k_target = k + lb_target[ZDIM]; */
            /*   (grid_ptr->U)[field][hindex(i_target, j_target, k_target)] = */
            /*     (*uneighbor)[field][hindex(i_orig, j_orig, k_orig)]; */
            /* } */
          }
        }
      }

      if (!opts().gravity && !is_refined) {
        /* auto neighbor_promises_p = neighbors[dir].ready_for_hydro_update; */
        hpx::future<std::shared_ptr<node_server>> pf = hpx::get_ptr<node_server>(neighbors[dir].get_gid());
        auto direct_access = pf.get();
        neighbors_finished_reading.emplace_back(
            (direct_access->ready_for_hydro_update)[hcycle % number_hydro_exchange_promises]
                .get_shared_future());
      }
      //neighbors_finished_reading
    } else if (use_local_amr_optimization && neighbors[dir].empty() && parent.is_local() && my_location.level() != 0) { 
      // Get neighbor data and the required boundaries for copying the ghostlayer
      hpx::future<std::shared_ptr<node_server>> pf = hpx::get_ptr<node_server>(parent.get_gid());
      auto direct_access = pf.get();
      const auto &uneighbor = direct_access->grid_ptr->U;
      std::array<integer, NDIM> lb_orig, ub_orig;
      std::array<integer, NDIM> lb_target, ub_target;
      get_boundary_size(lb_target, ub_target, dir, OUTER, INX / 2, H_BW);
      // Set is_coarse 
      for (integer i = 0; i < ub_target[XDIM] - lb_target[XDIM]; ++i) {
        const int i_target = i + lb_target[XDIM];
        for (integer j = 0; j < ub_target[YDIM] - lb_target[YDIM]; ++j) {
          const int j_target = j + lb_target[YDIM];
          for (integer k = 0; k < ub_target[ZDIM] - lb_target[ZDIM]; ++k) {
            const int k_target = k + lb_target[ZDIM];
            grid_ptr->is_coarse[hSindex(i_target, j_target, k_target)]++;
            assert(i_target < H_BW || i_target >= HS_NX - H_BW || j_target <
                H_BW || j_target >= HS_NX - H_BW || k_target < H_BW ||
                k_target >= HS_NX - H_BW);
          }
        }
      }
      // Adjust target region
      for (int dim = 0; dim < NDIM; dim++) {
        lb_target[dim] = std::max(lb_target[dim] - 1, integer(0));
        ub_target[dim] = std::min(ub_target[dim] + 1, integer(HS_NX));
      }
      // Get orig region
      get_boundary_size(lb_orig, ub_orig, dir, OUTER, INX / 2, H_BW);
      for (integer dim = 0; dim != NDIM; ++dim) {
        lb_orig[dim] = std::max(lb_orig[dim] - 1, integer(0));
        ub_orig[dim] = std::min(ub_orig[dim] + 1, integer(HS_NX));
        // TODO ci correct?
        /* lb_orig[dim] = lb_orig[dim] + ci.get_side(dim) * (INX / 2); */
        /* ub_orig[dim] = ub_orig[dim] + ci.get_side(dim) * (INX / 2); */
        lb_orig[dim] = lb_orig[dim] + my_location.get_child_side(dim) * (INX / 2);
        ub_orig[dim] = ub_orig[dim] + my_location.get_child_side(dim) * (INX / 2);
      }
      // Set has_coarse 
      for (integer field = 0; field != opts().n_fields; ++field) {
        if (!energy_only || field == egas_i) {
          for (integer i = 0; i < ub_target[XDIM] - lb_target[XDIM]; ++i) {
            const int i_orig = i + lb_orig[XDIM];
            const int i_target = i + lb_target[XDIM];
            for (integer j = 0; j < ub_target[YDIM] - lb_target[YDIM]; ++j) {
              const int j_orig = j + lb_orig[YDIM];
              const int j_target = j + lb_target[YDIM];
              const int k_orig = lb_orig[ZDIM];
              const int k_target = lb_target[ZDIM];
              std::copy(uneighbor[field].begin() + hindex(i_orig, j_orig, k_orig),
                  uneighbor[field].begin() + hindex(i_orig, j_orig, k_orig) + ub_target[ZDIM] - lb_target[ZDIM],
                  (grid_ptr->Ushad)[field].begin() + hSindex(i_target, j_target, k_target));

              /* for (integer k = 0; k < ub_target[ZDIM] - lb_target[ZDIM]; ++k) { */
              /*   const int k_orig = k + lb_orig[ZDIM]; */
              /*   const int k_target = k + lb_target[ZDIM]; */
              /*   grid_ptr->has_coarse[hSindex(i_target, j_target, k_target)]++; */
              /*   /1* (grid_ptr->Ushad)[field][hSindex(i_target, j_target, k_target)] = *1/ */
              /*   /1*   (uneighbor)[field][hindex(i_orig, j_orig, k_orig)]; *1/ */
              /* } */
            }
          }
        }
      }
    } else if (!neighbors[dir].empty()) {
        auto bdata = grid_ptr->get_hydro_boundary(dir, energy_only);
        neighbors[dir].send_hydro_boundary(std::move(bdata), dir.flip(), hcycle);
    }
	}
  if (!opts().gravity) {
    ready_for_hydro_update[hcycle%number_hydro_exchange_promises].set_value();
    if (!is_refined)
      all_neighbors_got_hydro[hcycle%number_hydro_exchange_promises] = hpx::when_all(neighbors_finished_reading);
  }
	std::array<future<void>, geo::direction::count()> results; 
	integer index = 0;
	for (auto const &dir : geo::direction::full_set()) {
		if (!(neighbors[dir].empty() && my_location.level() == 0)) {
      // receive data from neighbor via sibling_hydro_channels
      bool is_local = neighbors[dir].is_local();
      if (is_local && use_local_optimization && !neighbors[dir].empty()) {
        // TODO Add synchronization mechanism for U pot... (probably some sort of promise future for the actual node_sever method)
      } else if (is_local && use_local_amr_optimization && neighbors[dir].empty()){
      } else {
        results[index++] = sibling_hydro_channels[dir].get_future(hcycle).then( // 3s?
        hpx::util::annotated_function([this, energy_only, dir](future<sibling_hydro_type> &&f) -> void {
          auto &&tmp = GET(f);
          if (!neighbors[dir].empty()) {
            grid_ptr->set_hydro_boundary(tmp.data, tmp.direction, energy_only); // 1.5s
          } else {
            grid_ptr->set_hydro_amr_boundary(tmp.data, tmp.direction, energy_only); // 1.5s

          }
        }, "node_server::collect_hydro_boundaries::set_hydro_boundary"));
        // sync
        results[index - 1].get();
      }
		}
	}
	/* while (index < geo::direction::count()) { */
	/* 	results[index++] = hpx::make_ready_future(); */
	/* } */
//	wait_all_and_propagate_exceptions(std::move(results));
	/* for (auto &f : results) { */
	/* 	GET(f); */
	/* } */

	amr_boundary_type kernel_type = opts().amr_boundary_kernel_type;
  hpx::util::annotated_function([&]() {
	if (kernel_type == AMR_LEGACY) {
		grid_ptr->complete_hydro_amr_boundary(energy_only);
	} else {
		std::array<double, NDIM> xmin;
		for (int dim = 0; dim < NDIM; dim++) {
			xmin[dim] = grid_ptr->X[dim][0];
		}
// CUDA implementation supports optional execution on GPU, hence we need to check if a stram is available
#ifdef OCTOTIGER_HAVE_CUDA
		bool avail = false;
		if (kernel_type == AMR_CUDA)
      avail = true;
			/* avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor, */
			/* 		pool_strategy>(opts().cuda_buffer_capacity); */
		if (!avail) { // no stream is available or flag is turned off, proceed with CPU implementations
	#if defined __x86_64__ && defined OCTOTIGER_HAVE_VC
			complete_hydro_amr_boundary_vc(dx, energy_only, grid_ptr->Ushad, grid_ptr->is_coarse, xmin, grid_ptr->U);
	#else
			complete_hydro_amr_boundary_cpu(dx, energy_only, grid_ptr->Ushad, grid_ptr->is_coarse, xmin, grid_ptr->U);
	#endif
		} else { // Run on GPU
			launch_complete_hydro_amr_boundary_cuda(dx, energy_only, grid_ptr->Ushad,
			grid_ptr->is_coarse, xmin, grid_ptr->U);
		}
// None GPU build -> run on CPU
#else
	#if defined __x86_64__ && defined OCTOTIGER_HAVE_VC
		complete_hydro_amr_boundary_vc(dx, energy_only, grid_ptr->Ushad, grid_ptr->is_coarse, xmin, grid_ptr->U);
	#else
		complete_hydro_amr_boundary_cpu(dx, energy_only, grid_ptr->Ushad, grid_ptr->is_coarse, xmin, grid_ptr->U);
	#endif
#endif
	}
  }, "collect_hydro_boundaries::complete_hydro_amr_boundary")();
	for (auto &face : geo::face::full_set()) {
		if (my_location.is_physical_boundary(face)) {
			grid_ptr->set_physical_boundaries(face, current_time);
		}
	}
  }, "all_hydro_bounds::collect_hydro_boundaries")();
}

void node_server::send_hydro_amr_boundaries(bool energy_only) {
  hpx::util::annotated_function([&]() {
    if (is_refined) {
      // set promise 
      ready_for_amr_hydro_exchange[hcycle%number_hydro_exchange_promises].set_value();
      const bool use_local_optimization = opts().optimize_local_communication;
      // TODO only set if at least one of the children is local?
      constexpr auto full_set = geo::octant::full_set();
      for (auto &ci : full_set) {
        const auto &flags = amr_flags[ci]; // does that nephew exist and need our values?
        for (auto &dir : geo::direction::full_set()) {
          // TODO If flags and children_ci is_local, then set child amr_hydro_parent_ready_promise
          if (flags[dir] && (!children[ci].is_local() || !use_local_optimization)) { 
            std::array<integer, NDIM> lb, ub;
            std::vector<real> data;
            get_boundary_size(lb, ub, dir, OUTER, INX / 2, H_BW);
            for (integer dim = 0; dim != NDIM; ++dim) {
              lb[dim] = std::max(lb[dim] - 1, integer(0));
              ub[dim] = std::min(ub[dim] + 1, integer(HS_NX));
              lb[dim] = lb[dim] + ci.get_side(dim) * (INX / 2);
              ub[dim] = ub[dim] + ci.get_side(dim) * (INX / 2);
            }
            data = grid_ptr->get_subset(lb, ub, energy_only);
            children[ci].send_hydro_amr_boundary(std::move(data), dir, hcycle);
          }
        }
      }
    }
  }, "all_hydro_bounds::send_hydro_amr_boundaries")();
}

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y, int ulp) {
	// the machine epsilon has to be scaled to the magnitude of the values used
	// and multiplied by the desired precision in ULPs (units in the last place)
	return std::abs(x - y) < std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
	// unless the result is subnormal
			|| std::abs(x - y) < std::numeric_limits<T>::min();
}

inline bool file_exists(const std::string &name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

void node_server::clear_family() {
	parent = me = hpx::invalid_id;
	std::fill(aunts.begin(), aunts.end(), hpx::invalid_id);
	std::fill(nieces.begin(), nieces.end(), 0);
	std::fill(neighbors.begin(), neighbors.end(), hpx::invalid_id);
}

integer child_index_to_quadrant_index(integer ci, integer dim) {
	integer index;
	if (dim == XDIM) {
		index = ci >> 1;
	} else if (dim == ZDIM) {
		index = ci & 0x3;
	} else {
		index = (ci & 1) | ((ci >> 1) & 0x2);
	}
	return index;
}

void node_server::static_initialize() {
	if (!static_initialized) {
		bool test = (static_initializing++ != 0) ? true : false;
		if (!test) {
			static_initialized = true;
		}
		while (!static_initialized) {
			hpx::this_thread::yield();
		}
	}
}

void node_server::initialize(real t, real rt) {
	for (auto const &dir : geo::direction::full_set()) {
		neighbor_signals[dir].signal();
	}
	gcycle = hcycle = rcycle = 0;
	step_num = 0;
	refinement_flag = 0;
	static_initialize();
	is_refined = false;
	neighbors.resize(geo::direction::count());
	nieces.resize(NFACE);
	current_time = t;
	rotational_time = rt;
	dx = TWO * grid::get_scaling_factor() / real(INX << my_location.level());
	for (auto &d : geo::dimension::full_set()) {
		xmin[d] = grid::get_scaling_factor() * my_location.x_location(d);
	}
	if (current_time == ZERO && opts().restart_filename=="") {
		const auto p = get_problem();
		grid_ptr = std::make_shared<grid>(p, dx, xmin);
	} else {
		grid_ptr = std::make_shared<grid>(dx, xmin);
	}
	if (opts().radiation) {
		rad_grid_ptr = grid_ptr->get_rad_grid();
		rad_grid_ptr->set_dx(dx);
	}
	if (my_location.level() == 0) {
		grid_ptr->set_root();
	}
	aunts.resize(NFACE);


  number_hydro_exchange_promises = (refinement_freq() + 1) * (NRK + 1 + static_cast<int>(opts().radiation));
  /* std::cout << "promises:" << number_hydro_exchange_promises << std::endl; */
  ready_for_hydro_exchange.clear();
  for (int i = 0; i < number_hydro_exchange_promises; i++)
    ready_for_hydro_exchange.emplace_back();
  ready_for_amr_hydro_exchange.clear();
  for (int i = 0; i < number_hydro_exchange_promises; i++)
    ready_for_amr_hydro_exchange.emplace_back();
  if (!opts().gravity) {
    ready_for_hydro_update.clear();
    for (int i = 0; i < number_hydro_exchange_promises; i++)
      ready_for_hydro_update.emplace_back();
    all_neighbors_got_hydro.clear();
    for (int i = 0; i < number_hydro_exchange_promises; i++)
      all_neighbors_got_hydro.emplace_back(hpx::make_ready_future());
  }
}

node_server::~node_server() {
}

node_server::node_server(const node_location &loc, const node_client &parent_id, real t, real rt, std::size_t _step_num, std::size_t _hcycle,
		std::size_t _rcycle, std::size_t _gcycle) :
		my_location(loc), parent(parent_id) {
	initialize(t, rt);
	step_num = _step_num;
	gcycle = _gcycle;
	hcycle = _hcycle;
	rcycle = _rcycle;
}

node_server::node_server(const node_location &_my_location, integer _step_num, bool _is_refined, real _current_time, real _rotational_time,
		const std::array<integer, NCHILD> &_child_d, grid _grid, const std::vector<hpx::id_type> &_c, std::size_t _hcycle, std::size_t _rcycle,
		std::size_t _gcycle, integer position_) {
	my_location = _my_location;
	initialize(_current_time, _rotational_time);
	position = position_;
	hcycle = _hcycle;
	gcycle = _gcycle;
	rcycle = _rcycle;
	is_refined = _is_refined;
	step_num = _step_num;
	current_time = _current_time;
	rotational_time = _rotational_time;
//     grid test;
	grid_ptr = std::make_shared<grid>(std::move(_grid));
	if (is_refined) {
		std::copy(_c.begin(), _c.end(), children.begin());
	}
	child_descendant_count = _child_d;
}

void node_server::compute_fmm(gsolve_type type, bool energy_account, bool aonly) {
	if (!opts().gravity) {
		return;
	}

	future<void> parent_fut;
	if (energy_account) {
		grid_ptr->egas_to_etot();
	}
	multipole_pass_type m_out;
	m_out.first.resize(INX * INX * INX);
	m_out.second.resize(INX * INX * INX);

	for (auto const &dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			neighbor_signals[dir].wait();
		}
	}

	if (is_refined) {
		std::array<future<void>, geo::octant::count()> futs;
		integer index = 0;
		for (auto &ci : geo::octant::full_set()) {
			future<multipole_pass_type> m_in_future = child_gravity_channels[ci].get_future();

			futs[index++] = m_in_future.then(hpx::util::annotated_function([&m_out, ci](future<multipole_pass_type> &&fut) {
				const integer x0 = ci.get_side(XDIM) * INX / 2;
				const integer y0 = ci.get_side(YDIM) * INX / 2;
				const integer z0 = ci.get_side(ZDIM) * INX / 2;
				auto m_in = fut.get();
				for (integer i = 0; i != INX / 2; ++i) {
					for (integer j = 0; j != INX / 2; ++j) {
						for (integer k = 0; k != INX / 2; ++k) {
							const integer ii = i * INX * INX / 4 + j * INX / 2 + k;
							const integer io = (i + x0) * INX * INX + (j + y0) * INX + k + z0;
							m_out.first[io] = m_in.first[ii];
							m_out.second[io] = m_in.second[ii];
						}
					}
				}
			}, "node_server::compute_fmm::gather_from::child_gravity_channels"));
		}
		wait_all_and_propagate_exceptions(std::move(futs));
		m_out = grid_ptr->compute_multipoles(type, &m_out);
	} else {
		m_out = grid_ptr->compute_multipoles(type);
	}

	if (my_location.level() != 0) {
		parent.send_gravity_multipoles(std::move(m_out), my_location.get_child_index());
	}

	if (!aonly) {
		std::vector<future<void>> send_futs;
		for (auto const &dir : geo::direction::full_set()) {
			if (!neighbors[dir].empty()) {
				auto ndir = dir.flip();
				const bool is_monopole = !is_refined;
//             const auto gid = neighbors[dir].get_gid();
				const bool is_local = neighbors[dir].is_local();
				auto data = grid_ptr->get_gravity_boundary(dir, is_local);
				if (is_local) {
					data.local_semaphore = &neighbor_signals[dir];
				} else {
					neighbor_signals[dir].signal();
					data.local_semaphore = nullptr;
				}
				neighbors[dir].send_gravity_boundary(std::move(data), ndir, is_monopole, gcycle);
			}
		}
	}

	/****************************************************************************/
	// data managemenet for old and new version of interaction computation
	// all neighbors and placeholder for yourself
	bool contains_multipole = false;
	std::vector<neighbor_gravity_type> all_neighbor_interaction_data;
	for (geo::direction const &dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			all_neighbor_interaction_data.push_back(neighbor_gravity_channels[dir].get_future(gcycle).get());
			if (!all_neighbor_interaction_data[dir].is_monopole)
				contains_multipole = true;
		} else {
			all_neighbor_interaction_data.emplace_back();
		}
	}

	std::array<bool, geo::direction::count()> is_direction_empty;
	for (geo::direction const &dir : geo::direction::full_set()) {
		if (neighbors[dir].empty()) {
			is_direction_empty[dir] = true;
		} else {
			is_direction_empty[dir] = false;
		}
	}

	/* new-style interaction calculation */

	// Get all input structures we need as input
	std::vector<multipole> &M_ptr = grid_ptr->get_M();
	std::vector<real> &mon_ptr = grid_ptr->get_mon();
	std::vector<std::shared_ptr<std::vector<space_vector>>> &com_ptr = grid_ptr->get_com_ptr();

	// initialize to zero
	std::vector<expansion> &L = grid_ptr->get_L();
	std::vector<space_vector> &L_c = grid_ptr->get_L_c();
	std::fill(std::begin(L), std::end(L), ZERO);
	std::fill(std::begin(L_c), std::end(L_c), ZERO);

	// Check if we are a multipole
	if (!grid_ptr->get_leaf()) {
		// Input structure, needed for multipole-monopole interactions
		std::array<real, NDIM> Xbase = {
		grid_ptr->get_X()[0][hindex(H_BW, H_BW, H_BW)],
		grid_ptr->get_X()[1][hindex(H_BW, H_BW, H_BW)],
		grid_ptr->get_X()[2][hindex(H_BW, H_BW, H_BW)] };
		octotiger::fmm::multipole_interactions::multipole_kernel_interface(mon_ptr, M_ptr, com_ptr,
		all_neighbor_interaction_data, type, grid_ptr->get_dx(),
		is_direction_empty, Xbase, grid_ptr, grid_ptr->get_root());
	} else { // ... we are a monopole
		octotiger::fmm::monopole_interactions::monopole_kernel_interface(mon_ptr, com_ptr, all_neighbor_interaction_data, type,
		grid_ptr->get_dx(), is_direction_empty, grid_ptr, contains_multipole);
	}

	/* old-style interaction calculation
	// computes inner interactions
	grid_ptr->compute_interactions(type);
	// waits for boundary data and then computes boundary interactions
	for (auto const &dir : geo::direction::full_set()) {
		if (!is_direction_empty[dir]) {
			neighbor_gravity_type &neighbor_data = all_neighbor_interaction_data[dir];
			grid_ptr->compute_boundary_interactions(type, neighbor_data.direction, neighbor_data.is_monopole, neighbor_data.data);
		}
	} */

	/**************************************************************************/
	// now that all boundary information has been processed, signal all non-empty neighbors
	// note that this was done before during boundary calculations
	for (auto const &dir : geo::direction::full_set()) {

		if (!neighbors[dir].empty()) {
			neighbor_gravity_type &neighbor_data = all_neighbor_interaction_data[dir];
			if (neighbor_data.data.local_semaphore != nullptr) {
				neighbor_data.data.local_semaphore->signal();
			}
		}
	}
	/***************************************************************************/

	expansion_pass_type l_in;
	if (my_location.level() != 0) {
		l_in = parent_gravity_channel.get_future().get();
	}
	const expansion_pass_type ltmp = grid_ptr->compute_expansions(type, my_location.level() == 0 ? nullptr : &l_in);

	if (is_refined) {
		for (auto const &ci : geo::octant::full_set()) {
			expansion_pass_type l_out;
			l_out.first.resize(INX * INX * INX / NCHILD);
			if (type == RHO) {
				l_out.second.resize(INX * INX * INX / NCHILD);
			}
			const integer x0 = ci.get_side(XDIM) * INX / 2;
			const integer y0 = ci.get_side(YDIM) * INX / 2;
			const integer z0 = ci.get_side(ZDIM) * INX / 2;
			for (integer i = 0; i != INX / 2; ++i) {
				for (integer j = 0; j != INX / 2; ++j) {
					for (integer k = 0; k != INX / 2; ++k) {
						const integer io = i * INX * INX / 4 + j * INX / 2 + k;
						const integer ii = (i + x0) * INX * INX + (j + y0) * INX + k + z0;
						l_out.first[io] = ltmp.first[ii];
						if (type == RHO) {
							l_out.second[io] = ltmp.second[ii];
						}
					}
				}
			}
			children[ci].send_gravity_expansions(std::move(l_out));
		}
	}

	if (energy_account) {
		grid_ptr->etot_to_egas();
	}
	++gcycle;
}

void node_server::report_timing() {
	timings_.report("...");
}
