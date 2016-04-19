/*
 * regrid_gather.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */




#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::regrid_gather_action regrid_gather_action_type;
HPX_REGISTER_ACTION (regrid_gather_action_type);


hpx::future<integer> node_client::regrid_gather(bool rb) const {
	return hpx::async<typename node_server::regrid_gather_action>(get_gid(), rb);
}


integer node_server::regrid_gather(bool rebalance_only) {
	integer count = integer(1);

	if (is_refined) {
		if (!rebalance_only) {
			/* Turning refinement off */
			if (refinement_flag == 0) {
				children.clear();
				is_refined = false;
				grid_ptr->set_leaf(true);
			}
		}

		if (is_refined) {
			std::list<hpx::future<integer>> futs;
			for (auto& child : children) {
				futs.push_back(child.regrid_gather(rebalance_only));
			}
			for (auto& ci : geo::octant::full_set()) {
				auto child_cnt = futs.begin()->get();
				futs.pop_front();
				child_descendant_count[ci] = child_cnt;
				count += child_cnt;
			}
		} else {
			for (auto& ci : geo::octant::full_set()) {
				child_descendant_count[ci] = 0;
			}
		}
	} else if (!rebalance_only) {
		//		if (grid_ptr->refine_me(my_location.level())) {
		if (refinement_flag != 0) {
			refinement_flag = 0;
			count += NCHILD;

			children.resize(NCHILD);
			std::vector<node_location> clocs(NCHILD);

			/* Turning refinement on*/
			is_refined = true;
			grid_ptr->set_leaf(false);

			for (auto& ci : geo::octant::full_set()) {
				child_descendant_count[ci] = 1;
				children[ci] = hpx::new_<node_server>(hpx::find_here(), my_location.get_child(ci), me, current_time,
						rotational_time);
				std::array<integer, NDIM> lb = { 2 * H_BW, 2 * H_BW, 2 * H_BW };
				std::array<integer, NDIM> ub;
				lb[XDIM] += (1 & (ci >> 0)) * (INX);
				lb[YDIM] += (1 & (ci >> 1)) * (INX);
				lb[ZDIM] += (1 & (ci >> 2)) * (INX);
				for (integer d = 0; d != NDIM; ++d) {
					ub[d] = lb[d] + (INX);
				}
				std::vector<real> outflows(NF, ZERO);
				if (ci == 0) {
					outflows = grid_ptr->get_outflows();
				}
				if (current_time > ZERO) {
					children[ci].set_grid(grid_ptr->get_prolong(lb, ub), std::move(outflows)).get();
				}
			}
		}
	}

	return count;
}

