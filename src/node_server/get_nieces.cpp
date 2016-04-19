/*
 * get_nieces.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */

#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::get_nieces_action get_nieces_action_type;
HPX_REGISTER_ACTION (get_nieces_action_type);

hpx::future<std::vector<hpx::id_type>> node_client::get_nieces(const hpx::id_type& aunt, const geo::face& f) const {
	return hpx::async<typename node_server::get_nieces_action>(get_gid(), aunt, f);
}

std::vector<hpx::id_type> node_server::get_nieces(const hpx::id_type& aunt, const geo::face& face) const {
	std::vector < hpx::id_type > nieces;
	if (is_refined) {
		std::vector<hpx::future<void>> futs;
		nieces.reserve(geo::quadrant::count());
		futs.reserve(geo::quadrant::count());
		for (auto& ci : geo::octant::face_subset(face)) {
			nieces.push_back(children[ci].get_gid());
			futs.push_back(children[ci].set_aunt(aunt, face));
		}
		for (auto&& this_fut : futs) {
			GET(this_fut);
		}
	}
	return nieces;
}
