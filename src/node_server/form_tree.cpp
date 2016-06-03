/*
 * form_tree.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */


#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::form_tree_action form_tree_action_type;
HPX_REGISTER_ACTION (form_tree_action_type);

hpx::future<void> node_client::form_tree(const hpx::id_type& id1, const hpx::id_type& id2,
		const std::vector<hpx::id_type>& ids) {
	return hpx::async<typename node_server::form_tree_action>(get_gid(), id1, id2, std::move(ids));
}


void node_server::form_tree(const hpx::id_type& self_gid, const hpx::id_type& parent_gid,
		const std::vector<hpx::id_type>& neighbor_gids) {
	for (auto& dir : geo::direction::full_set()) {
		neighbors[dir] = neighbor_gids[dir];
	}
	for (auto& face : geo::face::full_set()) {
		siblings[face] = neighbors[face.get_direction()];
	}

	std::list<hpx::future<void>> cfuts;
	me = self_gid;
	parent = parent_gid;
	if (is_refined) {
		amr_flags.resize(NCHILD);
		for (integer cx = 0; cx != 2; ++cx) {
			for (integer cy = 0; cy != 2; ++cy) {
				for (integer cz = 0; cz != 2; ++cz) {
					std::vector<hpx::future<hpx::id_type>> child_neighbors_f(geo::direction::count());
					std::vector<hpx::id_type> child_neighbors(geo::direction::count());
					const integer ci = cx + 2 * cy + 4 * cz;
					for (integer dx = -1; dx != 2; ++dx) {
						for (integer dy = -1; dy != 2; ++dy) {
							for (integer dz = -1; dz != 2; ++dz) {
								if (!(dx == 0 && dy == 0 && dz == 0)) {
									const integer x = cx + dx + 2;
									const integer y = cy + dy + 2;
									const integer z = cz + dz + 2;
									geo::direction i;
									i.set(dx, dy, dz);
									auto& ref = child_neighbors_f[i];
									auto other_child = (x % 2) + 2 * (y % 2) + 4 * (z % 2);
									if (x / 2 == 1 && y / 2 == 1 && z / 2 == 1) {
										ref = hpx::make_ready_future<hpx::id_type>(children[other_child].get_gid());
									} else {
										geo::direction dir = geo::direction(
												(x / 2) + NDIM * ((y / 2) + NDIM * (z / 2)));
										ref = neighbors[dir].get_child_client(other_child);
									}
								}
							}
						}
					}

					for (auto& dir : geo::direction::full_set()) {
						child_neighbors[dir] = child_neighbors_f[dir].get();
						if (child_neighbors[dir] == hpx::invalid_id) {
							amr_flags[ci][dir] = true;
						} else {
							amr_flags[ci][dir] = false;
						}
					}
					cfuts.push_back(
							children[ci].form_tree(children[ci].get_gid(), me.get_gid(), std::move(child_neighbors)));
				}
			}
		}

		for (auto&& fut : cfuts) {
			GET(fut);
		}

	} else {
		std::vector<hpx::future<std::vector<hpx::id_type>>>nfuts(NFACE);
		for (auto& f : geo::face::full_set()) {
			if( !siblings[f].empty()) {
				nfuts[f] = siblings[f].get_nieces(me.get_gid(), f^1);
			} else {
				nfuts[f] = hpx::make_ready_future(std::vector<hpx::id_type>());
			}
		}
		for (auto& f : geo::face::full_set()) {
			auto ids = nfuts[f].get();
			nieces[f].resize(ids.size());
			for( std::size_t i = 0; i != ids.size(); ++i ) {
				nieces[f][i] = ids[i];
			}
		}
	}

}
