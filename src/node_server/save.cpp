/*
 * save.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */





#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::save_action save_action_type;
HPX_REGISTER_ACTION (save_action_type);


integer node_client::save(integer i, std::string s) const {
	return hpx::async<typename node_server::save_action>(get_gid(), i, s).get();
}


integer node_server::save(integer cnt, std::string filename) const {
	char flag = is_refined ? '1' : '0';
	FILE* fp = fopen(filename.c_str(), (cnt == 0) ? "wb" : "ab");
	fwrite(&flag, sizeof(flag), 1, fp);
	++cnt;
//	printf("                                   \rSaved %li sub-grids\r", (long int) cnt);
	integer value = cnt;
	std::array<integer, NCHILD> values;
	for (auto& ci : geo::octant::full_set()) {
		if (ci != 0 && is_refined) {
			value += child_descendant_count[ci - 1];
		}
		values[ci] = value;
		fwrite(&value, sizeof(value), 1, fp);
	}
	const integer record_size = save_me(fp) + sizeof(flag) + NCHILD * sizeof(integer);
	fclose(fp);
	if (is_refined) {
		for (auto& ci : geo::octant::full_set()) {
			cnt = children[ci].save(cnt,filename);
		}
	}

	if (my_location.level() == 0) {
		FILE* fp = fopen(filename.c_str(), "ab");
		real omega = grid::get_omega();
		space_vector pivot = grid::get_pivot();
		fwrite(&omega, sizeof(real), 1, fp);
		for (auto& d : geo::dimension::full_set()) {
			fwrite(&(pivot[d]), sizeof(real), 1, fp);
		}
		fwrite(&record_size, sizeof(integer), 1, fp);
		fclose(fp);
		printf("Saved %li grids to checkpoint file\n", (long int) cnt);
	}

	return cnt;
}
