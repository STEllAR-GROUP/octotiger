/*
 * line_of_centers.cpp
 *
 *  Created on: Apr 20, 2016
 *      Author: dmarce1
 */

#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::line_of_centers_action line_of_centers_action_type;
HPX_REGISTER_ACTION (line_of_centers_action_type);

hpx::future<line_of_centers_t> node_client::line_of_centers(const space_vector& line) {
	return hpx::async<typename node_server::line_of_centers_action>(get_gid(), line);
}

void output_line_of_centers(FILE* fp, const line_of_centers_t& loc) {
	for (integer i = 0; i != loc.size(); ++i) {
		fprintf(fp, "%e ", loc[i].first);
		for (integer j = 0; j != NF; ++j) {
			fprintf(fp, "%e ", loc[i].second[j]);
		}
		fprintf(fp, "\n");
	}
}

line_of_centers_t node_server::line_of_centers(const space_vector& line) {
	std::list<hpx::future<line_of_centers_t>> futs;
	line_of_centers_t return_line;
	if (is_refined) {
		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].line_of_centers(line));
		}
		std::map<real, std::array<real, NF>> map;
		for (auto&& fut : futs) {
			auto tmp = fut.get();
			for (auto&& ln : tmp) {
				map.insert(ln);
			}
		}
		return_line.resize(map.size());
		std::move(map.begin(), map.end(), return_line.begin());
	} else {
		return_line = grid_ptr->line_of_centers(line);
	}
	return_line;
}
