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

hpx::future<line_of_centers_t> node_client::line_of_centers(const std::pair<space_vector, space_vector>& line) const {
	return hpx::async<typename node_server::line_of_centers_action>(get_gid(), line);
}

void output_line_of_centers(FILE* fp, const line_of_centers_t& loc) {
	for (integer i = 0; i != loc.size(); ++i) {
		fprintf(fp, "%e ", loc[i].first);
		for (integer j = 0; j != NF+NGF; ++j) {
			fprintf(fp, "%e ", loc[i].second[j]);
		}
		fprintf(fp, "\n");
	}
}

line_of_centers_t node_server::line_of_centers(const std::pair<space_vector, space_vector>& line) const {
	std::list<hpx::future<line_of_centers_t>> futs;
	line_of_centers_t return_line;
	if (is_refined) {
		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].line_of_centers(line));
		}
		std::multimap<real, std::vector<real>> map;
		for (auto&& fut : futs) {
			auto tmp = fut.get();
			for (integer ii = 0; ii != tmp.size(); ++ii) {
				map.emplace(std::move(tmp[ii]));
			}
		}
		return_line.resize(map.size());
		std::move(map.begin(), map.end(), return_line.begin());
	} else {
		return_line = grid_ptr->line_of_centers(line);
	}

	return return_line;
}

void line_of_centers_analyze(const line_of_centers_t& loc, real omega, std::pair<real, real>& rho1_max,
		std::pair<real, real>& rho2_max, std::pair<real, real>& l1_phi, real& l1_outer1, real& l1_outer2) {
	rho1_max.second = rho2_max.second = 0.0;
	integer rho1_maxi, rho2_maxi;
	///	printf( "LOCSIZE %i\n", loc.size());
	for (integer i = 0; i != loc.size(); ++i) {
		const real x = loc[i].first;
		const real rho = loc[i].second[rho_i];
	//	printf( "%e %e\n", x, rho);
		if (rho1_max.second < rho) {
		//	printf( "!\n");
			rho1_max.second = rho;
			rho1_max.first = x;
			rho1_maxi = i;
		}
	}
	for (integer i = 0; i != loc.size(); ++i) {
		const real x = loc[i].first;
		if (x * rho1_max.first < 0.0) {
			const real rho = loc[i].second[rho_i];
			if (rho2_max.second < rho) {
				rho2_max.second = rho;
				rho2_max.first = x;
				rho2_maxi = i;
			}
		}
	}
	l1_phi.second = -std::numeric_limits < real > ::max();
	for (integer i = 0; i != loc.size(); ++i) {
		const real x = loc[i].first;
		if (x > std::min(rho1_max.first, rho2_max.first) && x < std::max(rho1_max.first, rho2_max.first)) {
			const real rho = loc[i].second[rho_i];
			const real pot = loc[i].second[pot_i];
			real phi_eff = pot / rho - 0.5 * x * x * omega * omega;
			if (phi_eff > l1_phi.second) {
				l1_phi.second = phi_eff;
				l1_phi.first = x;
			}
		}
	}
	for( integer i = rho2_maxi; i < loc.size() - 1; ++i) {
		const real rho1 = loc[i].second[rho_i];
		const real pot1 = loc[i].second[pot_i];
		const real rho2 = loc[i+1].second[rho_i];
		const real pot2 = loc[i+1].second[pot_i];
		const real x1 = loc[i].first;
		const real x2 = loc[i+1].first;
		real phi_eff1 = pot1 / rho1 - 0.5 * x1 * x1 * omega * omega;
		real phi_eff2 = pot2 / rho2 - 0.5 * x2 * x2 * omega * omega;
		if( (phi_eff1 - l1_phi.second)*(l1_phi.second - phi_eff2) >= 0.0) {
			l1_outer2 = (x1+x2)/2.0;
			break;
		}
	}
	for( integer i = rho1_maxi; i >= 1; --i) {
		const real rho1 = loc[i].second[rho_i];
		const real pot1 = loc[i].second[pot_i];
		const real rho2 = loc[i-1].second[rho_i];
		const real pot2 = loc[i-1].second[pot_i];
		const real x1 = loc[i].first;
		const real x2 = loc[i-1].first;
		real phi_eff1 = pot1 / rho1 - 0.5 * x1 * x1 * omega * omega;
		real phi_eff2 = pot2 / rho2 - 0.5 * x2 * x2 * omega * omega;
		if( (phi_eff1 - l1_phi.second)*(l1_phi.second - phi_eff2) >= 0.0) {
			l1_outer1 = (x1+x2)/2.0;
			break;
		}
	}
}
