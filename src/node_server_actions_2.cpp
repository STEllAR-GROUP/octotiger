#include "node_server.hpp"
#include "node_client.hpp"
#include "diagnostics.hpp"
#include "options.hpp"
#include "taylor.hpp"
#include "util.hpp"
#include "profiler.hpp"
#include <mpi.h>

typedef node_server::check_for_refinement_action check_for_refinement_action_type;
HPX_REGISTER_ACTION(check_for_refinement_action_type);

hpx::future<bool> node_client::check_for_refinement() const {
	return hpx::async<typename node_server::check_for_refinement_action>(get_gid());
}

bool node_server::check_for_refinement() {
	bool rc = false;
	std::list<hpx::future<bool>> futs;
	if (is_refined) {
		for (auto& child : children) {
			futs.push_back(child.check_for_refinement());
		}
	}
	if (hydro_on) {
		all_hydro_bounds().get();
	}
	hpx::wait_all(futs.begin(), futs.end());
	futs.clear();
	if (!rc) {
		rc = grid_ptr->refine_me(my_location.level());
	}
	if (rc) {
		if (refinement_flag++ == 0) {
			if (!parent.empty()) {
				const auto neighbors = my_location.get_neighbors();
				parent.force_nodes_to_exist(std::list<node_location>(neighbors.begin(), neighbors.end())).get();
			}
		}
	}
	return refinement_flag;
}

typedef node_server::copy_to_locality_action copy_to_locality_action_type;
HPX_REGISTER_ACTION(copy_to_locality_action_type);

hpx::future<hpx::id_type> node_client::copy_to_locality(const hpx::id_type& id) const {
	return hpx::async<typename node_server::copy_to_locality_action>(get_gid(), id);
}

hpx::future<hpx::id_type> node_server::copy_to_locality(const hpx::id_type& id) {

	std::vector<hpx::id_type> cids;
	if (is_refined) {
		cids.resize(NCHILD);
		for (auto& ci : geo::octant::full_set()) {
			cids[ci] = children[ci].get_gid();
		}
	}
	auto rc = hpx::new_<node_server>(id, my_location, step_num, is_refined, current_time, rotational_time, child_descendant_count, std::move(*grid_ptr), cids);
	clear_family();
	return rc;
}

extern options opts;

typedef node_server::diagnostics_action diagnostics_action_type;
HPX_REGISTER_ACTION(diagnostics_action_type);

hpx::future<diagnostics_t> node_client::diagnostics(const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, real c1,
	real c2) const {
	return hpx::async<typename node_server::diagnostics_action>(get_gid(), axis, l1, c1, c2);
}

diagnostics_t node_server::diagnostics() const {
	auto axis = grid_ptr->find_axis();
	auto loc = line_of_centers(axis);
	real this_omega = grid::get_omega();
	std::pair<real, real> rho1, rho2, l1, l2, l3;
	real phi_1, phi_2;
	line_of_centers_analyze(loc, this_omega, rho1, rho2, l1, l2, l3, phi_1, phi_2);
	//if( rho1.first > rho2.first ) {
	//	for( integer d = 0; d != NDIM; ++d ) {
	//		//printf( "Flipping axis\n" );
	//		axis.first[d] = -axis.first[d];
	//		loc = line_of_centers(axis);
	//		line_of_centers_analyze(loc, this_omega, rho1, rho2, l1, phi_1, phi_2);
	//	}
//	}
	auto diags = diagnostics(axis, l1, rho1.first, rho2.first);

	diags.z_moment -= diags.grid_sum[rho_i] * (std::pow(diags.grid_com[XDIM], 2) + std::pow(diags.grid_com[YDIM], 2));
	diags.primary_z_moment -= diags.primary_sum[rho_i] * (std::pow(diags.primary_com[XDIM], 2) + std::pow(diags.primary_com[YDIM], 2));
	diags.secondary_z_moment -= diags.secondary_sum[rho_i] * (std::pow(diags.secondary_com[XDIM], 2) + std::pow(diags.secondary_com[YDIM], 2));
	if (diags.primary_sum[rho_i] < diags.secondary_sum[rho_i]) {
		std::swap(diags.primary_sum, diags.secondary_sum);
		std::swap(diags.primary_com, diags.secondary_com);
		std::swap(diags.primary_com_dot, diags.secondary_com_dot);
		std::swap(rho1, rho2);
		std::swap(phi_1, phi_2);
		std::swap(diags.primary_z_moment, diags.secondary_z_moment);
	}

	if (opts.problem != SOLID_SPHERE) {
		FILE* fp = fopen("diag.dat", "at");
		fprintf(fp, "%23.16e ", double(current_time));
		for (integer f = 0; f != NF; ++f) {
			fprintf(fp, "%23.16e ", double(diags.grid_sum[f] + diags.outflow_sum[f]));
			fprintf(fp, "%23.16e ", double(diags.outflow_sum[f]));
		}
		for (integer f = 0; f != NDIM; ++f) {
			fprintf(fp, "%23.16e ", double(diags.l_sum[f]));
		}
		fprintf(fp, "\n");
		fclose(fp);
		real a = 0.0;
		for (integer d = 0; d != NDIM; ++d) {
			a += std::pow(diags.primary_com[d] - diags.secondary_com[d], 2);
		}
		a = std::sqrt(a);
		real j1 = 0.0;
		real j2 = 0.0;
		real m1 = diags.primary_sum[rho_i];
		real m2 = diags.secondary_sum[rho_i];
		j1 -= diags.primary_com_dot[XDIM] * (diags.primary_com[YDIM] - diags.grid_com[YDIM]) * m1;
		j1 += diags.primary_com_dot[YDIM] * (diags.primary_com[XDIM] - diags.grid_com[XDIM]) * m1;
		j2 -= diags.secondary_com_dot[XDIM] * (diags.secondary_com[YDIM] - diags.grid_com[YDIM]) * m2;
		j2 += diags.secondary_com_dot[YDIM] * (diags.secondary_com[XDIM] - diags.grid_com[XDIM]) * m2;
		const real jorb = j1 + j2;
		j1 = diags.primary_sum[zz_i] - j1;
		j2 = diags.secondary_sum[zz_i] - j2;
		fp = fopen("binary.dat", "at");
		fprintf(fp, "%15.8e ", double(current_time));
		fprintf(fp, "%15.8e ", double(m1));
		fprintf(fp, "%15.8e ", double(m2));
		fprintf(fp, "%15.8e ", double(this_omega));
		fprintf(fp, "%15.8e ", double(a));
		fprintf(fp, "%15.8e ", double(rho1.second));
		fprintf(fp, "%15.8e ", double(rho2.second));
		fprintf(fp, "%15.8e ", double(jorb));
		fprintf(fp, "%15.8e ", double(j1));
		fprintf(fp, "%15.8e ", double(j2));
		fprintf(fp, "%15.8e ", double(diags.z_moment));
		fprintf(fp, "%15.8e ", double(diags.primary_z_moment));
		fprintf(fp, "%15.8e ", double(diags.secondary_z_moment));
		fprintf(fp, "\n");
		fclose(fp);

		fp = fopen("minmax.dat", "at");
		fprintf(fp, "%23.16e ", double(current_time));
		for (integer f = 0; f != NF; ++f) {
			fprintf(fp, "%23.16e ", double(diags.field_min[f]));
			fprintf(fp, "%23.16e ", double(diags.field_max[f]));
		}
		fprintf(fp, "\n");
		fclose(fp);

		fp = fopen("com.dat", "at");
		fprintf(fp, "%23.16e ", double(current_time));
		for (integer d = 0; d != NDIM; ++d) {
			fprintf(fp, "%23.16e ", double(diags.primary_com[d]));
		}
		for (integer d = 0; d != NDIM; ++d) {
			fprintf(fp, "%23.16e ", double(diags.secondary_com[d]));
		}
		for (integer d = 0; d != NDIM; ++d) {
			fprintf(fp, "%23.16e ", double(diags.grid_com[d]));
		}
		fprintf(fp, "\n");
		fclose(fp);

	} else {
		printf("L1\n");
		printf("Gravity Phi Error - %e\n", (diags.l1_error[0] / diags.l1_error[4]));
		printf("Gravity gx Error - %e\n", (diags.l1_error[1] / diags.l1_error[5]));
		printf("Gravity gy Error - %e\n", (diags.l1_error[2] / diags.l1_error[6]));
		printf("Gravity gz Error - %e\n", (diags.l1_error[3] / diags.l1_error[7]));
		printf("L2\n");
		printf("Gravity Phi Error - %e\n", std::sqrt(diags.l2_error[0] / diags.l2_error[4]));
		printf("Gravity gx Error - %e\n", std::sqrt(diags.l2_error[1] / diags.l2_error[5]));
		printf("Gravity gy Error - %e\n", std::sqrt(diags.l2_error[2] / diags.l2_error[6]));
		printf("Gravity gz Error - %e\n", std::sqrt(diags.l2_error[3] / diags.l2_error[7]));
		printf("Total Mass = %e\n", diags.grid_sum[rho_i]);
		for (integer d = 0; d != NDIM; ++d) {
			printf("%e %e\n", diags.gforce_sum[d], diags.gtorque_sum[d]);
		}
	}

	return diags;
}

diagnostics_t node_server::diagnostics(const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1, real c1, real c2) const {

	diagnostics_t sums;
	if (is_refined) {
		std::list<hpx::future<diagnostics_t>> futs;
		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].diagnostics(axis, l1, c1, c2));
		}
		for (auto ci = futs.begin(); ci != futs.end(); ++ci) {
			auto this_sum = ci->get();
			sums += this_sum;
		}
	} else {

		sums.primary_sum = grid_ptr->conserved_sums(sums.primary_com, sums.primary_com_dot, axis, l1, +1);
		sums.secondary_sum = grid_ptr->conserved_sums(sums.secondary_com, sums.secondary_com_dot, axis, l1, -1);
		sums.primary_z_moment = grid_ptr->z_moments(axis, l1, +1);
		sums.secondary_z_moment = grid_ptr->z_moments(axis, l1, -1);
		sums.grid_sum = grid_ptr->conserved_sums(sums.grid_com, sums.grid_com_dot, axis, l1, 0);
		sums.outflow_sum = grid_ptr->conserved_outflows();
		sums.l_sum = grid_ptr->l_sums();
		auto tmp = grid_ptr->field_range();
		sums.field_min = std::move(tmp.first);
		sums.field_max = std::move(tmp.second);
		sums.gforce_sum = grid_ptr->gforce_sum(false);
		sums.gtorque_sum = grid_ptr->gforce_sum(true);
		auto tmp2 = grid_ptr->diagnostic_error();
		sums.l1_error = tmp2.first;
		sums.l2_error = tmp2.second;
		auto vols = grid_ptr->frac_volumes();
		sums.roche_vol1 = grid_ptr->roche_volume(axis, l1, std::min(c1, c2), false);
		sums.roche_vol2 = grid_ptr->roche_volume(axis, l1, std::max(c1, c2), true);
		sums.primary_volume = vols[spc_ac_i - spc_i] + vols[spc_ae_i - spc_i];
		sums.secondary_volume = vols[spc_dc_i - spc_i] + vols[spc_de_i - spc_i];
		sums.z_moment = grid_ptr->z_moments(axis, l1, 0);
	}

	return sums;
}

diagnostics_t::diagnostics_t() :
	primary_sum(NF, ZERO), secondary_sum(NF, ZERO), grid_sum(NF, ZERO), outflow_sum(NF, ZERO), l_sum(NDIM, ZERO), field_max(NF,
		-std::numeric_limits<real>::max()), field_min(NF, +std::numeric_limits<real>::max()), gforce_sum(NDIM, ZERO), gtorque_sum(NDIM, ZERO) {
	for (integer d = 0; d != NDIM; ++d) {
		primary_z_moment = secondary_z_moment = z_moment = 0.0;
		roche_vol1 = roche_vol2 = primary_volume = secondary_volume = 0.0;
		primary_com[d] = secondary_com[d] = grid_com[d] = 0.0;
		primary_com_dot[d] = secondary_com_dot[d] = grid_com_dot[d] = 0.0;
	}
}

diagnostics_t& diagnostics_t::operator+=(const diagnostics_t& other) {
	primary_z_moment += other.primary_z_moment;
	secondary_z_moment += other.secondary_z_moment;
	z_moment += other.z_moment;
	for (integer d = 0; d != NDIM; ++d) {
		primary_com[d] *= primary_sum[rho_i];
		secondary_com[d] *= secondary_sum[rho_i];
		grid_com[d] *= grid_sum[rho_i];
		primary_com_dot[d] *= primary_sum[rho_i];
		secondary_com_dot[d] *= secondary_sum[rho_i];
		grid_com_dot[d] *= grid_sum[rho_i];
	}
	for (integer f = 0; f != NF; ++f) {
		grid_sum[f] += other.grid_sum[f];
		primary_sum[f] += other.primary_sum[f];
		secondary_sum[f] += other.secondary_sum[f];
		outflow_sum[f] += other.outflow_sum[f];
		field_max[f] = std::max(field_max[f], other.field_max[f]);
		field_min[f] = std::min(field_min[f], other.field_min[f]);
	}
	for (integer d = 0; d != NDIM; ++d) {
		l_sum[d] += other.l_sum[d];
		gforce_sum[d] += other.gforce_sum[d];
		gtorque_sum[d] += other.gtorque_sum[d];
	}
	if (l1_error.size() < other.l1_error.size()) {
		l1_error.resize(other.l1_error.size(), ZERO);
		l2_error.resize(other.l2_error.size(), ZERO);
	}
	for (std::size_t i = 0; i != l1_error.size(); ++i) {
		l1_error[i] += other.l1_error[i];
	}
	for (std::size_t i = 0; i != l1_error.size(); ++i) {
		l2_error[i] += other.l2_error[i];
	}
	for (integer d = 0; d != NDIM; ++d) {
		primary_com[d] += other.primary_com[d] * other.primary_sum[rho_i];
		secondary_com[d] += other.secondary_com[d] * other.secondary_sum[rho_i];
		grid_com[d] += other.grid_com[d] * other.grid_sum[rho_i];
		primary_com_dot[d] += other.primary_com_dot[d] * other.primary_sum[rho_i];
		secondary_com_dot[d] += other.secondary_com_dot[d] * other.secondary_sum[rho_i];
		grid_com_dot[d] += other.grid_com_dot[d] * other.grid_sum[rho_i];
	}
	for (integer d = 0; d != NDIM; ++d) {
		if (primary_sum[rho_i] > 0.0) {
			primary_com[d] /= primary_sum[rho_i];
			primary_com_dot[d] /= primary_sum[rho_i];
		}
		if (secondary_sum[rho_i] > 0.0) {
			secondary_com[d] /= secondary_sum[rho_i];
			secondary_com_dot[d] /= secondary_sum[rho_i];
		}
		grid_com[d] /= grid_sum[rho_i];
		grid_com_dot[d] /= grid_sum[rho_i];
	}
	roche_vol1 += other.roche_vol1;
	roche_vol2 += other.roche_vol2;
	primary_volume += other.primary_volume;
	secondary_volume += other.secondary_volume;
	return *this;
}

typedef node_server::force_nodes_to_exist_action force_nodes_to_exist_action_type;
HPX_REGISTER_ACTION(force_nodes_to_exist_action_type);

hpx::future<void> node_client::force_nodes_to_exist(std::list<node_location>&& locs) const {
	return hpx::async<typename node_server::force_nodes_to_exist_action>(get_gid(), std::move(locs));
}

void node_server::force_nodes_to_exist(const std::list<node_location>& locs) {
	std::list<hpx::future<void>> futs;
	std::list<node_location> parent_list;
	std::vector<std::list<node_location>> child_lists(NCHILD);
	for (auto& loc : locs) {
		assert(loc != my_location);
		if (loc.is_child_of(my_location)) {
			if (refinement_flag++ == 0 && !parent.empty()) {
				const auto neighbors = my_location.get_neighbors();
				parent.force_nodes_to_exist(std::list<node_location>(neighbors.begin(), neighbors.end())).get();
			}
			if (is_refined) {
				for (auto& ci : geo::octant::full_set()) {
					if (loc.is_child_of(my_location.get_child(ci))) {
						child_lists[ci].push_back(loc);
						break;
					}
				}
			}
		} else {
			assert(!parent.empty());
			parent_list.push_back(loc);
		}
	}
	for (auto& ci : geo::octant::full_set()) {
		if (is_refined && child_lists[ci].size()) {
			futs.push_back(children[ci].force_nodes_to_exist(std::move(child_lists[ci])));
		}
	}
	if (parent_list.size()) {
		futs.push_back(parent.force_nodes_to_exist(std::move(parent_list)));
	}
	for (auto&& fut : futs) {
		fut.get();
	}
}

typedef node_server::form_tree_action form_tree_action_type;
HPX_REGISTER_ACTION(form_tree_action_type);

hpx::future<void> node_client::form_tree(const hpx::id_type& id1, const hpx::id_type& id2, const std::vector<hpx::id_type>& ids) {
	return hpx::async<typename node_server::form_tree_action>(get_gid(), id1, id2, std::move(ids));
}

void node_server::form_tree(const hpx::id_type& self_gid, const hpx::id_type& parent_gid, const std::vector<hpx::id_type>& neighbor_gids) {
	for (auto& dir : geo::direction::full_set()) {
		neighbors[dir] = neighbor_gids[dir];
	}
	for (auto& face : geo::face::full_set()) {
		siblings[face] = neighbors[face.to_direction()];
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
										geo::direction dir = geo::direction((x / 2) + NDIM * ((y / 2) + NDIM * (z / 2)));
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
					cfuts.push_back(children[ci].form_tree(children[ci].get_gid(), me.get_gid(), std::move(child_neighbors)));
				}
			}
		}

		for (auto&& fut : cfuts) {
			fut.get();
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

typedef node_server::get_child_client_action get_child_client_action_type;
HPX_REGISTER_ACTION(get_child_client_action_type);

hpx::future<hpx::id_type> node_client::get_child_client(const geo::octant& ci) {
	if (get_gid() != hpx::invalid_id) {
		return hpx::async<typename node_server::get_child_client_action>(get_gid(), ci);
	} else {
		auto tmp = hpx::invalid_id;
		return hpx::make_ready_future<hpx::id_type>(std::move(tmp));
	}
}

hpx::id_type node_server::get_child_client(const geo::octant& ci) {
	if (is_refined) {
		return children[ci].get_gid();
	} else {
		return hpx::invalid_id;
	}
}

typedef node_server::get_nieces_action get_nieces_action_type;
HPX_REGISTER_ACTION(get_nieces_action_type);

hpx::future<std::vector<hpx::id_type>> node_client::get_nieces(const hpx::id_type& aunt, const geo::face& f) const {
	return hpx::async<typename node_server::get_nieces_action>(get_gid(), aunt, f);
}

std::vector<hpx::id_type> node_server::get_nieces(const hpx::id_type& aunt, const geo::face& face) const {
	std::vector<hpx::id_type> nieces;
	if (is_refined) {
		std::vector<hpx::future<void>> futs;
		nieces.reserve(geo::quadrant::count());
		futs.reserve(geo::quadrant::count());
		for (auto& ci : geo::octant::face_subset(face)) {
			nieces.push_back(children[ci].get_gid());
			futs.push_back(children[ci].set_aunt(aunt, face));
		}
		for (auto&& this_fut : futs) {
			this_fut.get();
		}
	}
	return nieces;
}

typedef node_server::get_ptr_action get_ptr_action_type;
HPX_REGISTER_ACTION(get_ptr_action_type);

std::uintptr_t node_server::get_ptr() {
	return reinterpret_cast<std::uintptr_t>(this);
}

hpx::future<node_server*> node_client::get_ptr() const {
	return hpx::async<typename node_server::get_ptr_action>(get_gid()).then([](hpx::future<std::uintptr_t>&& fut) {
		return reinterpret_cast<node_server*>(fut.get());
	});
}

typedef node_server::send_gravity_boundary_action send_gravity_boundary_action_type;
HPX_REGISTER_ACTION(send_gravity_boundary_action_type);

hpx::future<void> node_client::send_gravity_boundary(gravity_boundary_type&& data, const geo::direction& dir, bool monopole) const {
	return hpx::async<typename node_server::send_gravity_boundary_action>(get_gid(), std::move(data), dir, monopole);
}

void node_server::recv_gravity_boundary(gravity_boundary_type&& bdata, const geo::direction& dir, bool monopole) {
	neighbor_gravity_type tmp;
	tmp.data = std::move(bdata);
	tmp.is_monopole = monopole;
	tmp.direction = dir;
	neighbor_gravity_channels[dir].set_value(std::move(tmp));
}

typedef node_server::send_gravity_expansions_action send_gravity_expansions_action_type;
HPX_REGISTER_ACTION(send_gravity_expansions_action_type);

void node_server::recv_gravity_expansions(expansion_pass_type&& v) {
	parent_gravity_channel.set_value(std::move(v));
}

hpx::future<void> node_client::send_gravity_expansions(expansion_pass_type&& data) const {
	return hpx::async<typename node_server::send_gravity_expansions_action>(get_gid(), std::move(data));
}

typedef node_server::send_gravity_multipoles_action send_gravity_multipoles_action_type;
HPX_REGISTER_ACTION(send_gravity_multipoles_action_type);

hpx::future<void> node_client::send_gravity_multipoles(multipole_pass_type&& data, const geo::octant& ci) const {
	return hpx::async<typename node_server::send_gravity_multipoles_action>(get_gid(), std::move(data), ci);
}

void node_server::recv_gravity_multipoles(multipole_pass_type&& v, const geo::octant& ci) {
	child_gravity_channels[ci].set_value(std::move(v));
}

typedef node_server::send_hydro_boundary_action send_hydro_boundary_action_type;
HPX_REGISTER_ACTION(send_hydro_boundary_action_type);

hpx::future<void> node_client::send_hydro_boundary(std::vector<real>&& data, const geo::direction& dir) const {
	return hpx::async<typename node_server::send_hydro_boundary_action>(get_gid(), std::move(data), dir);
}

void node_server::recv_hydro_boundary(std::vector<real>&& bdata, const geo::direction& dir) {
	sibling_hydro_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_hydro_channels[dir].set_value(std::move(tmp));
}

typedef node_server::send_hydro_children_action send_hydro_children_action_type;
HPX_REGISTER_ACTION(send_hydro_children_action_type);

void node_server::recv_hydro_children(std::vector<real>&& data, const geo::octant& ci) {
	child_hydro_channels[ci].set_value(std::move(data));
}

hpx::future<void> node_client::send_hydro_children(std::vector<real>&& data, const geo::octant& ci) const {
	return hpx::async<typename node_server::send_hydro_children_action>(get_gid(), std::move(data), ci);
}

typedef node_server::send_hydro_flux_correct_action send_hydro_flux_correct_action_type;
HPX_REGISTER_ACTION(send_hydro_flux_correct_action_type);

hpx::future<void> node_client::send_hydro_flux_correct(std::vector<real>&& data, const geo::face& face,
const geo::octant& ci) const {
	return hpx::async<typename node_server::send_hydro_flux_correct_action>(get_gid(), std::move(data), face, ci);
}

void node_server::recv_hydro_flux_correct(std::vector<real>&& data, const geo::face& face, const geo::octant& ci) {
	const geo::quadrant index(ci, face.get_dimension());
	niece_hydro_channels[face][index].set_value(std::move(data));
}

typedef node_server::line_of_centers_action line_of_centers_action_type;
HPX_REGISTER_ACTION(line_of_centers_action_type);

hpx::future<line_of_centers_t> node_client::line_of_centers(const std::pair<space_vector, space_vector>& line) const {
	return hpx::async<typename node_server::line_of_centers_action>(get_gid(), line);
}

void output_line_of_centers(FILE* fp, const line_of_centers_t& loc) {
	for (integer i = 0; i != loc.size(); ++i) {
		fprintf(fp, "%e ", loc[i].first);
		for (integer j = 0; j != NF + NGF; ++j) {
			fprintf(fp, "%e ", loc[i].second[j]);
		}
		fprintf(fp, "\n");
	}
}

line_of_centers_t node_server::line_of_centers(const std::pair<space_vector, space_vector>& line) const {
	std::list < hpx::future < line_of_centers_t >> futs;
	line_of_centers_t return_line;
	if (is_refined) {
		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].line_of_centers(line));
		}
		std::map < real, std::vector < real >> map;
		for (auto&& fut : futs) {
			auto tmp = fut.get();
			for (integer ii = 0; ii != tmp.size(); ++ii) {
				if( map.find(tmp[ii].first) == map.end()) {
					map.emplace(std::move(tmp[ii]));
				}
			}
		}
		return_line.resize(map.size());
		std::move(map.begin(), map.end(), return_line.begin());
	} else {
		return_line = grid_ptr->line_of_centers(line);
	}

	return return_line;
}

void line_of_centers_analyze(const line_of_centers_t& loc, real omega, std::pair<real, real>& rho1_max, std::pair<real, real>& rho2_max,
	std::pair<real, real>& l1_phi, std::pair<real, real>& l2_phi, std::pair<real, real>& l3_phi, real& rho1_phi, real& rho2_phi) {

	for (auto& l : loc) {
		ASSERT_NONAN(l.first);
		for (integer f = 0; f != NF + NGF; ++f) {
			ASSERT_NONAN(l.second[f]);
		}
	}

	rho1_max.second = rho2_max.second = 0.0;
	integer rho1_maxi, rho2_maxi;
	///	printf( "LOCSIZE %i\n", loc.size());
	for (integer i = 0; i != loc.size(); ++i) {
		const real x = loc[i].first;
		const real rho = loc[i].second[rho_i];
		const real pot = loc[i].second[pot_i];
		//	printf( "%e %e\n", x, rho);
		if (rho1_max.second < rho) {
			//	printf( "!\n");
			rho1_max.second = rho;
			rho1_max.first = x;
			rho1_maxi = i;
			real phi_eff = pot / ASSERT_POSITIVE(rho) - 0.5 * x * x * omega * omega;
			rho1_phi = phi_eff;
		}
	}
	for (integer i = 0; i != loc.size(); ++i) {
		const real x = loc[i].first;
		if (x * rho1_max.first < 0.0) {
			const real rho = loc[i].second[rho_i];
			const real pot = loc[i].second[pot_i];
			if (rho2_max.second < rho) {
				rho2_max.second = rho;
				rho2_max.first = x;
				rho2_maxi = i;
				real phi_eff = pot / ASSERT_POSITIVE(rho) - 0.5 * x * x * omega * omega;
				rho2_phi = phi_eff;
			}
		}
	}
	l1_phi.second = -std::numeric_limits<real>::max();
	l2_phi.second = -std::numeric_limits<real>::max();
	l3_phi.second = -std::numeric_limits<real>::max();
	for (integer i = 0; i != loc.size(); ++i) {
		const real x = loc[i].first;
		const real rho = loc[i].second[rho_i];
		const real pot = loc[i].second[pot_i];
		real phi_eff = pot / ASSERT_POSITIVE(rho) - 0.5 * x * x * omega * omega;
		if (x > std::min(rho1_max.first, rho2_max.first) && x < std::max(rho1_max.first, rho2_max.first)) {
			if (phi_eff > l1_phi.second) {
				l1_phi.second = phi_eff;
				l1_phi.first = x;
			}
		} else if (std::abs(x) > std::abs(rho2_max.first) && x * rho2_max.first > 0.0) {
			if (phi_eff > l2_phi.second) {
				l2_phi.second = phi_eff;
				l2_phi.first = x;
			}
		} else if (std::abs(x) > std::abs(rho1_max.first)) {
			if (phi_eff > l3_phi.second) {
				l3_phi.second = phi_eff;
				l3_phi.first = x;
			}
		}
	}
}

