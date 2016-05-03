/*
 * diagnostics.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */

#include "../diagnostics.hpp"
#include "../node_server.hpp"
#include "../node_client.hpp"
#include "../options.hpp"

extern options opts;

typedef node_server::diagnostics_action diagnostics_action_type;
HPX_REGISTER_ACTION (diagnostics_action_type);

hpx::future<diagnostics_t> node_client::diagnostics(const std::pair<space_vector, space_vector>& axis,
		const std::pair<real, real>& l1) const {
	return hpx::async<typename node_server::diagnostics_action>(get_gid(), axis, l1);
}

diagnostics_t node_server::diagnostics() const {
	auto axis = grid_ptr->find_axis();
	auto loc = line_of_centers(axis);
	real this_omega = find_omega();
	std::pair<real, real> rho1, rho2, l1;
	real dummy;
	line_of_centers_analyze(loc, this_omega, rho1, rho2, l1, dummy, dummy);
	if( rho1 > rho2 ) {
		for( integer d = 0; d != NDIM; ++d ) {
			//printf( "Flipping axis\n" );
			axis.first[d] = -axis.first[d];
			loc = line_of_centers(axis);
			line_of_centers_analyze(loc, this_omega, rho1, rho2, l1, dummy, dummy);
		}
	}
	auto diags = diagnostics(axis, l1);
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

diagnostics_t node_server::diagnostics(const std::pair<space_vector, space_vector>& axis,
		const std::pair<real, real>& l1) const {

	diagnostics_t sums;
	if (is_refined) {
		std::list<hpx::future<diagnostics_t>> futs;
		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].diagnostics(axis, l1));
		}
		for (auto ci = futs.begin(); ci != futs.end(); ++ci) {
			auto this_sum = GET(*ci);
			sums += this_sum;
		}
	} else {
		sums.primary_sum = grid_ptr->conserved_sums(sums.primary_com, sums.primary_com_dot, axis, l1, +1);
		sums.secondary_sum = grid_ptr->conserved_sums(sums.secondary_com, sums.secondary_com_dot, axis, l1, -1);
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
	}

	return sums;
}

diagnostics_t::diagnostics_t() :
		primary_sum(NF, ZERO), secondary_sum(NF, ZERO), grid_sum(NF, ZERO), outflow_sum(NF, ZERO), l_sum(NDIM, ZERO), field_max(
				NF, -std::numeric_limits < real > ::max()), field_min(NF, +std::numeric_limits < real > ::max()), gforce_sum(
				NDIM, ZERO), gtorque_sum(NDIM, ZERO) {
	for (integer d = 0; d != NDIM; ++d) {
		primary_com[d] = secondary_com[d] = grid_com[d] = 0.0;
		primary_com_dot[d] = secondary_com_dot[d] = grid_com_dot[d] = 0.0;
	}
}

diagnostics_t& diagnostics_t::operator+=(const diagnostics_t& other) {
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
	return *this;
}

