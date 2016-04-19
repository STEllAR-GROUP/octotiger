/*
 * diagnostics.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */



#include "../node_server.hpp"
#include "../node_client.hpp"
#include "../options.hpp"

extern options opts;


typedef node_server::diagnostics_action diagnostics_action_type;
HPX_REGISTER_ACTION (diagnostics_action_type);


hpx::future<diagnostics_t> node_client::diagnostics() const {
	return hpx::async<typename node_server::diagnostics_action>(get_gid());
}


diagnostics_t node_server::diagnostics() const {
	diagnostics_t sums;
	if (is_refined) {
		std::list<hpx::future<diagnostics_t>> futs;
		for (integer ci = 0; ci != NCHILD; ++ci) {
			futs.push_back(children[ci].diagnostics());
		}
		for (auto ci = futs.begin(); ci != futs.end(); ++ci) {
			auto this_sum = GET(*ci);
			sums += this_sum;
		}
	} else {
		sums.grid_sum = grid_ptr->conserved_sums();
		sums.outflow_sum = grid_ptr->conserved_outflows();
		sums.donor_mass = grid_ptr->conserved_sums([](real x, real, real) {return x > 0.09;})[rho_i];
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

	if (my_location.level() == 0) {
		if (opts.problem != SOLID_SPHERE) {
			auto diags = sums;
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

			fp = fopen("minmax.dat", "at");
			fprintf(fp, "%23.16e ", double(current_time));
			for (integer f = 0; f != NF; ++f) {
				fprintf(fp, "%23.16e ", double(diags.field_min[f]));
				fprintf(fp, "%23.16e ", double(diags.field_max[f]));
			}
			fprintf(fp, "\n");
			fclose(fp);

			auto com = grid_ptr->center_of_mass();
			fp = fopen("com.dat", "at");
			fprintf(fp, "%23.16e ", double(current_time));
			for (integer d = 0; d != NDIM; ++d) {
				fprintf(fp, "%23.16e ", double(com[d]));
			}
			fprintf(fp, "\n");
			fclose(fp);

			fp = fopen("m_don.dat", "at");
			fprintf(fp, "%23.16e ", double(current_time));
			fprintf(fp, "%23.16e ", double(diags.grid_sum[rho_i] - diags.donor_mass));
			fprintf(fp, "%23.16e ", double(diags.donor_mass));
			fprintf(fp, "\n");
			fclose(fp);
		} else {
			printf("L1\n");
			printf("Gravity Phi Error - %e\n", (sums.l1_error[0] / sums.l1_error[4]));
			printf("Gravity gx Error - %e\n", (sums.l1_error[1] / sums.l1_error[5]));
			printf("Gravity gy Error - %e\n", (sums.l1_error[2] / sums.l1_error[6]));
			printf("Gravity gz Error - %e\n", (sums.l1_error[3] / sums.l1_error[7]));
			printf("L2\n");
			printf("Gravity Phi Error - %e\n", std::sqrt(sums.l2_error[0] / sums.l2_error[4]));
			printf("Gravity gx Error - %e\n", std::sqrt(sums.l2_error[1] / sums.l2_error[5]));
			printf("Gravity gy Error - %e\n", std::sqrt(sums.l2_error[2] / sums.l2_error[6]));
			printf("Gravity gz Error - %e\n", std::sqrt(sums.l2_error[3] / sums.l2_error[7]));
			printf("Total Mass = %e\n", sums.grid_sum[rho_i]);
			for (integer d = 0; d != NDIM; ++d) {
				printf("%e %e\n", sums.gforce_sum[d], sums.gtorque_sum[d]);
			}
		}
	}

	return sums;
}

