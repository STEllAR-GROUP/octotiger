//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/options_enum.hpp"

#include <hpx/include/naming.hpp>

#include <cstddef>
#include <string>
#include <vector>

/* Must look like this - no spaces
 COMMAND_LINE_ENUM(problem_type,DWD,SOD,BLAST,NONE,SOLID_SPHERE,STAR,MOVING_STAR,RADIATION_TEST,ROTATING_STAR,MARSHAK,AMR_TEST);

 COMMAND_LINE_ENUM(eos_type,IDEAL,WD);
 */

COMMAND_LINE_ENUM(problem_type, DWD, SOD, BLAST, NONE, SOLID_SPHERE, STAR, MOVING_STAR, RADIATION_TEST, ROTATING_STAR, MARSHAK, AMR_TEST,
				  ADVECTION, RADIATION_DIFFUSION, RADIATION_COUPLING);
COMMAND_LINE_ENUM(eos_type, IDEAL, WD, IPR);

class options {
public:
	bool inflow_bc;
	bool reflect_bc;
	bool cdisc_detect;
	bool unigrid;
	bool disable_diagnostics;
	bool bench;
	bool disable_output;
	bool disable_analytic;
	bool core_refine;
	bool gravity;
	bool hydro;
	bool periodic;
	bool radiation;
	bool v1309;
	bool rad_implicit;
	bool rewrite_silo;
	bool correct_am_grav;
	bool correct_am_hydro;
	bool rotating_star_amr;
	bool idle_rates;
	bool ipr_test;
	bool detected_intel_compiler;
	bool print_times_per_timestep;

	int experiment;
	int scf_output_frequency;
	int silo_num_groups;
	int amrbnd_order;
	int extra_regrid;
	int accretor_refine;
	int donor_refine;
	int min_level;
	int max_level;
	int ngrids;
	int stop_step;
	int silo_offset_x;
	int silo_offset_y;
	int silo_offset_z;
	int future_wait_time;
	int ipr_nr_maxiter;
	int rad_diff_ndim;
	double rad_diff_Er0;
	double rad_diff_t0;
	double rad_diff_D;

	double grad_rho_refine;
	double clight_retard;
	double dt_max;
	double eblast0;
	double rotating_star_x;
	double dual_energy_sw2;
	double dual_energy_sw1;
	double hard_dt;
	double driving_rate;
	double driving_time;
	double entropy_driving_rate;
	double entropy_driving_time;
	double omega;
	double output_dt;
	double refinement_floor;
	double stop_time;
	double theta;
	double xscale;
	double code_to_g;
	double code_to_s;
	double code_to_cm;
	double cfl;
	double rho_floor;
	double tau_floor;
	double scf_rho_floor;
	double ipr_eint_floor;
	double ipr_nr_tol;

	double sod_rhol;
	double sod_rhor;
	double sod_pl;
	double sod_pr;
	double sod_theta;
	double sod_phi;
	double sod_gamma;

	double solid_sphere_xcenter;
	double solid_sphere_ycenter;
	double solid_sphere_zcenter;
	double solid_sphere_radius;
	double solid_sphere_mass;
	double solid_sphere_rho_min;

	double star_xcenter;
	double star_ycenter;
	double star_zcenter;
	double star_rmax;
	double star_alpha;
	double star_rho_out;
	double star_egas_out;
	double star_dr;
	double star_n;
	double star_rho_center;

	double moving_star_xvelocity;
	double moving_star_yvelocity;
	double moving_star_zvelocity;

	size_t number_gpus;
	size_t executors_per_gpu;
	size_t max_gpu_executor_queue_length;
	size_t max_kernels_fused;

	bool root_node_on_device;
	bool optimize_local_communication;
	int polling_threads;

	std::string input_file;
	std::string config_file;
	std::string data_dir;
	std::string output_filename;
	std::string restart_filename;
	int n_species;
	int n_fields;

	eos_type eos;

	problem_type problem;

	amr_boundary_type amr_boundary_kernel_type;
	interaction_host_kernel_type multipole_host_kernel_type;
	interaction_device_kernel_type multipole_device_kernel_type;
	interaction_host_kernel_type monopole_host_kernel_type;
	interaction_device_kernel_type monopole_device_kernel_type;
	interaction_host_kernel_type hydro_host_kernel_type;
	interaction_device_kernel_type hydro_device_kernel_type;

	std::vector<double> atomic_mass;
	std::vector<double> atomic_number;

	template <class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & eblast0;
		arc & rho_floor;
		arc & tau_floor;
		arc & scf_rho_floor;
		arc & ipr_eint_floor;
		arc & ipr_nr_tol;
		arc & ipr_test;
		arc & ipr_nr_maxiter;
		arc & sod_rhol;
		arc & sod_rhor;
		arc & sod_pl;
		arc & sod_pr;
		arc & sod_theta;
		arc & sod_phi;
		arc & sod_gamma;
		arc & solid_sphere_xcenter;
		arc & solid_sphere_ycenter;
		arc & solid_sphere_zcenter;
		arc & solid_sphere_radius;
		arc & solid_sphere_mass;
		arc & solid_sphere_rho_min;
		arc & star_xcenter;
		arc & star_ycenter;
		arc & star_zcenter;
		arc & star_rmax;
		arc & star_alpha;
		arc & star_dr;
		arc & star_n;
		arc & star_rho_center;
		arc & star_rho_out;
		arc & star_egas_out;
		arc & moving_star_xvelocity;
		arc & moving_star_yvelocity;
		arc & moving_star_zvelocity;
		arc & inflow_bc;
		arc & reflect_bc;
		arc & cdisc_detect;
		arc & experiment;
		arc & unigrid;
		arc & rotating_star_amr;
		arc & rotating_star_x;
		arc & future_wait_time;
		arc & silo_offset_x;
		arc & silo_offset_y;
		arc & silo_offset_z;
		arc & scf_output_frequency;
		arc & silo_num_groups;
		arc & amrbnd_order;
		arc & dual_energy_sw1;
		arc & dual_energy_sw2;
		arc & hard_dt;
		arc & correct_am_grav;
		arc & correct_am_hydro;
		arc & rewrite_silo;
		arc & rad_implicit;
		arc & n_fields;
		arc & n_species;
		arc & input_file;
		arc & config_file;
		arc & hydro;
		arc & periodic;
		arc & gravity;
		arc & bench;
		arc & radiation;
		arc & multipole_host_kernel_type;
		arc & multipole_device_kernel_type;
		arc & monopole_host_kernel_type;
		arc & monopole_device_kernel_type;
		arc & hydro_host_kernel_type;
		arc & hydro_device_kernel_type;
		arc & entropy_driving_rate;
		arc & entropy_driving_time;
		arc & driving_rate;
		arc & driving_time;
		arc & refinement_floor;
		arc & ngrids;
		arc & v1309;
		arc & clight_retard;
		arc & stop_time;
		arc & min_level;
		arc & max_level;
		arc & xscale;
		arc & dt_max;
		arc & cfl;
		arc & omega;
		arc & restart_filename;
		arc & output_filename;
		arc & output_dt;
		arc & stop_step;
		arc & disable_diagnostics;
		arc & disable_output;
		arc & disable_analytic;
		arc & theta;
		arc & core_refine;
		arc & donor_refine;
		arc & extra_regrid;
		arc & accretor_refine;
		arc & idle_rates;
		int tmp = problem;
		arc & tmp;
		problem = static_cast<problem_type>(tmp);
		tmp = eos;
		arc & tmp;
		eos = static_cast<eos_type>(tmp);
		arc & data_dir;
		arc & number_gpus;
		arc & executors_per_gpu;
		arc & max_gpu_executor_queue_length;
		arc & max_kernels_fused;
		arc & root_node_on_device;
		arc & optimize_local_communication;
		arc & polling_threads;
		arc & detected_intel_compiler;
		arc & print_times_per_timestep;
		arc & atomic_mass;
		arc & atomic_number;
		arc & grad_rho_refine;
		arc & code_to_g;
		arc & code_to_s;
		arc & code_to_cm;
		arc & rad_diff_Er0;
		arc & rad_diff_t0;
		arc & rad_diff_D;
	}

	OCTOTIGER_EXPORT bool process_options(int argc, char *argv[]);

	static OCTOTIGER_EXPORT std::vector<hpx::id_type> all_localities;
};

OCTOTIGER_EXPORT options &opts();

template <class T = double>
struct hydro_state_t : public std::vector<T> {
	hydro_state_t() :
		std::vector<T>(opts().n_fields) {
	}
};

#endif /* OPTIONS_HPP_ */
