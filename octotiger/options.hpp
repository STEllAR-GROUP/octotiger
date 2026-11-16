//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include <hpx/include/naming.hpp>

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/options_enum.hpp"
#include "octotiger/real.hpp"

/* Must look like this - no spaces
 COMMAND_LINE_ENUM(problem_type,DWD,SOD,BLAST,NONE,SOLID_SPHERE,STAR,MOVING_STAR,RADIATION_TEST,ROTATING_STAR,MARSHAK,AMR_TEST);

 COMMAND_LINE_ENUM(eos_type,IDEAL,WD);
 */

COMMAND_LINE_ENUM(problem_type, DWD, SOD, BLAST, NONE, SOLID_SPHERE, STAR, MOVING_STAR, RADIATION_TEST, ROTATING_STAR, MARSHAK, AMR_TEST, ADVECTION, RADIATION_DIFFUSION, RADIATION_COUPLING);

COMMAND_LINE_ENUM(eos_type, IDEAL, WD, IPR);

class options {
public:
	bool inflow_bc;
	bool reflect_bc;
	int experiment;
	bool cdisc_detect;
	bool unigrid;
	bool disable_diagnostics;
	bool bench;
	bool disable_output;
	bool disable_analytic;
	bool core_refine;
	bool gravity;
	// Run-only diagnostics, intentionally absent from checkpoint serialization.
	bool gravity_compare = false;
	bool gravity_compare_output = false;
	integer gravity_mg_max_cycles = 100;
	real gravity_mg_tolerance = 1.0e-9;
	std::string gravity_mg_boundary = "analytic";
	bool hydro;
	bool radiation;
	// Independent comoving FLD model; coefficients and energies are in code units.
	bool fld = false;
	std::string fld_opacity_model = "legacy";
	std::string fld_material_model = "auto";
	std::string fld_unit_system = "auto";
	real fld_rosseland = 1.0;
	real fld_rosseland_density_exponent = 0.0;
	real fld_rosseland_temperature_exponent = 0.0;
	real fld_opacity_temperature_floor = 0.0;
	real fld_gaunt = 1.0;
	real fld_absorption = 1.0;
	real fld_scattering = 0.0;
	real fld_density_exponent = 0.0;
	real fld_temperature_exponent = 0.0;
	real fld_cv = 1.0;
	real fld_c = 1.0;
	real fld_ar = 1.0;
	real fld_initial_energy = 0.0;
	real fld_initial_pulse = 0.0;
	real fld_tolerance = 1.0e-8;
	integer fld_max_iterations = 100;
	integer fld_max_cycles = 100;

	real grad_rho_refine;
	real clight_retard;
	bool v1309;
	bool rad_implicit;
	bool rewrite_silo;
	bool correct_am_grav;
	// Retained for legacy configuration and serialized-option compatibility.
	bool correct_am_hydro;
	bool rotating_star_amr;
	bool idle_rates;
	bool ipr_test;
  bool detected_intel_compiler;
  bool print_times_per_timestep;

	integer scf_output_frequency;
	integer scf_max_iterations = 256;
	integer silo_num_groups;
	integer amrbnd_order;
	integer extra_regrid;
	integer accretor_refine;
	integer donor_refine;
	integer min_level;
	integer max_level;
	integer ngrids;
	integer stop_step;
	integer silo_offset_x;
	integer silo_offset_y;
	integer silo_offset_z;
	integer future_wait_time;
	integer ipr_nr_maxiter;

	real dt_max;
	real eblast0;
	real rotating_star_x;
	real dual_energy_sw2;
	real dual_energy_sw1;
	real hard_dt;
	real driving_rate;
	real driving_time;
	real entropy_driving_rate;
	real entropy_driving_time;
	real omega;
	real output_dt;
	real refinement_floor;
	real stop_time;
	real theta;
	real xscale;
	real code_to_g;
	real code_to_s;
	real code_to_cm;
	real cfl;
	real rho_floor;
	real tau_floor;
	real scf_rho_floor;
	real ipr_eint_floor;
	real ipr_nr_tol;

	real sod_rhol;
	real sod_rhor;
	real sod_pl;
	real sod_pr;
	real sod_theta;
	real sod_phi;
	real sod_gamma;

	real solid_sphere_xcenter;
	real solid_sphere_ycenter;
	real solid_sphere_zcenter;
	real solid_sphere_radius;
	real solid_sphere_mass;
	real solid_sphere_rho_min;

	real star_xcenter;
	real star_ycenter;
	real star_zcenter;
	real star_rmax;
	real star_alpha;
	real star_rho_out;
	real star_egas_out;
	real star_dr;
	real star_n;
	real star_rho_center;

	real moving_star_xvelocity;
        real moving_star_yvelocity;
        real moving_star_zvelocity;

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
	integer n_species;
	integer n_fields;

	eos_type eos;

	problem_type problem;

	amr_boundary_type amr_boundary_kernel_type;
	interaction_host_kernel_type multipole_host_kernel_type;
	interaction_device_kernel_type multipole_device_kernel_type;
	interaction_host_kernel_type monopole_host_kernel_type;
	interaction_device_kernel_type monopole_device_kernel_type;
	interaction_host_kernel_type hydro_host_kernel_type;
	interaction_device_kernel_type hydro_device_kernel_type;

	std::vector<real> atomic_mass;
	std::vector<real> atomic_number;
	std::vector<real> atomic_number_squared;
	std::vector<real> X;
	std::vector<real> Z;

	template<class Arc>
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
		arc & scf_max_iterations;
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
		arc & atomic_number_squared;
		arc & X;
		arc & Z;
		arc & grad_rho_refine;
		arc & code_to_g;
		arc & code_to_s;
		arc & code_to_cm;
		arc & fld;
		arc & fld_opacity_model;
		arc & fld_material_model;
		arc & fld_unit_system;
		arc & fld_rosseland;
		arc & fld_rosseland_density_exponent;
		arc & fld_rosseland_temperature_exponent;
		arc & fld_opacity_temperature_floor;
		arc & fld_gaunt;
		arc & fld_absorption;
		arc & fld_scattering;
		arc & fld_density_exponent;
		arc & fld_temperature_exponent;
		arc & fld_cv;
		arc & fld_c;
		arc & fld_ar;
		arc & fld_initial_energy;
		arc & fld_initial_pulse;
		arc & fld_tolerance;
		arc & fld_max_iterations;
		arc & fld_max_cycles;

	}

	OCTOTIGER_EXPORT bool process_options(int argc, char *argv[]);

	static OCTOTIGER_EXPORT std::vector<hpx::id_type> all_localities;
};

OCTOTIGER_EXPORT options& opts();

template<class T = real>
struct hydro_state_t: public std::vector<T> {
	hydro_state_t() :
			std::vector<T>(opts().n_fields) {
	}
};

#endif /* OPTIONS_HPP_ */
