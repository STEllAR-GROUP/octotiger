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
#include "octotiger/real.hpp"

#include <hpx/include/naming.hpp>

#include <cstddef>
#include <string>
#include <vector>

/* Must look like this - no spaces
 COMMAND_LINE_ENUM(problem_type,DWD,SOD,BLAST,NONE,SOLID_SPHERE,STAR,MOVING_STAR,RADIATION_TEST,ROTATING_STAR,MARSHAK,AMR_TEST);

 COMMAND_LINE_ENUM(eos_type,IDEAL,WD);
 */

COMMAND_LINE_ENUM(problem_type, DWD, SOD, BLAST, NONE, SOLID_SPHERE, STAR, MOVING_STAR, RADIATION_TEST, ROTATING_STAR, MARSHAK, AMR_TEST, ADVECTION);

COMMAND_LINE_ENUM(eos_type, IDEAL, WD);

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
	bool hydro;
	bool radiation;
	real grad_rho_refine;
	real clight_retard;
	bool v1309;
	bool rad_implicit;
	bool rewrite_silo;
	bool correct_am_grav;
	bool correct_am_hydro;
	bool rotating_star_amr;
	bool idle_rates;

	integer scf_output_frequency;
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

	size_t cuda_number_gpus;
	size_t cuda_streams_per_gpu;
	size_t cuda_buffer_capacity;
	bool root_node_on_device;

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
	std::vector<real> X;
	std::vector<real> Z;

	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & eblast0;
		arc & rho_floor;
		arc & tau_floor;
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
		arc & cuda_number_gpus;
		arc & cuda_streams_per_gpu;
		arc & cuda_buffer_capacity;
	  arc & root_node_on_device;
		arc & atomic_mass;
		arc & atomic_number;
		arc & X;
		arc & Z;
		arc & grad_rho_refine;
		arc & code_to_g;
		arc & code_to_s;
		arc & code_to_cm;
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
