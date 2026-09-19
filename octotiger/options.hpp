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
#include "octotiger/math/Real.hpp"

#include <hpx/include/naming.hpp>

#include <cstddef>
#include <string>
#include <vector>

/* Must look like this - no spaces
 COMMAND_LINE_ENUM(problem_type,DWD,SOD,BLAST,NONE,SOLID_SPHERE,STAR,MOVING_STAR,RADIATION_TEST,ROTATING_STAR,MARSHAK,AMR_TEST);

 COMMAND_LINE_ENUM(eos_type,IDEAL,WD);
 */

COMMAND_LINE_ENUM(problem_type, DWD, SOD, BLAST, NONE, SOLID_SPHERE, STAR, MOVING_STAR, RADIATION_TEST, ROTATING_STAR, MARSHAK, AMR_TEST, ADVECTION, RADIATION_DIFFUSION, RADIATION_COUPLING, RADIATION_STREAMING_WAVE, RADIATION_STREAMING_FRONT, RADIATION_GAUSSIAN_PULSE, RADIATION_EQUILIBRIUM_SPHERE);

COMMAND_LINE_ENUM(eos_type, IDEAL, WD, IPR);

// A lightweight reference used by the hierarchical option views below.  The
// existing flat data members remain the serialized storage, so checkpoints and
// HPX wire archives keep their exact historical layout.
template <class T>
class option_reference {
public:
	explicit option_reference(T& value) : value_(&value) {
	}
	operator T&() const {
		return *value_;
	}
	T& get() const {
		return *value_;
	}
	option_reference& operator=(T const& value) {
		*value_ = value;
		return *this;
	}
	option_reference& operator=(option_reference const& other) {
		*value_ = other.get();
		return *this;
	}

private:
	T* value_;
};

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
	bool periodic;
	bool radiation;
	Real grad_rho_refine;
	bool v1309;
	bool rad_implicit;
	// S&O radiation controls; independent of the hydrodynamic RK integrator.
	bool rad_subcycling = true;
	Real rad_c_ratio = 1.0;
	Real rad_cfl = 0.4;
	integer rad_max_subcycles = 1024;
	Real rad_theta = 1.0;
	bool rad_velocity_terms = true;
	Real rad_opacity = -1.0;
	std::string rad_energy_mode = "thermal";
	bool rad_log_subcycles = false;

	// Prescribed-medium regression parameters, all in physical code units.
	std::string radReference = "gaussian_pulse.bin";
	Real radTestChi = 5, radTestWidth = .2;
	Real radTestBackground = 1, radTestAmplitude = .01, radTestLuminosity = .01;
	bool rewrite_silo;
	bool correct_am_grav;
	bool correct_am_hydro;
	bool rotating_star_amr;
	bool idle_rates;
	bool ipr_test;
  bool detected_intel_compiler;
  bool print_times_per_timestep;

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
	integer ipr_nr_maxiter;

	Real dt_max;
	Real eblast0;
	Real rotating_star_x;
	Real dual_energy_sw2;
	Real dual_energy_sw1;
	Real hard_dt;
	Real driving_rate;
	Real driving_time;
	Real entropy_driving_rate;
	Real entropy_driving_time;
	Real omega;
	Real output_dt;
	Real refinement_floor;
	Real stop_time;
	Real theta;
	Real xscale;
	Real code_to_g;
	Real code_to_s;
	Real code_to_cm;
	Real cfl;
	Real rho_floor;
	Real tau_floor;
	Real scf_rho_floor;
	Real ipr_eint_floor;
	Real ipr_nr_tol;

	Real sod_rhol;
	Real sod_rhor;
	Real sod_pl;
	Real sod_pr;
	Real sod_theta;
	Real sod_phi;
	Real sod_gamma;

	Real solid_sphere_xcenter;
	Real solid_sphere_ycenter;
	Real solid_sphere_zcenter;
	Real solid_sphere_radius;
	Real solid_sphere_mass;
	Real solid_sphere_rho_min;

	Real star_xcenter;
	Real star_ycenter;
	Real star_zcenter;
	Real star_rmax;
	Real star_alpha;
	Real star_rho_out;
	Real star_egas_out;
	Real star_dr;
	Real star_n;
	Real star_rho_center;

	Real moving_star_xvelocity;
        Real moving_star_yvelocity;
        Real moving_star_zvelocity;

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

	std::vector<Real> atomic_mass;
	std::vector<Real> atomic_number;
	std::vector<Real> X;
	std::vector<Real> Z;

	// Canonical hierarchical C++ views.  Boost.Program_options still sees flat
	// dotted strings; these structures supply hierarchy to C++ without moving or
	// duplicating the legacy serialized storage.
	struct radiation_group {
		struct opacity_group {
			option_reference<Real> constant;
			explicit opacity_group(Real& value) : constant(value) {
			}
		} opacity;
		option_reference<bool> enabled;
		option_reference<bool> implicit;
		option_reference<bool> subcycling;
		option_reference<Real> reduced_light_speed_ratio;
		option_reference<Real> cfl;
		option_reference<integer> max_subcycles;
		option_reference<Real> source_theta;
		option_reference<bool> velocity_terms;
		option_reference<std::string> energy_mode;
		option_reference<bool> log_subcycles;
		radiation_group(options& owner) : opacity(owner.rad_opacity),
			enabled(owner.radiation), implicit(owner.rad_implicit),
			subcycling(owner.rad_subcycling), reduced_light_speed_ratio(owner.rad_c_ratio),
			cfl(owner.rad_cfl), max_subcycles(owner.rad_max_subcycles),
			source_theta(owner.rad_theta), velocity_terms(owner.rad_velocity_terms),
			energy_mode(owner.rad_energy_mode), log_subcycles(owner.rad_log_subcycles) {
		}
	} radiation_options;

	struct hydro_group {
		option_reference<bool> enabled;
		option_reference<Real> gamma;
		option_reference<Real> cfl;
		option_reference<eos_type> eos;
		option_reference<Real> density_floor;
		option_reference<Real> entropy_floor;
		option_reference<bool> periodic;
		hydro_group(options& owner) : enabled(owner.hydro), gamma(owner.sod_gamma),
			cfl(owner.cfl), eos(owner.eos), density_floor(owner.rho_floor),
			entropy_floor(owner.tau_floor), periodic(owner.periodic) {
		}
	} hydro_options;

	struct gravity_group {
		option_reference<bool> enabled;
		option_reference<Real> opening_angle;
		option_reference<Real> angular_frequency;
		option_reference<bool> angular_momentum_correction;
		gravity_group(options& owner) : enabled(owner.gravity), opening_angle(owner.theta),
			angular_frequency(owner.omega), angular_momentum_correction(owner.correct_am_grav) {
		}
	} gravity_options;

	struct problem_group {
		struct blast_group {
			option_reference<Real> energy;
			explicit blast_group(Real& value) : energy(value) {
			}
		} blast;
		option_reference<problem_type> name;
		option_reference<std::string> input_file;
		option_reference<int> experiment;
		problem_group(options& owner) : blast(owner.eblast0), name(owner.problem),
			input_file(owner.input_file), experiment(owner.experiment) {
		}
	} problem_options;

	struct output_group {
		option_reference<std::string> directory;
		option_reference<std::string> filename;
		option_reference<Real> interval;
		option_reference<bool> disabled;
		output_group(options& owner) : directory(owner.data_dir), filename(owner.output_filename),
			interval(owner.output_dt), disabled(owner.disable_output) {
		}
	} output_options;

	struct mesh_group {
		option_reference<Real> scale;
		option_reference<integer> minimum_level;
		option_reference<integer> maximum_level;
		option_reference<bool> unigrid;
		mesh_group(options& owner) : scale(owner.xscale), minimum_level(owner.min_level),
			maximum_level(owner.max_level), unigrid(owner.unigrid) {
		}
	} mesh_options;

	struct restart_group {
		option_reference<std::string> filename;
		explicit restart_group(options& owner) : filename(owner.restart_filename) {
		}
	} restart_options;

	struct units_group {
		option_reference<Real> grams;
		option_reference<Real> centimeters;
		option_reference<Real> seconds;
		units_group(options& owner) : grams(owner.code_to_g), centimeters(owner.code_to_cm),
			seconds(owner.code_to_s) {
		}
	} units_options;

	struct blast_group {
		option_reference<Real> energy;
		explicit blast_group(options& owner) : energy(owner.eblast0) {
		}
	} blast_options;

	struct timestep_group {
		option_reference<Real> maximum_change;
		option_reference<Real> fixed;
		timestep_group(options& owner) : maximum_change(owner.dt_max), fixed(owner.hard_dt) {
		}
	} timestep_options;

	struct runtime_group {
		option_reference<std::string> config_file;
		option_reference<Real> stop_time;
		option_reference<integer> stop_step;
		runtime_group(options& owner) : config_file(owner.config_file),
			stop_time(owner.stop_time), stop_step(owner.stop_step) {
		}
	} runtime_options;

	struct execution_group {
		option_reference<size_t> gpu_count;
		option_reference<size_t> executors_per_gpu;
		option_reference<int> polling_threads;
		execution_group(options& owner) : gpu_count(owner.number_gpus),
			executors_per_gpu(owner.executors_per_gpu), polling_threads(owner.polling_threads) {
		}
	} execution_options;

	options() : radiation_options(*this), hydro_options(*this), gravity_options(*this),
		problem_options(*this), output_options(*this), mesh_options(*this),
		restart_options(*this), units_options(*this), blast_options(*this),
		timestep_options(*this), runtime_options(*this), execution_options(*this) {
	}
	options(options const& other) : options() {
		*this = other;
	}
	options& operator=(options const&) = default;

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
		arc & silo_num_groups;
		arc & amrbnd_order;
		arc & dual_energy_sw1;
		arc & dual_energy_sw2;
		arc & hard_dt;
		arc & correct_am_grav;
		arc & correct_am_hydro;
		arc & rewrite_silo;
		arc & rad_implicit;
		arc & rad_subcycling;
		arc & rad_c_ratio;
		arc & rad_cfl;
		arc & rad_max_subcycles;
		arc & rad_theta;
		arc & rad_velocity_terms;
		arc & rad_opacity;
		arc & rad_energy_mode;
		arc & rad_log_subcycles;

		arc & radReference & radTestChi & radTestWidth;
		arc & radTestBackground & radTestAmplitude & radTestLuminosity;
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

template<class T = Real>
struct hydro_state_t: public std::vector<T> {
	hydro_state_t() :
			std::vector<T>(opts().n_fields) {
	}
};

#endif /* OPTIONS_HPP_ */
