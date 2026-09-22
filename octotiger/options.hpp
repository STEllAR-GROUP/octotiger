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
#include "octotiger/radiation/grey_opacity.hpp"

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
class OptionReference {
public:
	explicit OptionReference(T& value) : valuePointer(&value) {
	}
	operator T&() const {
		return *valuePointer;
	}
	T& get() const {
		return *valuePointer;
	}
	OptionReference& operator=(T const& value) {
		*valuePointer = value;
		return *this;
	}
	OptionReference& operator=(OptionReference const& other) {
		*valuePointer = other.get();
		return *this;
	}

private:
	T* valuePointer;
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
	bool radSubcycling = true;
	Real radCRatio = 1.0;
	Real radCfl = 0.4;
	integer radMaxSubcycles = 1024;
	Real radTheta = 1.0;
	bool radVelocityTerms = true;
	Real radOpacity = -1.0;
	radiation::GreyOpacity radiationOpacity;
	std::string radEnergyMode = "thermal";
	bool radLogSubcycles = false;

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
	integer dimensionCount;
    bool modularHydro = false;
    bool modularTransport = false;
    bool modularRadiationSourceFree = false;
    std::string modularRadiationProblem = "streamingGaussian";
    std::string modularProblem = "sod";
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
	Real omegaX;
	Real omegaY;
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
	std::string detailedLogPath;
	std::string resultsPath;
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
	struct RadiationGroup {
		struct OpacityGroup {
			OptionReference<Real> constant;
			OptionReference<std::string> model, units;
			OptionReference<double> absorption, scattering, transportAbsorption;
			explicit OpacityGroup(options& o) : constant(o.radOpacity),
				model(o.radiationOpacity.model), units(o.radiationOpacity.units),
				absorption(o.radiationOpacity.absorption), scattering(o.radiationOpacity.scattering),
				transportAbsorption(o.radiationOpacity.transportAbsorption) {
			}
		} opacity;
		OptionReference<bool> enabled;
		OptionReference<bool> implicit;
		OptionReference<bool> subcycling;
		OptionReference<Real> reducedLightSpeedRatio;
		OptionReference<Real> cfl;
		OptionReference<integer> maxSubcycles;
		OptionReference<Real> sourceTheta;
		OptionReference<bool> velocityTerms;
		OptionReference<std::string> energyMode;
		OptionReference<bool> logSubcycles;
		RadiationGroup(options& owner) : opacity(owner),
			enabled(owner.radiation), implicit(owner.rad_implicit),
			subcycling(owner.radSubcycling), reducedLightSpeedRatio(owner.radCRatio),
			cfl(owner.radCfl), maxSubcycles(owner.radMaxSubcycles),
			sourceTheta(owner.radTheta), velocityTerms(owner.radVelocityTerms),
			energyMode(owner.radEnergyMode), logSubcycles(owner.radLogSubcycles) {
		}
	} radiationOptions;

	struct HydroGroup {
		OptionReference<bool> enabled;
		OptionReference<Real> gamma;
		OptionReference<Real> cfl;
		OptionReference<eos_type> eos;
		OptionReference<Real> densityFloor;
		OptionReference<Real> entropyFloor;
		HydroGroup(options& owner) : enabled(owner.hydro), gamma(owner.sod_gamma),
			cfl(owner.cfl), eos(owner.eos), densityFloor(owner.rho_floor),
			entropyFloor(owner.tau_floor) {
		}
	} hydroOptions;

	struct GravityGroup {
		OptionReference<bool> enabled;
		OptionReference<Real> openingAngle;
		OptionReference<bool> angularMomentumCorrection;
		GravityGroup(options& owner) : enabled(owner.gravity), openingAngle(owner.theta),
			angularMomentumCorrection(owner.correct_am_grav) {
		}
	} gravityOptions;

	struct ProblemGroup {
		struct BlastGroup {
			OptionReference<Real> energy;
			explicit BlastGroup(Real& value) : energy(value) {
			}
		} blast;
		OptionReference<problem_type> name;
		OptionReference<std::string> inputFile;
		OptionReference<int> experiment;
		ProblemGroup(options& owner) : blast(owner.eblast0), name(owner.problem),
			inputFile(owner.input_file), experiment(owner.experiment) {
		}
	} problemOptions;

	struct OutputGroup {
		OptionReference<std::string> directory;
		OptionReference<std::string> filename;
		OptionReference<Real> interval;
		OptionReference<bool> disabled;
		OptionReference<std::string> detailedLogPath;
		OptionReference<std::string> resultsPath;
		OutputGroup(options& owner) : directory(owner.data_dir), filename(owner.output_filename),
			interval(owner.output_dt), disabled(owner.disable_output),
			detailedLogPath(owner.detailedLogPath), resultsPath(owner.resultsPath) {
		}
	} outputOptions;

	struct MeshGroup {
		OptionReference<integer> dimensionCount;
		OptionReference<Real> scale;
		OptionReference<Real> omegaX;
		OptionReference<Real> omegaY;
		OptionReference<Real> omegaZ;
		OptionReference<integer> minimumLevel;
		OptionReference<integer> maximumLevel;
		OptionReference<bool> unigrid;
		OptionReference<bool> inflow;
		OptionReference<bool> periodic;
		OptionReference<bool> reflecting;
		MeshGroup(options& owner) : dimensionCount(owner.dimensionCount), scale(owner.xscale), minimumLevel(owner.min_level),
			omegaX(owner.omegaX), omegaY(owner.omegaY), omegaZ(owner.omega),
			maximumLevel(owner.max_level), unigrid(owner.unigrid),
			inflow(owner.inflow_bc), periodic(owner.periodic), reflecting(owner.reflect_bc) {
		}
	} meshOptions;

	struct RefinementGroup {
		OptionReference<integer> accretorLevels;
		OptionReference<bool> core;
		OptionReference<Real> densityFloor;
		OptionReference<Real> densityGradient;
		OptionReference<integer> donorLevels;
		RefinementGroup(options& owner) : accretorLevels(owner.accretor_refine),
			core(owner.core_refine), densityFloor(owner.refinement_floor),
			densityGradient(owner.grad_rho_refine), donorLevels(owner.donor_refine) {
		}
	} refinementOptions;

	struct RestartGroup {
		OptionReference<std::string> filename;
		explicit RestartGroup(options& owner) : filename(owner.restart_filename) {
		}
	} restartOptions;

	struct UnitsGroup {
		OptionReference<Real> grams;
		OptionReference<Real> centimeters;
		OptionReference<Real> seconds;
		UnitsGroup(options& owner) : grams(owner.code_to_g), centimeters(owner.code_to_cm),
			seconds(owner.code_to_s) {
		}
	} unitsOptions;

	struct BlastGroup {
		OptionReference<Real> energy;
		explicit BlastGroup(options& owner) : energy(owner.eblast0) {
		}
	} blastOptions;

	struct TimestepGroup {
		OptionReference<Real> maximumChange;
		OptionReference<Real> fixed;
		TimestepGroup(options& owner) : maximumChange(owner.dt_max), fixed(owner.hard_dt) {
		}
	} timestepOptions;

	struct RuntimeGroup {
		OptionReference<std::string> configFile;
		OptionReference<Real> stopTime;
		OptionReference<integer> stopStep;
		RuntimeGroup(options& owner) : configFile(owner.config_file),
			stopTime(owner.stop_time), stopStep(owner.stop_step) {
		}
	} runtimeOptions;

	struct ExecutionGroup {
		OptionReference<size_t> gpuCount;
		OptionReference<size_t> executorsPerGpu;
		OptionReference<int> pollingThreads;
		ExecutionGroup(options& owner) : gpuCount(owner.number_gpus),
			executorsPerGpu(owner.executors_per_gpu), pollingThreads(owner.polling_threads) {
		}
	} executionOptions;

	options() : radiationOptions(*this), hydroOptions(*this), gravityOptions(*this),
		problemOptions(*this), outputOptions(*this), meshOptions(*this),
		refinementOptions(*this), restartOptions(*this), unitsOptions(*this), blastOptions(*this),
		timestepOptions(*this), runtimeOptions(*this), executionOptions(*this) {
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
		arc & radSubcycling;
		arc & radCRatio;
		arc & radCfl;
		arc & radMaxSubcycles;
		arc & radTheta;
		arc & radVelocityTerms;
		arc & radOpacity;
		arc & radEnergyMode;
		arc & radLogSubcycles;

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
		arc & dimensionCount;
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
		// Same-build HPX broadcast extension. Disk checkpoints use versioned Silo fields.
		arc & radiationOpacity;
		arc & detailedLogPath;
		arc & resultsPath;
        arc & modularHydro;
        arc & modularProblem;
        arc & modularTransport;
        arc & modularRadiationSourceFree;
        arc & modularRadiationProblem;
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
