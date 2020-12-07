//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/real.hpp"
#include "octotiger/common_kernel/interaction_constants.hpp"

#include <boost/program_options.hpp>

#include <cmath>
#include <iosfwd>
#include <sstream>
#include <string>
#include <vector>

#define IN_OPTIONS_CPP

constexpr real mass_solar = 1.2969;
constexpr real number_solar = 1.0994;
constexpr real X_solar = 0.7068;
constexpr real Z_solar = 0.0181;

options& opts() {
	static options opts_;
	return opts_;
}

inline std::string to_string(const std::string &str) {
	return str;
}

inline std::string to_string(const real &num) {
	std::ostringstream strm;
	strm << std::scientific << num;
	return strm.str();
}

inline std::string to_string(const integer &num) {
	return std::to_string(num);
}

inline std::string to_string(const size_t &num) {
	return std::to_string(num);
}

inline std::string to_string(const bool &b) {
	return b ? "T" : "F";
}

bool options::process_options(int argc, char *argv[]) {
	namespace po = boost::program_options;
	code_to_s = code_to_g = code_to_cm = 1.0;

	po::options_description command_opts("options");

	command_opts.add_options() //
	("help", "produce help message")("xscale", po::value<real>(&(opts().xscale))->default_value(1.0), "grid scale")           //
	("dt_max", po::value<real>(&(opts().dt_max))->default_value(0.333333), "max allowed pct change for positive fields in a timestep")           //
	("cfl", po::value<real>(&(opts().cfl))->default_value(0.4), "cfl factor")           //
	("omega", po::value<real>(&(opts().omega))->default_value(0.0), "(initial) angular frequency")                          //
	("v1309", po::value<bool>(&(opts().v1309))->default_value(false), "V1309 subproblem of DWD")                   //
	("idle_rates", po::value<bool>(&(opts().idle_rates))->default_value(false), "show idle rates and locality info in SILO")                 //
	("eblast0", po::value<real>(&(opts().eblast0))->default_value(1.0), "energy for blast wave")     //
	("rho_floor", po::value<real>(&(opts().rho_floor))->default_value(0.0), "density floor")     //
	("tau_floor", po::value<real>(&(opts().tau_floor))->default_value(0.0), "entropy tracer floor")     //
	("sod_rhol", po::value<real>(&(opts().sod_rhol))->default_value(1.0), "density in the left part of the grid")     //
	("sod_rhor", po::value<real>(&(opts().sod_rhor))->default_value(0.125), "density in the right part of the grid")     //
	("sod_pl", po::value<real>(&(opts().sod_pl))->default_value(1.0), "pressure in the left part of the grid")     //
	("sod_pr", po::value<real>(&(opts().sod_pr))->default_value(0.1), "pressure in the right part of the grid")     //
	("sod_theta", po::value<real>(&(opts().sod_theta))->default_value(0.0), "angle made by diaphragm normal w/x-axis (deg)")     //
	("sod_phi", po::value<real>(&(opts().sod_phi))->default_value(90.0), "angle made by diaphragm normal w/z-axis (deg)")     //
	("sod_gamma", po::value<real>(&(opts().sod_gamma))->default_value(1.4), "ratio of specific heats for gas")     //
        ("solid_sphere_xcenter", po::value<real>(&(opts().solid_sphere_xcenter))->default_value(0.25), "x-position of the sphere center")     //
        ("solid_sphere_ycenter", po::value<real>(&(opts().solid_sphere_ycenter))->default_value(0.0), "y-position of the sphere center")     //
        ("solid_sphere_zcenter", po::value<real>(&(opts().solid_sphere_zcenter))->default_value(0.0), "z-position of the sphere center")     //
        ("solid_sphere_radius", po::value<real>(&(opts().solid_sphere_radius))->default_value(1.0 / 3.0), "radius of the sphere")     //
        ("solid_sphere_mass", po::value<real>(&(opts().solid_sphere_mass))->default_value(1.0), "total mass enclosed inside the sphere")     //
        ("solid_sphere_rho_min", po::value<real>(&(opts().solid_sphere_rho_min))->default_value(1.0e-12), "minimal density outside (and within) the sphere")     //
        ("star_xcenter", po::value<real>(&(opts().star_xcenter))->default_value(0.0), "x-position of the star center")     //
        ("star_ycenter", po::value<real>(&(opts().star_ycenter))->default_value(0.0), "y-position of the star center")     //
        ("star_zcenter", po::value<real>(&(opts().star_zcenter))->default_value(0.0), "z-position of the star center")     //
        ("star_n", po::value<real>(&(opts().star_n))->default_value(1.5), "polytropic index of the star")     //
        ("star_rmax", po::value<real>(&(opts().star_rmax))->default_value(1.0 / 3.0), "maximal star radius")     //
        ("star_dr", po::value<real>(&(opts().star_dr))->default_value(1.0 / (3.0 * 128.0)), "differential radius for solving the Lane-Emden equation")     //
        ("star_alpha", po::value<real>(&(opts().star_alpha))->default_value(1.0 / (3.0 * 3.65375)), "scaling factor for the Lane-Emden equation") // for default n=3/2, ksi_1=3.65375 and alpha=rmax/ksi_1
        ("star_rho_center", po::value<real>(&(opts().star_rho_center))->default_value(1.0), "density at the center of the star")     //
        ("star_rho_out", po::value<real>(&(opts().star_rho_out))->default_value(1.0e-10), "density outside the star")     //
        ("moving_star_xvelocity", po::value<real>(&(opts().moving_star_xvelocity))->default_value(1.0), "velocity of the star in the x-direction")     //
        ("moving_star_yvelocity", po::value<real>(&(opts().moving_star_yvelocity))->default_value(1.0), "velocity of the star in the y-direction")     //
        ("moving_star_zvelocity", po::value<real>(&(opts().moving_star_zvelocity))->default_value(1.0), "velocity of the star in the z-direction")     //
	("clight_retard", po::value<real>(&(opts().clight_retard))->default_value(1.0), "retardation factor for speed of light")                 //
	("driving_rate", po::value<real>(&(opts().driving_rate))->default_value(0.0), "angular momentum loss driving rate")     //
	("driving_time", po::value<real>(&(opts().driving_time))->default_value(0.0), "A.M. driving rate time")                 //
	("entropy_driving_time", po::value<real>(&(opts().entropy_driving_time))->default_value(0.0), "entropy driving rate time")                 //
	("entropy_driving_rate", po::value<real>(&(opts().entropy_driving_rate))->default_value(0.0), "entropy loss driving rate")      //
	("future_wait_time", po::value<integer>(&(opts().future_wait_time))->default_value(-1), "")      //
	("silo_offset_x", po::value<integer>(&(opts().silo_offset_x))->default_value(0), "")      //
	("silo_offset_y", po::value<integer>(&(opts().silo_offset_y))->default_value(0), "")      //
	("silo_offset_z", po::value<integer>(&(opts().silo_offset_z))->default_value(0), "")      //
	("amrbnd_order", po::value<integer>(&(opts().amrbnd_order))->default_value(1), "amr boundary interpolation order")        //
	("scf_output_frequency", po::value<integer>(&(opts().scf_output_frequency))->default_value(25), "Frequency of SCF output")        //
	("silo_num_groups", po::value<integer>(&(opts().silo_num_groups))->default_value(-1), "Number of SILO I/O groups")        //
	("core_refine", po::value<bool>(&(opts().core_refine))->default_value(false), "refine cores by one more level")           //
	("accretor_refine", po::value<integer>(&(opts().accretor_refine))->default_value(0), "number of extra levels for accretor") //
	("extra_regrid", po::value<integer>(&(opts().extra_regrid))->default_value(0), "number of extra regrids on startup") //
	("donor_refine", po::value<integer>(&(opts().donor_refine))->default_value(0), "number of extra levels for donor")      //
	("ngrids", po::value<integer>(&(opts().ngrids))->default_value(-1), "fix numbger of grids")                             //
	("refinement_floor", po::value<real>(&(opts().refinement_floor))->default_value(1.0e-3), "density refinement floor")      //
	("theta", po::value<real>(&(opts().theta))->default_value(0.5), "controls nearness determination for FMM, must be between 1/3 and 1/2")               //
	("eos", po::value<eos_type>(&(opts().eos))->default_value(IDEAL), "gas equation of state")                              //
	("hydro", po::value<bool>(&(opts().hydro))->default_value(true), "hydro on/off")    //
	("radiation", po::value<bool>(&(opts().radiation))->default_value(false), "radiation on/off")    //
	("correct_am_hydro", po::value<bool>(&(opts().correct_am_hydro))->default_value(true), "Angular momentum correction switch for hydro")    //
	("correct_am_grav", po::value<bool>(&(opts().correct_am_grav))->default_value(true), "Angular momentum correction switch for gravity")    //
	("rewrite_silo", po::value<bool>(&(opts().rewrite_silo))->default_value(false), "rewrite silo and exit")    //
	("rad_implicit", po::value<bool>(&(opts().rad_implicit))->default_value(true), "implicit radiation on/off")    //
	("gravity", po::value<bool>(&(opts().gravity))->default_value(true), "gravity on/off")    //
	("bench", po::value<bool>(&(opts().bench))->default_value(false), "run benchmark") //
	("datadir", po::value<std::string>(&(opts().data_dir))->default_value("./"), "directory for output") //
	("output", po::value<std::string>(&(opts().output_filename))->default_value(""), "filename for output") //
	("odt", po::value<real>(&(opts().output_dt))->default_value(1.0 / 100.0), "output frequency") //
	("dual_energy_sw1", po::value<real>(&(opts().dual_energy_sw1))->default_value(0.001), "dual energy switch 1") //
	("dual_energy_sw2", po::value<real>(&(opts().dual_energy_sw2))->default_value(0.1), "dual energy switch 2") //
	("hard_dt", po::value<real>(&(opts().hard_dt))->default_value(-1), "timestep size") //
	("experiment", po::value<int>(&(opts().experiment))->default_value(0), "experiment") //
	("unigrid", po::value<bool>(&(opts().unigrid))->default_value(false), "unigrid") //
	("inflow_bc", po::value<bool>(&(opts().inflow_bc))->default_value(false), "Inflow Boundary Conditions") //
	("reflect_bc", po::value<bool>(&(opts().reflect_bc))->default_value(false), "Reflecting Boundary Conditions") //
	("cdisc_detect", po::value<bool>(&(opts().cdisc_detect))->default_value(true), "PPM contact discontinuity detection") //
	("disable_output", po::value<bool>(&(opts().disable_output))->default_value(false), "disable silo output") //
	("disable_diagnostics", po::value<bool>(&(opts().disable_diagnostics))->default_value(false), "disable diagnostics") //
	("problem", po::value<problem_type>(&(opts().problem))->default_value(NONE), "problem type")                            //
	("restart_filename", po::value<std::string>(&(opts().restart_filename))->default_value(""), "restart filename")         //
	("stop_time", po::value<real>(&(opts().stop_time))->default_value(std::numeric_limits<real>::max()), "time to end simulation") //
	("stop_step", po::value<integer>(&(opts().stop_step))->default_value(std::numeric_limits<integer>::max() - 1), "number of timesteps to run")          //
	("min_level", po::value<integer>(&(opts().min_level))->default_value(1), "minimum number of refinement levels")         //
	("max_level", po::value<integer>(&(opts().max_level))->default_value(1), "maximum number of refinement levels")         //
	("amr_boundary_kernel_type", po::value<amr_boundary_type>(&(opts().amr_boundary_kernel_type))->default_value(AMR_OPTIMIZED), "amr completion kernel type") //
	("multipole_kernel_type", po::value<interaction_kernel_type>(&(opts().m2m_kernel_type))->default_value(SOA_CPU), "boundary multipole-multipole kernel type") //
	("p2p_kernel_type", po::value<interaction_kernel_type>(&(opts().p2p_kernel_type))->default_value(SOA_CPU), "boundary particle-particle kernel type")   //
	("p2m_kernel_type", po::value<interaction_kernel_type>(&(opts().p2m_kernel_type))->default_value(SOA_CPU), "boundary particle-multipole kernel type") //
	("cuda_number_gpus", po::value<size_t>(&(opts().cuda_number_gpus))->default_value(size_t(0)), "cuda streams per HPX locality") //
	("cuda_streams_per_gpu", po::value<size_t>(&(opts().cuda_streams_per_gpu))->default_value(size_t(0)), "cuda streams per GPU (per locality)") //
	("cuda_buffer_capacity", po::value<size_t>(&(opts().cuda_buffer_capacity))->default_value(size_t(5)), "How many launches should be buffered before using the CPU") //
	("cuda_polling_executor", po::value<bool>(&(opts().cuda_polling_executor))->default_value(true), "Use polling (true) or callback (false) executor") //
	("root_node_on_device", po::value<bool>(&(opts().root_node_on_device))->default_value(true), "Offload root node gravity kernels to the GPU? May degrade performance given weak GPUs") //
	("legacy_hydro", po::value<bool>(&(opts().legacy_hydro))->default_value(false), "Use new hydro (false) or legacy hydro (true)") //
	("input_file", po::value<std::string>(&(opts().input_file))->default_value(""), "input file for test problems") //
	("config_file", po::value<std::string>(&(opts().config_file))->default_value(""), "configuration file") //
	("n_species", po::value<integer>(&(opts().n_species))->default_value(5), "number of mass species") //
	("atomic_mass", po::value<std::vector<real>>(&(opts().atomic_mass))->multitoken(), "atomic masses") //
	("atomic_number", po::value<std::vector<real>>(&(opts().atomic_number))->multitoken(), "atomic numbers") //
	("X", po::value<std::vector<real>>(&(opts().X))->multitoken(), "X - hydrogen mass fraction") //
	("Z", po::value<std::vector<real>>(&(opts().Z))->multitoken(), "Z - metallicity") //
	("code_to_g", po::value<real>(&(opts().code_to_g))->default_value(1), "code units to grams") //
	("code_to_cm", po::value<real>(&(opts().code_to_cm))->default_value(1), "code units to centimeters") //
	("code_to_s", po::value<real>(&(opts().code_to_s))->default_value(1), "code units to seconds") //
	("rotating_star_amr", po::value<bool>(&(opts().rotating_star_amr))->default_value(false), "rotating star with AMR boundary in star") //
	("rotating_star_x", po::value<real>(&(opts().rotating_star_x))->default_value(0.0), "x center of rotating_star") //
			;

	boost::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << command_opts << "\n";
		return false;
	}
	if (!config_file.empty()) {
		std::ifstream cfg_fs { vm["config_file"].as<std::string>() };
		if (cfg_fs) {
			po::store(po::parse_config_file(cfg_fs, command_opts), vm);
		} else {
			printf("Configuration file %s not found!\n", config_file.c_str());
			return false;
		}
	}
	po::notify(vm);
	if (opts().silo_num_groups == -1) {
		opts().silo_num_groups = hpx::find_all_localities().size();

	}
	if (opts().problem == DWD) {
		opts().n_species = std::max(int(5), int(opts().n_species));
	}
        if (opts().problem == MOVING_STAR || opts().problem == ROTATING_STAR) {
                opts().n_species = std::max(int(2), int(opts().n_species));
	}
	n_fields = n_species + 10;
	if (!opts().restart_filename.empty()) {
		FILE *fp = fopen(opts().restart_filename.c_str(), "rb");
		if (fp == NULL) {
			printf("restart.silo does not exist or invalid permissions\n");
			sleep(10);
			abort();
		} else {
			fclose(fp);
		}
		load_options_from_silo(opts().restart_filename);
	}
    if (opts().cuda_streams_per_gpu > 0 && opts().cuda_number_gpus == 0) {
        opts().cuda_number_gpus = 1;
	}
	if (opts().theta < octotiger::fmm::THETA_FLOOR) {
		std::cerr << "theta " << theta << " is too small since Octo-Tiger was compiled for a minimum of " << octotiger::fmm::THETA_FLOOR << std::endl;
		std::cerr << "Either increase theta or recompile with a new theta minimum using the cmake parameter OCTOTIGER_THETA_MINIMUM";
		abort();
	}
	{
#define SHOW( opt ) std::cout << std::string( #opt ) << " = " << to_string(opt) << '\n';
		std::cout << "atomic_number=";
		for (auto r : atomic_number) {
			std::cout << std::to_string(r) << ',';
		}
		std::cout << '\n';
		std::cout << "atomic_mass=";
		for (auto r : atomic_mass) {
			std::cout << std::to_string(r) << ',';
		}
		std::cout << '\n';
		std::cout << "X=";
		for (auto r : X) {
			std::cout << std::to_string(r) << ',';
		}
		std::cout << '\n';
		std::cout << "Z=";
		for (auto r : Z) {
			std::cout << std::to_string(r) << ',';
		}
		std::cout << '\n';
		const auto num_loc = hpx::find_all_localities().size();
		if (silo_num_groups > num_loc) {
			printf("Number of SILO file groups cannot be greater than number of localities. Setting silo_num_groupds to %li\n", num_loc);
			silo_num_groups = num_loc;
		}
		SHOW(accretor_refine);
		SHOW(amrbnd_order);
		SHOW(bench);
		SHOW(cdisc_detect);
		SHOW(cfl);
		SHOW(clight_retard);
		SHOW(config_file);
		SHOW(core_refine);
		SHOW(correct_am_grav);
		SHOW(correct_am_hydro);
		SHOW(code_to_cm);
		SHOW(code_to_g);
		SHOW(code_to_s);
		SHOW(cuda_number_gpus);
		SHOW(cuda_streams_per_gpu);
		SHOW(cuda_buffer_capacity);
		SHOW(cuda_polling_executor);
		SHOW(legacy_hydro);
		SHOW(data_dir);
		SHOW(disable_output);
		SHOW(driving_rate);
		SHOW(driving_time);
		SHOW(dt_max);
		SHOW(donor_refine);
		SHOW(dual_energy_sw1);
		SHOW(dual_energy_sw2);
		SHOW(eblast0);
		SHOW(eos);
		SHOW(entropy_driving_rate);
		SHOW(entropy_driving_time);
		SHOW(future_wait_time);
		SHOW(hard_dt);
		SHOW(hydro);
		SHOW(inflow_bc);
		SHOW(input_file);
		SHOW(m2m_kernel_type);
		SHOW(min_level);
		SHOW(max_level);
		SHOW(n_species);
		SHOW(ngrids);
		SHOW(omega);
		SHOW(output_dt);
		SHOW(output_filename);
		SHOW(p2m_kernel_type);
		SHOW(p2p_kernel_type);
		SHOW(problem);
		SHOW(rad_implicit);
		SHOW(radiation);
		SHOW(refinement_floor);
		SHOW(reflect_bc);
		SHOW(restart_filename);
		SHOW(rotating_star_amr);
		SHOW(rotating_star_x);
		SHOW(scf_output_frequency);
		SHOW(silo_num_groups);
		SHOW(stop_step);
		SHOW(stop_time);
		SHOW(theta);
		SHOW(unigrid);
		SHOW(v1309);
		SHOW(idle_rates);
		SHOW(xscale);

	}
	while (atomic_number.size() < opts().n_species) {
		atomic_number.push_back(number_solar);
	}
	while (atomic_mass.size() < opts().n_species) {
		atomic_mass.push_back(mass_solar);
	}
	while (X.size() < opts().n_species) {
		X.push_back(X_solar);
	}
	while (Z.size() < opts().n_species) {
		Z.push_back(Z_solar);
	}
	normalize_constants();
	if (opts().problem == DWD) {
		if (opts().restart_filename == "" && opts().disable_diagnostics) {
      std::cerr << "Diagnostics must be enabled for DWD" << std::endl;
			sleep(10);
			abort();
		}
	}
#ifdef OCTOTIGER_HAVE_KOKKOS
#ifdef OCTOTIGER_HAVE_CUDA
  /*if (!opts().cuda_polling_executor) {
      std::cerr << "\n\nError:" << std::endl;
      std::cerr << "Using KOKKOS with CUDA backend enabled requires --cuda_polling_executor=1 !" << std::endl;
      std::cerr << "Aborting..." << std::endl;
      abort();
  }*/
#endif
#endif
	return true;
}

std::vector<hpx::id_type> options::all_localities = { };
