/*
 * options.cpp
 *
 *  Created on: Nov 13, 2015
 *      Author: dmarce1
 */

#include "octotiger/options.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/real.hpp"

#include <boost/program_options.hpp>

#include <cmath>
#include <iosfwd>
#include <sstream>
#include <string>
#include <vector>

void normalize_constants();

#define IN_OPTIONS_CPP

constexpr real mass_solar = 1.2969;
constexpr real number_solar = 1.0994;
constexpr real X_solar = 0.7068;
constexpr real Z_solar = 0.0181;

options& opts() {
	static options opts_;
	return opts_;
}

inline std::string to_string(const std::string& str) {
	return str;
}

inline std::string to_string(const real& num) {
	std::ostringstream strm;
	strm << std::scientific << num;
	return strm.str();
}

inline std::string to_string(const integer& num) {
	return std::to_string(num);
}

inline std::string to_string(const size_t& num) {
	return std::to_string(num);
}

inline std::string to_string(const bool& b) {
	return b ? "T" : "F";
}

bool options::process_options(int argc, char* argv[]) {
	namespace po = boost::program_options;
	code_to_s = code_to_g = code_to_cm = 1.0;

	po::options_description command_opts("options");

	command_opts.add_options() //
	("help", "produce help message")
	("xscale", po::value<real>(&(opts().xscale))->default_value(1.0), "grid scale")           //
	("cfl", po::value<real>(&(opts().cfl))->default_value(2./15.), "cfl factor")           //
	("omega", po::value<real>(&(opts().omega))->default_value(0.0), "(initial) angular frequency")                          //
	("compress_silo", po::value<bool>(&(opts().compress_silo))->default_value(true), "compress SILO files to fewer grids")    //
	("v1309", po::value<bool>(&(opts().v1309))->default_value(false), "V1309 subproblem of DWD")                   //
	("variable_omega", po::value<bool>(&(opts().variable_omega))->default_value(false), "use variable omega")                 //
	("driving_rate", po::value<real>(&(opts().driving_rate))->default_value(0.0), "angular momentum loss driving rate")     //
	("driving_time", po::value<real>(&(opts().driving_time))->default_value(0.0), "A.M. driving rate time")                 //
	("entropy_driving_rate", po::value<real>(&(opts().driving_rate))->default_value(0.0), "entropy loss driving rate")      //
	("entropy_driving_time", po::value<real>(&(opts().driving_time))->default_value(0.0), "entropy driving rate time")      //
	("core_refine", po::value<bool>(&(opts().core_refine))->default_value(false), "refine cores by one more level")           //
	("accretor_refine", po::value<integer>(&(opts().accretor_refine))->default_value(0), "number of extra levels for accretor") //
	("donor_refine", po::value<integer>(&(opts().donor_refine))->default_value(0), "number of extra levels for donor")      //
	("ngrids", po::value<integer>(&(opts().ngrids))->default_value(-1), "fix numbger of grids")                             //
	("refinement_floor", po::value<real>(&(opts().refinement_floor))->default_value(1.0e-3), "density refinement floor")      //
	("theta", po::value<real>(&(opts().theta))->default_value(0.5),
			"controls nearness determination for FMM, must be between 1/3 and 1/2")                                           //
	("eos", po::value<eos_type>(&(opts().eos))->default_value(IDEAL), "gas equation of state")                              //
	("hydro", po::value<bool>(&(opts().hydro))->default_value(true), "hydro on/off")    //
	("radiation", po::value<bool>(&(opts().radiation))->default_value(false), "radiation on/off")    //
	("rad_implicit", po::value<bool>(&(opts().rad_implicit))->default_value(true), "implicit radiation on/off")    //
	("gravity", po::value<bool>(&(opts().gravity))->default_value(true), "gravity on/off")    //
	("bench", po::value<bool>(&(opts().bench))->default_value(false), "run benchmark") //
	("datadir", po::value<std::string>(&(opts().data_dir))->default_value("./"), "directory for output") //
	("output", po::value<std::string>(&(opts().output_filename))->default_value(""), "filename for output") //
	("odt", po::value<real>(&(opts().output_dt))->default_value(1.0 / 25.0), "output frequency") //
	("dual_energy_sw1", po::value<real>(&(opts().dual_energy_sw1))->default_value(0.1), "dual energy switch 1") //
	("dual_energy_sw2", po::value<real>(&(opts().dual_energy_sw2))->default_value(0.001), "dual energy switch 2") //
	("hard_dt", po::value<real>(&(opts().hard_dt))->default_value(-1), "timestep size") //
	("disable_output", po::value<bool>(&(opts().disable_output)), "disable silo output") //
	("silo_planes_only", po::value<bool>(&(opts().silo_planes_only)), "disable silo output") //
	("problem", po::value<problem_type>(&(opts().problem))->default_value(NONE), "problem type")                            //
	("restart_filename", po::value<std::string>(&(opts().restart_filename))->default_value(""), "restart filename")         //
	("stop_time", po::value<real>(&(opts().stop_time))->default_value(std::numeric_limits<real>::max()),
			"time to end simulation") //
	("stop_step", po::value<integer>(&(opts().stop_step))->default_value(std::numeric_limits<integer>::max() - 1),
			"number of timesteps to run")                                //
	("max_level", po::value<integer>(&(opts().max_level))->default_value(1), "maximum number of refinement levels")         //
	("multipole_kernel_type", po::value<interaction_kernel_type>(&(opts().m2m_kernel_type))->default_value(OLD),
			"boundary multipole-multipole kernel type") //
	("p2p_kernel_type", po::value<interaction_kernel_type>(&(opts().p2p_kernel_type))->default_value(OLD),
			"boundary particle-particle kernel type")   //
	("p2m_kernel_type", po::value<interaction_kernel_type>(&(opts().p2m_kernel_type))->default_value(OLD),
			"boundary particle-multipole kernel type")  //
	("cuda_streams_per_locality", po::value<size_t>(&(opts().cuda_streams_per_locality))->default_value(size_t(0)),
			"cuda streams per HPX locality") //
	("cuda_streams_per_gpu", po::value<size_t>(&(opts().cuda_streams_per_gpu))->default_value(size_t(0)),
			"cuda streams per GPU (per locality)") //
	("input_file", po::value<std::string>(&(opts().input_file))->default_value(""), "input file for test problems") //
	("config_file", po::value<std::string>(&(opts().config_file))->default_value(""), "configuration file") //
	("n_species", po::value<integer>(&(opts().n_species))->default_value(1), "number of mass species") //
	("atomic_mass", po::value<std::vector<real>>(&(opts().atomic_mass))->multitoken(), "atomic masses") //
	("atomic_number", po::value<std::vector<real>>(&(opts().atomic_number))->multitoken(), "atomic numbers") //
	("X", po::value<std::vector<real>>(&(opts().X))->multitoken(), "X - hydrogen mass fraction") //
	("Z", po::value<std::vector<real>>(&(opts().Z))->multitoken(), "Z - metallicity") //
	("code_to_g", po::value<real>(&(opts().code_to_g))->default_value(1), "code units to grams") //
	("code_to_cm", po::value<real>(&(opts().code_to_cm))->default_value(1), "code units to centimeters") //
	("code_to_s", po::value<real>(&(opts().code_to_s))->default_value(1), "code units to seconds") //
			;

	boost::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << command_opts << "\n";
		exit(0);
	}
	if (!config_file.empty()) {
		std::ifstream ifs { vm["config_file"].as<std::string>().c_str() };
		if (ifs) {
			store(parse_config_file(ifs, command_opts), vm);
		} else {
			std::cout << "Configuration file \"" << config_file << " not found!\n";
			exit(0);
		}
	}
	po::notify(vm);
	if (opts().problem == DWD || opts().problem == ROTATING_STAR ) {
		opts().n_species = std::max(int(5), int(opts().n_species));
	}
	n_fields = n_species + 10;
	if (!opts().restart_filename.empty()) {
		printf("1\n");
		load_options_from_silo(opts().restart_filename);
		printf("1\n");
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
		SHOW(dual_energy_sw1);
		SHOW(dual_energy_sw2);
		SHOW(hard_dt);
		SHOW(bench);
		SHOW(disable_output);
		SHOW(v1309);
		SHOW(compress_silo);
		SHOW(core_refine);
		SHOW(gravity);
		SHOW(hydro);
		SHOW(radiation);
		SHOW(rad_implicit);
		SHOW(silo_planes_only);
		SHOW(variable_omega);
		SHOW(accretor_refine);
		SHOW(donor_refine);
		SHOW(max_level);
		SHOW(ngrids);
		SHOW(stop_step);
		SHOW(driving_rate);
		SHOW(driving_time);
		SHOW(entropy_driving_rate);
		SHOW(entropy_driving_time);
		SHOW(omega);
		SHOW(output_dt);
		SHOW(refinement_floor);
		SHOW(stop_time);
		SHOW(theta);
		SHOW(cfl);
		SHOW(xscale);
		SHOW(cuda_streams_per_locality);
		SHOW(cuda_streams_per_gpu);
		SHOW(config_file);
		SHOW(data_dir);
		SHOW(input_file);
		SHOW(output_filename);
		SHOW(restart_filename);
		SHOW(problem);
		SHOW(eos);
		SHOW(m2m_kernel_type);
		SHOW(p2m_kernel_type);
		SHOW(p2p_kernel_type);
		SHOW(n_species);
		SHOW(code_to_g);
		SHOW(code_to_s);
		SHOW(code_to_cm);

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
	return true;
}

std::vector<hpx::id_type> options::all_localities = { };
