/*
 * options.cpp
 *
 *  Created on: Nov 13, 2015
 *      Author: dmarce1
 */

#include "defs.hpp"
#include "options.hpp"
#include <math.h>
#include "grid.hpp"
#define IN_OPTIONS_CPP

options opts;

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

	po::options_description command_opts("options");

	command_opts.add_options() //
    ("help", "produce help message")
	("xscale", po::value<real>(&(opts.xscale))->default_value(1.0), "grid scale")                   //
	("omega", po::value<real>(&(opts.omega))->default_value(0.0), "(initial) angular frequency")                              //
	("variable_omega", po::value<bool>(&(opts.variable_omega))->default_value(false), "use variable omega")                           //
	("driving_rate", po::value<real>(&(opts.driving_rate))->default_value(0.0), "angular momentum loss driving rate")         //
	("driving_time", po::value<real>(&(opts.driving_time))->default_value(0.0), "A.M. driving rate time")                     //
	("entropy_driving_rate", po::value<real>(&(opts.driving_rate))->default_value(0.0), "entropy loss driving rate")          //
	("entropy_driving_time", po::value<real>(&(opts.driving_time))->default_value(0.0), "entropy driving rate time")          //
	("core_refine", po::value<bool>(&(opts.core_refine))->default_value(false), "refine cores by one more level")             //
	("accretor_refine", po::value<integer>(&(opts.accretor_refine))->default_value(0), "number of extra levels for accretor") //
	("donor_refine", po::value<integer>(&(opts.donor_refine))->default_value(0), "number of extra levels for donor")          //
	("ngrids", po::value<integer>(&(opts.ngrids))->default_value(-1), "fix numbger of grids")                                 //
	("refinement_floor", po::value<real>(&(opts.refinement_floor))->default_value(-1.0), "density refinement floor")          //
	("theta", po::value<real>(&(opts.theta))->default_value(0.35),
			"controls nearness determination for FMM, must be between 1/3 and 1/2")                                           //
	("eos", po::value<eos_type>(&(opts.eos))->default_value(IDEAL), "gas equation of state")                                  //
	("hydro", po::value<bool>(&(opts.hydro))->default_value(true), "hydro on/off")    //
	("radiation", po::value<bool>(&(opts.radiation))->default_value(false), "radiation on/off")    //
	("gravity", po::value<bool>(&(opts.gravity))->default_value(true), "gravity on/off")    //
	("bench", po::value<bool>(&(opts.bench))->default_value(false), "run benchmark") //
	("datadir", po::value<std::string>(&(opts.data_dir))->default_value("./"), "directory for output") //
	("output", po::value<std::string>(&(opts.output_filename))->default_value(""), "filename for output") //
	("odt", po::value<real>(&(opts.output_dt))->default_value(1.0), "output frequency") //
	("disableoutput", po::value<bool>(&(opts.disable_output)), "disable silo output") //
	("siloplanesonly", po::value<bool>(&(opts.silo_planes_only)), "disable silo output") //
	("problem", po::value<problem_type>(&(opts.problem))->default_value(NONE), "problem type")                                //
	("restart_filename", po::value<std::string>(&(opts.restart_filename))->default_value(""), "restart filename")                      //
	("stop_time", po::value<real>(&(opts.stop_time))->default_value(std::numeric_limits<real>::max()), "time to end simulation") //
	("stop_step", po::value<integer>(&(opts.stop_step))->default_value(std::numeric_limits<integer>::max()-1),
			"number of timesteps to run")                                //
	("max_level", po::value<integer>(&(opts.max_level))->default_value(1), "maximum number of refinement levels")              //
	("multipole_kernel_type", po::value<interaction_kernel_type>(&(opts.m2m_kernel_type))->default_value(OLD),
			"boundary multipole-multipole kernel type") //
	("p2p_kernel_type", po::value<interaction_kernel_type>(&(opts.p2p_kernel_type))->default_value(OLD),
			"boundary particle-particle kernel type")   //
	("p2m_kernel_type", po::value<interaction_kernel_type>(&(opts.p2m_kernel_type))->default_value(OLD),
			"boundary particle-multipole kernel type")  //
	("cuda_streams_per_locality", po::value<size_t>(&(opts.cuda_streams_per_locality))->default_value(size_t(0)),
			"cuda streams per HPX locality") //
	("cuda_streams_per_gpu", po::value<size_t>(&(opts.cuda_streams_per_locality))->default_value(size_t(0)),
			"cuda streams per GPU (per locality)") //
			("input_file", po::value<std::string>(&(opts.input_file))->default_value(""), "input file for test problems") //
			("config_file", po::value<std::string>(&(opts.config_file))->default_value(""), "configuration file") //
			;

	boost::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);
    if (vm.count("help")) {
        std::cout << command_opts << "\n";
        return 1;
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

	{
#define SHOW( opt ) std::cout << std::string( #opt ) << " = " << to_string(opt) << '\n';
		SHOW(bench);
		SHOW(disable_output);
		SHOW(core_refine);
		SHOW(gravity);
		SHOW(hydro);
		SHOW(radiation);
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

	}
	return true;
}

std::vector<hpx::id_type> options::all_localities = { };
