/*
 * options.hpp
 *
 *  Created on: Nov 13, 2015
 *      Author: dmarce1
 */

#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include <string>
#include <vector>
#include <hpx/hpx.hpp>
#include "defs.hpp"
#include "interaction_types.hpp"
#include "options_enum.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

COMMAND_LINE_ENUM(problem_type,DWD,SOD,BLAST,NONE,SOLID_SPHERE,STAR,MOVING_STAR,RADIATION_TEST,ROTATING_STAR);

COMMAND_LINE_ENUM(eos_type,IDEAL,WD);

class options {
public:

	bool bench;
	bool disable_output;
	bool core_refine;
	bool gravity;
	bool hydro;
	bool radiation;
	bool silo_planes_only;
	bool variable_omega;

	integer accretor_refine;
	integer donor_refine;
	integer max_level;
	integer ngrids;
	integer stop_step;

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

    size_t cuda_streams_per_locality;
	size_t cuda_streams_per_gpu;

	std::string input_file;
	std::string config_file;
	std::string data_dir;
	std::string output_filename;
	std::string restart_filename;

	eos_type eos;

	problem_type problem;

	interaction_kernel_type m2m_kernel_type;
	interaction_kernel_type p2m_kernel_type;
	interaction_kernel_type p2p_kernel_type;

	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & input_file;
		arc & config_file;
		arc & hydro;
		arc & gravity;
		arc & bench;
		arc & radiation;
		arc & m2m_kernel_type;
		arc & p2m_kernel_type;
		arc & p2p_kernel_type;
		arc & entropy_driving_rate;
		arc & entropy_driving_time;
		arc & driving_rate;
		arc & driving_time;
		arc & refinement_floor;
		arc & ngrids;
		arc & variable_omega;
		arc & silo_planes_only;
		arc & stop_time;
		arc & max_level;
		arc & xscale;
		arc & omega;
		arc & restart_filename;
		arc & output_filename;
		arc & output_dt;
		arc & stop_step;
		arc & disable_output;
		arc & theta;
		arc & core_refine;
		arc & donor_refine;
		arc & accretor_refine;
		int tmp = problem;
		arc & tmp;
		problem = (problem_type) tmp;
		tmp = eos;
		arc & tmp;
		eos = (eos_type) tmp;
		arc & data_dir;

		arc & m2m_kernel_type;
		arc & p2p_kernel_type;
		arc & p2m_kernel_type;
		arc & cuda_streams_per_locality;
		arc & cuda_streams_per_gpu;
	}

	bool process_options(int argc, char* argv[]);

	static std::vector<hpx::id_type> all_localities;
};

#ifndef IN_OPTIONS_CPP
extern options opts;
#endif

#endif /* OPTIONS_HPP_ */
