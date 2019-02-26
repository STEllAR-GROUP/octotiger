/*
 * options.hpp
 *
 *  Created on: Nov 13, 2015
 *      Author: dmarce1
 */

#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include "octotiger/defs.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/options_enum.hpp"
#include "octotiger/real.hpp"

#include <hpx/include/naming.hpp>

#include <cstddef>
#include <string>
#include <vector>

COMMAND_LINE_ENUM(problem_type,DWD,SOD,BLAST,NONE,SOLID_SPHERE,STAR,MOVING_STAR,RADIATION_TEST,ROTATING_STAR,MARSHAK);

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
	bool compress_silo;
	bool v1309;
	bool rad_implicit;

	integer accretor_refine;
	integer donor_refine;
	integer max_level;
	integer ngrids;
	integer stop_step;

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

    size_t cuda_streams_per_locality;
	size_t cuda_streams_per_gpu;

	std::string input_file;
	std::string config_file;
	std::string data_dir;
	std::string output_filename;
	std::string restart_filename;
	integer n_species;
	integer n_fields;

	eos_type eos;

	problem_type problem;

	interaction_kernel_type m2m_kernel_type;
	interaction_kernel_type p2m_kernel_type;
	interaction_kernel_type p2p_kernel_type;

	std::vector<real> atomic_mass;
	std::vector<real> atomic_number;
	std::vector<real> X;
	std::vector<real> Z;

	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & dual_energy_sw1;
		arc & dual_energy_sw2;
		arc & hard_dt;
		arc & rad_implicit;
		arc & n_fields;
		arc & n_species;
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
		arc & compress_silo;
		arc & v1309;
		arc & variable_omega;
		arc & silo_planes_only;
		arc & stop_time;
		arc & max_level;
		arc & xscale;
		arc & cfl;
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
		arc & atomic_mass;
		arc & atomic_number;
		arc & X;
		arc & Z;
		arc & code_to_g;
		arc & code_to_s;
		arc & code_to_cm;
	}

	bool process_options(int argc, char* argv[]);

	static std::vector<hpx::id_type> all_localities;
};


options& opts();


template<class T = real>
struct hydro_state_t: public std::vector<T> {
	hydro_state_t() :
			std::vector<T>(opts().n_fields) {
	}
};


#endif /* OPTIONS_HPP_ */
