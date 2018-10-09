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
#include <boost/algorithm/string.hpp>

#define COMMAND_LINE_ENUM( enum_name, option_name, args... )               \
	enum enum_name {                                                       \
		args                                                               \
	};                                                                     \
	static inline std::istream& operator>>(std::istream& in, enum_name& e) \
	{                                                                      \
		std::vector<std::string> strings;                                  \
		boost::split(strings, #args, boost::is_any_of(" ,"));              \
		enum_name enums[] = {args};                                        \
		bool success = false;                                              \
		std::string token;                                                 \
	    in >> token;                                                       \
		for( std::size_t i = 0; i < strings.size(); i++ ) {                \
			if( boost::iequals(strings[i], token)) {                       \
				success = true;                                            \
				e = enums[i];                                              \
			}                                                              \
		}                                                                  \
	   if( !success ) {                                                    \
	        in.setstate(std::ios_base::failbit);                           \
		}                                                                  \
	    return in;                                                         \
	}


COMMAND_LINE_ENUM( problem_type, "problem", DWD, SOD, BLAST, NONE, SOLID_SPHERE, STAR, MOVING_STAR, RADIATION_TEST);
COMMAND_LINE_ENUM( eos_type, "eos", IDEAL, WD);

class options {

	std::string exe_name;

	bool cmp(const char* str1, const char* str2);
	bool cmp(const std::string str1, const char* str2);
	void show_help();
public:
	bool core_refine;
	integer donor_refine;
	integer accretor_refine;
	bool vomega;
	real refinement_floor;
	bool refinement_floor_specified;
	eos_type eos;
	integer max_level;
	integer max_restart_level;
	real xscale;
	real omega;
	problem_type problem;
	std::string restart_filename;
	bool found_restart_file;
	std::string output_filename;
	bool output_only;
	real output_dt;
	real stop_time;
    integer stop_step;
	real contact_fill;
	integer ngrids;
	bool bench;
	bool radiation;
	real theta;
	bool disable_output;
    bool parallel_silo;
    bool silo_planes_only;
    std::string data_dir;

    interaction_kernel_type m2m_kernel_type;
    interaction_kernel_type p2p_kernel_type;
    interaction_kernel_type p2m_kernel_type;
    size_t cuda_streams_per_thread;

    real driving_rate;
    real driving_time;
    real entropy_driving_rate;
    real entropy_driving_time;
    template<class Arc>
	void serialize(Arc& arc, unsigned) {
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
		arc & refinement_floor_specified;
		arc & ngrids;
		arc & vomega;
		arc & parallel_silo;
		arc & silo_planes_only;
		arc & stop_time;
		arc & max_level;
		arc & max_restart_level;
		arc & xscale;
		arc & omega;
		arc & restart_filename;
		arc & found_restart_file;
		arc & output_filename;
		arc & output_only;
		arc & output_dt;
        arc & stop_step;
        arc & disable_output;
		arc & theta;
		arc & core_refine;
		arc & donor_refine;
		arc & accretor_refine;
		int tmp = problem;
		arc & tmp;
		problem = (problem_type)tmp;
		tmp = eos;
		arc & tmp;
		eos = (eos_type)tmp;
        arc & data_dir;

        arc & m2m_kernel_type;
        arc & p2p_kernel_type;
        arc & p2m_kernel_type;
        arc & cuda_streams_per_thread;
    }

    bool process_options(int argc, char* argv[]);

    static std::vector<hpx::id_type> all_localities;
};

#endif /* OPTIONS_HPP_ */
