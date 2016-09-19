/*
 * profiler.hpp
 *
 *  Created on: Sep 13, 2016
 *      Author: dmarce1
 */

#ifndef PROFILER_HPP_
#define PROFILER_HPP_

#include "defs.hpp"


struct profiler_register {
	profiler_register(const char*);
	profiler_register(const char*, const char*);
};
void profiler_enter(const char* func);
void profiler_sub_enter(const char* func, const char* label);
void profiler_exit();
void profiler_output(FILE* fp);


//#define PROFILE_OFF

#ifdef PROFILE_OFF
#define PROF_BEGIN
#define PROF_END
#else
#define PROF_BEGIN static profiler_register prof_reg(__FUNCTION__); \
	                       profiler_enter(__FUNCTION__)
#define PROF_SUB_BEGIN(label) static profiler_register prof_reg(__FUNCTION__, label); \
	                           profiler_sub_enter(__FUNCTION__, label)
#define PROF_END profiler_exit()
#endif


#endif /* PROFILER_HPP_ */
