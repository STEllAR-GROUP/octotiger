/*
 * profiler.hpp
 *
 *  Created on: Sep 13, 2016
 *      Author: dmarce1
 */

#ifndef PROFILER_HPP_
#define PROFILER_HPP_

#include "defs.hpp"


void profiler_enter(const char* func);
void profiler_exit();
void profiler_output(FILE* fp);

#define PROFILE_OFF

#ifdef PROFILE_OFF
#define PROF_BEGIN
#define PROF_END
#else
#define PROF_BEGIN profiler_enter(__FUNCTION__)
#define PROF_END profiler_exit()
#endif


#endif /* PROFILER_HPP_ */
