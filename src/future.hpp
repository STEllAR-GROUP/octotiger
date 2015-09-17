/*
 * future.hpp
 *
 *  Created on: Sep 14, 2015
 *      Author: dmarce1
 */

#ifndef FUTURE3_HPP_
#define FUTURE3_HPP_

#include "defs.hpp"

#include <mpi.h>
#include <thread>
#include <chrono>

#define TIMEOUT 30.0

template<class T>
inline T get(hpx::future<T>& fut, const char* fname, integer line) {
//	return fut.get();
	double time_start = MPI_Wtime();
	auto flag = std::make_shared<bool>(false);
	hpx::thread([=](){
		double sleep_time = TIMEOUT - (MPI_Wtime() - time_start);
		hpx::this_thread::sleep_for(boost::chrono::milliseconds(int(sleep_time*1000)));
		if( !(*flag) ) {
			printf( "TIMEOUT WAITING ON FUTURE: FILE: %s, LINE: %i\n", fname, int(line));
		}
	}).detach();
	auto data = std::make_shared<T>(fut.get());
	*flag = true;
	return std::move(*data);
}


inline void get(hpx::future<void>& fut, const char* fname, integer line) {
//	fut.get();
	double time_start = MPI_Wtime();
	auto flag = std::make_shared<bool>(false);
	hpx::thread([=](){
		double sleep_time = TIMEOUT - (MPI_Wtime() - time_start);
		hpx::this_thread::sleep_for(boost::chrono::milliseconds(int(sleep_time*1000)));
		if( !(*flag) ) {
			printf( "TIMEOUT WAITING ON FUTURE: FILE: %s, LINE: %i\n", fname, int(line));
		}
	}).detach();
	fut.get();
	*flag = true;
}


#define GET(a) get(a, __FILE__, __LINE__)

#endif /* FUTURE_HPP_ */
