/*
 * future.hpp
 *
 *  Created on: Sep 14, 2015
 *      Author: dmarce1
 */

#ifndef FUTURE3_HPP_
#define FUTURE3_HPP_

#include "defs.hpp"

//#include <boost/chrono.hpp>

#define TIMEOUT 60.0

template<class T>
inline T __get(hpx::future<T> fut, const char fname[], int line) {
#ifdef NFUTDEBUG
	return fut.get();
#else
	auto time_start = std::chrono::steady_clock::now();
	while (!fut.is_ready()) {
		double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
				std::chrono::steady_clock::now() - time_start).count();
		if (time_elapsed > TIMEOUT) {
			printf("TIMEOUT WAITING ON FUTURE: FILE: %s, LINE: %i\n", fname, int(line));
			abort();
		}
		hpx::this_thread::yield();
	}
	return fut.get();
#endif

}

inline void __get(hpx::future<void> fut, const char fname[], int line) {
#ifdef NFUTDEBUG
	return fut.get();
#else
	auto time_start = std::chrono::steady_clock::now();
	while (!fut.is_ready()) {
		double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
				std::chrono::steady_clock::now() - time_start).count();
		if (time_elapsed > TIMEOUT) {
			printf("TIMEOUT WAITING ON FUTURE: FILE: %s, LINE: %i\n", fname, int(line));
			abort();
		}
		hpx::this_thread::yield();
	}
	fut.get();
#endif
}

#define GET(a) __get(std::move(a), __FILE__, __LINE__)

#endif /* FUTURE_HPP_ */
