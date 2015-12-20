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
//#define NFUTDEBUG

#define TIMEOUT 60.0

template<class T>
inline T __get(hpx::future<T> fut, const char fname[], int line) {
#ifndef NFUTDEBUG
	const auto timeout_time = boost::posix_time::time_duration(boost::posix_time::seconds(TIMEOUT));
	if( fut.wait_for(timeout_time) == hpx::lcos::future_status::timeout ) {
		printf("TIMEOUT WAITING ON FUTURE: FILE: %s, LINE: %i\n", fname, int(line));
		abort();
	}
#endif
	return fut.get();

}

inline void __get(hpx::future<void> fut, const char fname[], int line) {
#ifndef NFUTDEBUG
	const auto timeout_time = boost::posix_time::time_duration(boost::posix_time::seconds(TIMEOUT));
	if( fut.wait_for(timeout_time) == hpx::lcos::future_status::timeout ) {
		printf("TIMEOUT WAITING ON FUTURE: FILE: %s, LINE: %i\n", fname, int(line));
		abort();
	}
#endif
	fut.get();
}

#define GET(a) __get(std::move(a), __FILE__, __LINE__)

#endif /* FUTURE_HPP_ */
