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



#define TIMEOUT 30.0

template<class T>
inline T __get(hpx::future<T> fut, const char fname[], int line) {
//#ifdef NDEBUG
	return fut.get();
/*#else
	auto time_start = std::chrono::steady_clock::now();
	auto flag = std::make_shared<bool>(false);
	hpx::thread([=](){
		double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - time_start).count();
		double sleep_time = TIMEOUT - time_elapsed;
		hpx::this_thread::sleep_for(std::chrono::milliseconds(int(sleep_time*1000)));
		if( !(*flag) ) {
			printf( "TIMEOUT WAITING ON FUTURE: FILE: %s, LINE: %i\n", fname, int(line));
		}
	}).detach();
	auto data = std::make_shared<T>(fut.get());
	*flag = true;
	return std::move(*data);
#endif*/
}



inline void __get(hpx::future<void> fut, const char fname[], int line) {
//#ifdef NDEBUG
	fut.get();
/*#else
	auto time_start = std::chrono::steady_clock::now();
	auto flag = std::make_shared<bool>(false);
	hpx::thread([=](){
		double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - time_start).count();
		double sleep_time = TIMEOUT - time_elapsed;
		hpx::this_thread::sleep_for(std::chrono::milliseconds(int(sleep_time*1000)));
		if( !(*flag) ) {
			printf( "TIMEOUT WAITING ON FUTURE: FILE: %s, LINE: %i\n", fname, int(line));
		}
	}).detach();
	fut.get();
	*flag = true;
#endif*/
}



#define GET(a) __get(std::move(a), __FILE__, __LINE__)

#endif /* FUTURE_HPP_ */
