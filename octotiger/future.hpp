//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef FUTURE3_HPP_
#define FUTURE3_HPP_

#include "octotiger/options.hpp"

#include <hpx/include/lcos.hpp>

#include <array>
#include <vector>

template<class T>
using future = hpx::future<T>;

template<typename T>
inline void propagate_exceptions(hpx::future<T> &f) {
	if (f.has_exception())
		f.get();                // rethrow
}

template<typename T>
inline void propagate_exceptions(hpx::future<T> const &f) {
	typename hpx::traits::detail::shared_state_ptr_for<hpx::future<T> >::type state = hpx::traits::future_access<hpx::future<T> >::get_shared_state(f);
	if (state->has_exception())
		state->get_result();     // will rethrow exception
}

template<typename T>
inline void propagate_exceptions(std::vector<hpx::future<T> > &futs) {
	for (auto &f : futs)
		propagate_exceptions(f);
}

template<typename T>
inline void propagate_exceptions(std::vector<hpx::future<T> > const &futs) {
	for (auto const &f : futs)
		propagate_exceptions(f);
}

template<typename T, std::size_t N>
inline void propagate_exceptions(std::array<hpx::future<T>, N> &futs) {
	for (auto &f : futs)
		propagate_exceptions(f);
}

template<typename T, std::size_t N>
inline void propagate_exceptions(std::array<hpx::future<T>, N> const &futs) {
	for (auto const &f : futs)
		propagate_exceptions(f);
}

template<typename Ts>
inline void wait_all_and_propagate_exceptions(Ts &&futs) {
	hpx::wait_all(futs);
	for (auto &f : futs) {
		f.get();
	}
	/*  int const sequencer[] = {
	 0, (propagate_exceptions(futs), 0) ...
	 };
	 (void)sequencer;*/
}

/*template<class T>
inline T debug_get(future<T> &f, const char *file, int line) {
	if (f.valid() == false) {
		print("get on invalid future file %s line %i\n", file, line);
	}
	const int timeout = opts().future_wait_time;
	int count = 0;
	if (timeout > 0) {
		while (f.wait_for(std::chrono::duration<int>(timeout)) == hpx::lcos::future_status::timeout) {
			count++;
			print("future::get in file %s on line %i is taking a while - %i seconds so far.\n", file, line, 60 * count);
		}
	}
	return f.get();
}*/

/*template<class T>
inline T debug_get(future<T> &&_f, const char *file, int line) {
	future<T> f = std::move(_f);
	return debug_get(f, file, line);
}

template<class T>
inline T debug_get(const hpx::shared_future<T> &f, const char *file, int line) {
	if (f.valid() == false) {
		print("get on invalid future file %s line %i\n", file, line);
	}
// something wrong with hpx::future::wait_for
//	constexpr int timeout = 10;
//	int count = 0;
//	if (!f.is_ready()) {
//		while (f.wait_for(std::chrono::duration<int>(timeout)) == hpx::lcos::future_status::timeout) {
//			count++;
//			print("shared_future::get in file %s on line %i is taking a while - %i seconds so far.\n", file, line, 60 * count);
//		}
//	}
	return f.get();
}*/

//#ifndef NDEBUG
//#define GET(fut) debug_get(fut,__FILE__,__LINE__)
//#else
//#define GET(fut) ((fut).get())
//#endif

//#define GET(fut) debug_get(fut,__FILE__,__LINE__)
#define GET(fut) fut.get()

#endif /* FUTURE_HPP_ */
