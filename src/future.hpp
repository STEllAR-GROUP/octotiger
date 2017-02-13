/*
 * future.hpp
 *
 *  Created on: Sep 14, 2015
 *      Author: dmarce1
 */

#ifndef FUTURE3_HPP_
#define FUTURE3_HPP_

#include "defs.hpp"
#include <hpx/include/lcos.hpp>

#include <array>
#include <vector>

template <typename T>
inline void propagate_exceptions(hpx::future<T>& f)
{
    if (f.has_exception())
        f.get();                // rethrow
}

template <typename T>
inline void propagate_exceptions(hpx::future<T> const& f)
{
    typename hpx::traits::detail::shared_state_ptr_for<hpx::future<T> >::type
        state = hpx::traits::future_access<hpx::future<T> >::get_shared_state(f);
    if (state->has_exception())
        state->get_result();     // will rethrow exception
}

template <typename T>
inline void propagate_exceptions(std::vector<hpx::future<T> >& futs)
{
    for (auto& f : futs)
        propagate_exceptions(f);
}

template <typename T>
inline void propagate_exceptions(std::vector<hpx::future<T> > const& futs)
{
    for (auto const& f : futs)
        propagate_exceptions(f);
}

template <typename T, std::size_t N>
inline void propagate_exceptions(std::array<hpx::future<T>, N>& futs)
{
    for (auto& f : futs)
        propagate_exceptions(f);
}

template <typename T, std::size_t N>
inline void propagate_exceptions(std::array<hpx::future<T>, N> const& futs)
{
    for (auto const& f : futs)
        propagate_exceptions(f);
}

template <typename ... Ts>
inline void wait_all_and_propagate_exceptions(Ts&& ... futs)
{
    hpx::wait_all(futs...);
    int const sequencer[] = {
        0, (propagate_exceptions(futs), 0) ...
    };
    (void)sequencer;
}

#endif /* FUTURE_HPP_ */
