/*
 * profiler.hpp
 *
 *  Created on: Sep 13, 2016
 *      Author: dmarce1
 */

#ifndef PROFILER_HPP_
#define PROFILER_HPP_

#include "defs.hpp"

#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/serialization.hpp>

#include <algorithm>
#include <array>
#include <iostream>

struct profiler_register {
	profiler_register(const char*, int);
};
void profiler_enter(const char* func, int line);
void profiler_exit();
void profiler_output(FILE* fp);

struct timings
{
    enum timer {
        time_total = 0,
        time_computation = 1,
        time_regrid = 2,
        time_find_localities = 3,
        time_last = 4
    };

    struct scope
    {
        scope(timings &t, timer tt)
          : time_(t.times_[tt])
        {
        }

        ~scope()
        {
            time_ += timer_.elapsed();
        }

        hpx::util::high_resolution_timer timer_;
        double& time_;
    };

    timings()
    {
        for (std::size_t i = 0; i < timer::time_last; ++i)
        {
            times_[i] = 0.0;
        }
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & times_;
    }

    void min(timings const& other)
    {
        for (std::size_t i = 0; i < timer::time_last; ++i)
        {
            times_[i] = (std::min)(times_[i], other.times_[i]);
        }
    }

    void max(timings const& other)
    {
        for (std::size_t i = 0; i < timer::time_last; ++i)
        {
            times_[i] = (std::max)(times_[i], other.times_[i]);
        }
    }

    void report(std::string const& name)
    {
        std::cout << name << ":\n";
        std::cout << "   Total: "           << times_[time_total] << '\n';
        std::cout << "   Computation: "     << times_[time_computation] << '\n';
        std::cout << "   Regrid: "          << times_[time_regrid] << '\n';
        std::cout << "   Find Localities: " << times_[time_find_localities] << '\n';
    }

    std::array<double, timer::time_last> times_;
};

#define PROFILE_OFF

#ifdef PROFILE_OFF
#define PROF_BEGIN
#define PROF_END
#else
#define PROF_BEGIN static profiler_register prof_reg(__FUNCTION__, __LINE__); \
	                       profiler_enter(__FUNCTION__, __LINE__)
#define PROF_END profiler_exit()
#endif


#endif /* PROFILER_HPP_ */
