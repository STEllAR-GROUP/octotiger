//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef PROFILER_HPP_
#define PROFILER_HPP_

#include <hpx/include/serialization.hpp>
//#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/timing.hpp>

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
        time_compare_analytic = 3,
        time_find_localities = 4,
		  time_fmm = 5,
		  time_io = 6,
	     time_last = 7
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

        hpx::chrono::high_resolution_timer timer_;
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
        if (times_[time_total] > 0.0) {
            const auto tinv = 1.0/ times_[time_total];
            const auto tcr = times_[time_computation] + times_[time_regrid];
            std::cout << name << ":" << std::endl;
            std::cout << "   Total: "             << times_[time_total] << std::endl;
            std::cout << "   Computation: "       << times_[time_computation] << " (" <<  100*times_[time_computation] * tinv << " %)" << std::endl;
            std::cout << "   Regrid: "            << times_[time_regrid]  << " (" <<  100*times_[time_regrid] * tinv << " %)" << std::endl;
            std::cout << "   Computation + Regrid: "       << tcr << " (" <<  100*tcr * tinv << " %)" << std::endl;
        } else {
            std::cout << "   Warning! Total time is 0! " << std::endl;
        }
    }

    std::array<double, timer::time_last> times_;
};

struct profiler_scope {
	profiler_scope(const char* function, int line) {
		profiler_enter(function, line);
	}
	~profiler_scope() {
		profiler_exit();
	}
};
#define PROFILE_OFF

#ifdef PROFILE_OFF
#define PROFILE()
#else
#define PROFILE() static profiler_register prof_reg(__FUNCTION__, __LINE__); \
	             profiler_scope __profile_object__(__FUNCTION__, __LINE__)
#endif


#endif /* PROFILER_HPP_ */
