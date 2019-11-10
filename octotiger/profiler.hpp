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
	    time_last = 6
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
    	const auto tinv = 1.0/ times_[time_total];
    	const auto thydro = times_[time_computation] - times_[time_fmm];
        std::cout << name << ":\n";
        std::cout << "   Total: "             << times_[time_total] << '\n';
        std::cout << "   Computation: "       << times_[time_computation] << " (" <<  100*times_[time_computation] * tinv << "\%)\n";
        std::cout << "   Gravity:     "       << times_[time_fmm]  << " (" <<  100*times_[time_fmm] * tinv << "\%)\n";
        std::cout << "   Hydro: "             << thydro  << " (" <<  thydro * tinv << "\%)\n";
        std::cout << "   Regrid: "            << times_[time_regrid]  << " (" <<  100*times_[time_regrid] * tinv << "\%)\n";
        std::cout << "   Compare Analytic: "  << times_[time_compare_analytic]  << " (" <<  100*times_[time_compare_analytic] * tinv << "\%)\n";
        std::cout << "   Find Localities: "   << times_[time_find_localities]  << " (" <<  100*times_[time_find_localities] * tinv << "\%)\n";
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
//#define PROFILE_OFF

#ifdef PROFILE_OFF
#define PROFILE()
#else
#define PROFILE() static profiler_register prof_reg(__FUNCTION__, __LINE__); \
	             profiler_scope __profile_object__(__FUNCTION__, __LINE__)
#endif


#endif /* PROFILER_HPP_ */
