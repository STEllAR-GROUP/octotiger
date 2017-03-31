/*
 * profiler.hpp
 *
 *  Created on: Sep 13, 2016
 *      Author: dmarce1
 */

#ifndef PROFILER_HPP_
#define PROFILER_HPP_

#include "defs.hpp"
#include "options.hpp"
#include "tsb_fp_equals.hpp"

#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/serialization.hpp>

#include <algorithm>
#include <array>
#include <iostream>

template <typename Target>
struct op_stats_t
{
    double time;
#if !defined(OCTOTIGER_HAVE_DISABLE_OPS_TRACKING)
    integer fp_adds;
    integer fp_muls;
    integer fp_fmas;
    integer fp_divs;
    integer fp_sqrts;
    integer fp_memloads; 
    integer fp_memstores; 
    integer fp_tileloads; 
    integer fp_tilestores; 
    integer fp_cacheloads; 
    integer fp_cachestores; 
#endif

    constexpr op_stats_t() :
        time{0.0}
#if !defined(OCTOTIGER_HAVE_DISABLE_OPS_TRACKING)
      , fp_adds{0}
      , fp_muls{0}
      , fp_fmas{0}
      , fp_divs{0}
      , fp_sqrts{0}
      , fp_memloads{0} 
      , fp_memstores{0} 
      , fp_tileloads{0} 
      , fp_tilestores{0} 
      , fp_cacheloads{0} 
      , fp_cachestores{0}
#endif
    {}

    constexpr void add_time(double time_) noexcept { time += time_; }

#if !defined(OCTOTIGER_HAVE_DISABLE_OPS_TRACKING)
    constexpr void add_fp_adds(integer fp_adds_) noexcept { fp_adds += fp_adds_; }
    constexpr void add_fp_muls(integer fp_muls_) noexcept { fp_muls += fp_muls_; }
    constexpr void add_fp_fmas(integer fp_fmas_) noexcept { fp_fmas += fp_fmas_; }
    constexpr void add_fp_divs(integer fp_divs_) noexcept { fp_divs += fp_divs_; }
    constexpr void add_fp_sqrts(integer fp_sqrts_) noexcept { fp_sqrts += fp_sqrts_; }
    constexpr void add_fp_memloads(integer fp_memloads_) noexcept { fp_memloads += fp_memloads_; }
    constexpr void add_fp_memstores(integer fp_memstores_) noexcept { fp_memstores += fp_memstores_; }
    constexpr void add_fp_tileloads(integer fp_tileloads_) noexcept { fp_tileloads += fp_tileloads_; }
    constexpr void add_fp_tilestores(integer fp_tilestores_) noexcept { fp_tilestores += fp_tilestores_; }
    constexpr void add_fp_cacheloads(integer fp_cacheloads_) noexcept { fp_cacheloads += fp_cacheloads_; }
    constexpr void add_fp_cachestores(integer fp_cachestores_) noexcept { fp_cachestores += fp_cachestores_; }
#else
    constexpr void add_fp_adds(integer fp_adds_) noexcept { }
    constexpr void add_fp_muls(integer fp_muls_) noexcept { }
    constexpr void add_fp_fmas(integer fp_fmas_) noexcept { }
    constexpr void add_fp_divs(integer fp_divs_) noexcept { }
    constexpr void add_fp_sqrts(integer fp_sqrts_) noexcept { }
    constexpr void add_fp_memloads(integer fp_memloads_) noexcept { }
    constexpr void add_fp_memstores(integer fp_memstores_) noexcept { } 
    constexpr void add_fp_tileloads(integer fp_tileloads_) noexcept { }
    constexpr void add_fp_tilestores(integer fp_tilestores_) noexcept { }
    constexpr void add_fp_cacheloads(integer fp_cacheloads_) noexcept { }
    constexpr void add_fp_cachestores(integer fp_cachestores_) noexcept { }
#endif

    op_stats_t& operator+=(op_stats_t const& rhs) noexcept
    {
        time += rhs.time;
    #if !defined(OCTOTIGER_HAVE_DISABLE_OPS_TRACKING)
        fp_adds += rhs.fp_adds;
        fp_muls += rhs.fp_muls;
        fp_fmas += rhs.fp_fmas;
        fp_divs += rhs.fp_divs;
        fp_sqrts += rhs.fp_sqrts;
        fp_memloads += rhs.fp_memloads; 
        fp_memstores += rhs.fp_memstores; 
        fp_tileloads += rhs.fp_tileloads; 
        fp_tilestores += rhs.fp_tilestores; 
        fp_cacheloads += rhs.fp_cacheloads;
        fp_cachestores += rhs.fp_cachestores;
    #endif
        return *this;
    }

    op_stats_t operator+(op_stats_t const& rhs) const noexcept
    {
        op_stats_t tmp = *this;
        tmp += rhs;
        return tmp;
    }

	template <typename Archive>
	void serialize(Archive& ar, unsigned)
    {
        ar & time;
    #if !defined(OCTOTIGER_HAVE_DISABLE_OPS_TRACKING)
        ar & fp_adds;
        ar & fp_muls;
        ar & fp_fmas;
        ar & fp_divs;
        ar & fp_sqrts;
        ar & fp_memloads; 
        ar & fp_memstores; 
        ar & fp_tileloads; 
        ar & fp_tilestores; 
        ar & fp_cacheloads; 
        ar & fp_cachestores; 
    #endif
	}

    void print(std::ostream& os = std::cout) const
    {
    #if !defined(OCTOTIGER_HAVE_DISABLE_OPS_TRACKING)
        // KNL AVX512
        double const gflops       = double(fp_adds + fp_muls + 2 * fp_fmas + 10 * fp_divs + 2 * fp_sqrts) / double(1024 * 1024 * 1024);
        double const mem_gbytes   = double(sizeof(real) * fp_memloads + sizeof(real) * fp_memstores) / double(1024 * 1024 * 1024);
        double const tile_gbytes  = double(sizeof(real) * fp_tileloads + sizeof(real) * fp_tilestores) / double(1024 * 1024 * 1024);
        double const cache_gbytes = double(sizeof(real) * fp_cacheloads + sizeof(real) * fp_cachestores) / double(1024 * 1024 * 1024);
        double const gflopss       = (tsb::fp_equals(time, 0.0) ? 0.0 : gflops / time);
        double const mem_gbytess   = (tsb::fp_equals(time, 0.0) ? 0.0 : mem_gbytes / time);
        double const tile_gbytess  = (tsb::fp_equals(time, 0.0) ? 0.0 : tile_gbytes / time);
        double const cache_gbytess = (tsb::fp_equals(time, 0.0) ? 0.0 : cache_gbytes / time);
    #endif

        os << Target::name() << " stats:\n"
           << "time:           " << time << " [s]\n"
    #if !defined(OCTOTIGER_HAVE_DISABLE_OPS_TRACKING)
           << "fp_adds:        " << fp_adds << "\n"
           << "fp_muls:        " << fp_muls << "\n"
           << "fp_fmas:        " << fp_fmas << "\n"
           << "fp_divs:        " << fp_divs << "\n"
           << "fp_sqrts:       " << fp_sqrts << "\n"
           << "fp_memloads:    " << fp_memloads << "\n" 
           << "fp_memstores:   " << fp_memstores << "\n" 
           << "fp_tileloads:   " << fp_tileloads << "\n" 
           << "fp_tilestores:  " << fp_tilestores << "\n" 
           << "fp_cacheloads:  " << fp_cacheloads << "\n"
           << "fp_cachestores: " << fp_cachestores << "\n"
           << "gflops:         " << gflops        << " [GFLOPS]\n"
           << "mem_gbytes:     " << mem_gbytes    << " [Gbytes]\n"
           << "tile_gbytes:    " << tile_gbytes   << " [Gbytes]\n" 
           << "cache_gybtes:   " << cache_gbytes  << " [Gbytes]\n" 
           << "gflopss:        " << gflopss       << " [GFLOPS/s]\n"
           << "mem_gbytess:    " << mem_gbytess   << " [Gbytes/s]\n"
           << "tile_gbytess:   " << tile_gbytess  << " [Gbytes/s]\n" 
           << "cache_gybtess:  " << cache_gbytess << " [Gbytes/s]\n" 
    #endif
           ;
	}
};

struct compute_interactions_stats_t : op_stats_t<compute_interactions_stats_t>
{
    static char const* name() noexcept
    {
        if (opts.compute_interactions_legacy)
            return "compute_interactions_legacy";
        else
            return "compute_interactions";
    }
}; 

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
        time_last = 5
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
        std::cout << "   Total: "             << times_[time_total] << '\n';
        std::cout << "   Computation: "       << times_[time_computation] << '\n';
        std::cout << "   Regrid: "            << times_[time_regrid] << '\n';
        std::cout << "   Compare Analytic: "  << times_[time_compare_analytic] << '\n';
        std::cout << "   Find Localities: "   << times_[time_find_localities] << '\n';
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
