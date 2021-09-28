//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef PROBLEM_HPP_
#define PROBLEM_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/real.hpp"

#include <array>
#include <functional>
#include <vector>

using init_func_type = std::function<std::vector<real>(real, real, real, real)>;
using analytic_func_type = init_func_type;
using refine_test_type = std::function<bool(integer, integer, real, real, real,
    std::vector<real> const&, std::array<std::vector<real>, NDIM> const&)>;

const static init_func_type null_problem = nullptr;
OCTOTIGER_EXPORT std::vector<real> old_scf(
    real, real, real, real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> blast_wave(real, real, real, real);
#if defined(OCTOTIGER_HAVE_BLAST_TEST)
OCTOTIGER_EXPORT std::vector<real> blast_wave_analytic(
    real x, real y, real z, real t);
#endif
OCTOTIGER_EXPORT std::vector<real> advection_test_init(real,real,real,real);
OCTOTIGER_EXPORT std::vector<real> advection_test_analytic(real,real,real,real);

OCTOTIGER_EXPORT std::vector<real> sod_shock_tube_init(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> sod_shock_tube_analytic(
    real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> marshak_wave(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> marshak_wave_analytic(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> star(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> moving_star_analytic(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> moving_star(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> equal_mass_binary(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> scf_binary(real, real, real, real);
//std::vector<real> null_problem(real x, real y, real z, real);
OCTOTIGER_EXPORT std::vector<real> solid_sphere(real, real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> solid_sphere_analytic_phi(
    real x, real y, real z, real);
OCTOTIGER_EXPORT std::vector<real> double_solid_sphere(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> double_solid_sphere_analytic_phi(
    real x, real y, real z);

OCTOTIGER_EXPORT bool refine_test_center(integer level, integer maxl, real, real,
    real, std::vector<real> const& U,
    std::array<std::vector<real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_test(integer level, integer maxl, real, real, real,
    std::vector<real> const& U,
    std::array<std::vector<real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_test_marshak(integer level, integer maxl, real, real,
    real, std::vector<real> const& U,
    std::array<std::vector<real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_test_moving_star(integer level, integer maxl, real,
    real, real, std::vector<real> const& U,
    std::array<std::vector<real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_sod(integer level, integer max_level, real x, real y,
    real z, std::vector<real> const& U,
    std::array<std::vector<real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_blast(integer level, integer max_level, real x,
    real y, real z, std::vector<real> const& U,
    std::array<std::vector<real>, NDIM> const& dudx);

OCTOTIGER_EXPORT void set_refine_test(const refine_test_type&);
OCTOTIGER_EXPORT refine_test_type get_refine_test();
OCTOTIGER_EXPORT void set_problem(const init_func_type&);
OCTOTIGER_EXPORT void set_analytic(const analytic_func_type&);
OCTOTIGER_EXPORT init_func_type get_problem();
OCTOTIGER_EXPORT analytic_func_type get_analytic();

OCTOTIGER_EXPORT bool radiation_test_refine(integer level, integer max_level,
    real x, real y, real z, std::vector<real> U,
    std::array<std::vector<real>, NDIM> const& dudx);
OCTOTIGER_EXPORT std::vector<real> radiation_test_problem(real, real, real, real);

#endif /* PROBLEM_HPP_ */
