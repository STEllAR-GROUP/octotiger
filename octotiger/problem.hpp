//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef PROBLEM_HPP_
#define PROBLEM_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/math/Real.hpp"

#include <array>
#include <functional>
#include <vector>

using init_func_type = std::function<std::vector<Real>(Real, Real, Real, Real)>;
using analytic_func_type = init_func_type;
using refine_test_type = std::function<bool(integer, integer, Real, Real, Real,
    std::vector<Real> const&, std::array<std::vector<Real>, NDIM> const&)>;

const static init_func_type null_problem = nullptr;
OCTOTIGER_EXPORT std::vector<Real> old_scf(
    Real, Real, Real, Real, Real, Real, Real);
OCTOTIGER_EXPORT std::vector<Real> blast_wave(Real, Real, Real, Real);
#if defined(OCTOTIGER_HAVE_BLAST_TEST)
OCTOTIGER_EXPORT std::vector<Real> blast_wave_analytic(
    Real x, Real y, Real z, Real t);
#endif
OCTOTIGER_EXPORT std::vector<Real> advection_test_init(Real,Real,Real,Real);
OCTOTIGER_EXPORT std::vector<Real> advection_test_analytic(Real,Real,Real,Real);

OCTOTIGER_EXPORT std::vector<Real> sod_shock_tube_init(Real, Real, Real, Real);
OCTOTIGER_EXPORT std::vector<Real> sod_shock_tube_analytic(
    Real, Real, Real, Real);
OCTOTIGER_EXPORT std::vector<Real> marshak_wave(Real, Real, Real, Real);
OCTOTIGER_EXPORT std::vector<Real> marshak_wave_analytic(Real, Real, Real, Real);
OCTOTIGER_EXPORT std::vector<Real> star(Real, Real, Real, Real);
OCTOTIGER_EXPORT std::vector<Real> moving_star_analytic(Real, Real, Real, Real);
OCTOTIGER_EXPORT std::vector<Real> moving_star(Real, Real, Real, Real);
OCTOTIGER_EXPORT std::vector<Real> equal_mass_binary(Real, Real, Real, Real);
OCTOTIGER_EXPORT std::vector<Real> scf_binary(Real, Real, Real, Real);
//std::vector<Real> null_problem(Real x, Real y, Real z, Real);
OCTOTIGER_EXPORT std::vector<Real> solid_sphere(Real, Real, Real, Real, Real);
OCTOTIGER_EXPORT std::vector<Real> solid_sphere_analytic_phi(
    Real x, Real y, Real z, Real);
OCTOTIGER_EXPORT std::vector<Real> double_solid_sphere(Real, Real, Real, Real);
OCTOTIGER_EXPORT std::vector<Real> double_solid_sphere_analytic_phi(
    Real x, Real y, Real z);

OCTOTIGER_EXPORT bool refine_test_center(integer level, integer maxl, Real, Real,
    Real, std::vector<Real> const& U,
    std::array<std::vector<Real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_test(integer level, integer maxl, Real, Real, Real,
    std::vector<Real> const& U,
    std::array<std::vector<Real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_test_marshak(integer level, integer maxl, Real, Real,
    Real, std::vector<Real> const& U,
    std::array<std::vector<Real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_test_moving_star(integer level, integer maxl, Real,
    Real, Real, std::vector<Real> const& U,
    std::array<std::vector<Real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_sod(integer level, integer max_level, Real x, Real y,
    Real z, std::vector<Real> const& U,
    std::array<std::vector<Real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_blast(integer level, integer max_level, Real x,
    Real y, Real z, std::vector<Real> const& U,
    std::array<std::vector<Real>, NDIM> const& dudx);

OCTOTIGER_EXPORT void set_refine_test(const refine_test_type&);
OCTOTIGER_EXPORT refine_test_type get_refine_test();
OCTOTIGER_EXPORT void set_problem(const init_func_type&);
OCTOTIGER_EXPORT void set_analytic(const analytic_func_type&);
OCTOTIGER_EXPORT init_func_type get_problem();
OCTOTIGER_EXPORT analytic_func_type get_analytic();

OCTOTIGER_EXPORT bool radiation_test_refine(integer level, integer max_level,
    Real x, Real y, Real z, std::vector<Real> U,
    std::array<std::vector<Real>, NDIM> const& dudx);
OCTOTIGER_EXPORT std::vector<Real> radiation_test_problem(Real, Real, Real, Real);
std::vector<Real> radiation_diffusion_test_problem(Real x, Real y, Real z, Real dx);
std::vector<Real> radiation_coupling_test_problem(Real x, Real y, Real z, Real dx);
std::vector<Real> radiation_diffusion_analytic(Real x, Real y, Real z, Real t);
#endif /* PROBLEM_HPP_ */
