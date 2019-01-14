/*
 * problem.hpp
 *
 *  Created on: May 29, 2015
 *      Author: dmarce1
 */

#ifndef PROBLEM_HPP_
#define PROBLEM_HPP_

#include "octotiger/defs.hpp"
#include "octotiger/real.hpp"

#include <array>
#include <functional>
#include <vector>

using init_func_type = std::function<std::vector<real>(real,real,real,real)>;
using analytic_func_type = init_func_type;
using refine_test_type = std::function<bool(integer,integer, real,real,real,std::vector<real> const&,std::array<std::vector<real>,NDIM> const&)>;

const static init_func_type null_problem = nullptr;
std::vector<real> old_scf(real, real, real, real, real, real, real);
std::vector<real> blast_wave(real, real, real, real);
std::vector<real> sod_shock_tube_init(real, real, real, real);
std::vector<real> sod_shock_tube_analytic(real, real, real, real);
std::vector<real> marshak_wave(real, real, real, real);
std::vector<real> marshak_wave_analytic(real, real, real, real);
std::vector<real> blast_wave_analytic(real x, real y, real z, real t);
std::vector<real> star(real, real, real, real);
std::vector<real> moving_star_analytic(real, real, real, real);
std::vector<real> moving_star(real, real, real, real);
std::vector<real> equal_mass_binary(real, real, real, real);
std::vector<real> scf_binary(real, real, real, real);
//std::vector<real> null_problem(real x, real y, real z, real);
std::vector<real> solid_sphere(real, real, real, real, real);
std::vector<real> solid_sphere_analytic_phi(real x, real y, real z, real);
std::vector<real> double_solid_sphere(real, real, real, real);
std::vector<real> double_solid_sphere_analytic_phi(real x, real y, real z);

bool refine_test(integer level, integer maxl, real,real,real, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx);
bool refine_test_marshak(integer level, integer maxl, real,real,real, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx);
bool refine_test_moving_star(integer level, integer maxl, real,real,real, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx);
bool refine_sod(integer level, integer max_level, real x, real y, real z, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx);
bool refine_blast(integer level, integer max_level, real x, real y, real z, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx);

void set_refine_test(const refine_test_type&);
refine_test_type get_refine_test();
void set_problem(const init_func_type&);
void set_analytic(const analytic_func_type&);
init_func_type get_problem();
analytic_func_type get_analytic();

bool radiation_test_refine(integer level, integer max_level, real x, real y, real z, std::vector<real> U, std::array<std::vector<real>, NDIM> const& dudx);
std::vector<real> radiation_test_problem(real,real,real,real);


#endif /* PROBLEM_HPP_ */
