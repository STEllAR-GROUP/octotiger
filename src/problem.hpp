/*
 * problem.hpp
 *
 *  Created on: May 29, 2015
 *      Author: dmarce1
 */

#ifndef PROBLEM_HPP_
#define PROBLEM_HPP_

#include "defs.hpp"
#include <vector>

using init_func_type = std::function<std::vector<real>(real,real,real,real)>;
using refine_test_type = std::function<bool(integer,integer, real,real,real,std::vector<real>)>;

std::vector<real> old_scf(real, real, real, real, real, real, real);
std::vector<real> sod_shock_tube(real, real, real, real);
std::vector<real> star(real, real, real, real);
std::vector<real> equal_mass_binary(real, real, real, real);
std::vector<real> scf_binary(real, real, real, real);
std::vector<real> null_problem(real x, real y, real z, real);
std::vector<real> solid_sphere(real, real, real, real, real);
std::vector<real> solid_sphere_analytic_phi(real x, real y, real z, real);
std::vector<real> double_solid_sphere(real, real, real, real);
std::vector<real> double_solid_sphere_analytic_phi(real x, real y, real z);

bool refine_test(integer level, integer maxl, real,real,real, std::vector<real> state);
bool refine_test_bibi(integer level, integer maxl, real,real,real, std::vector<real> state);

void set_refine_test(const refine_test_type&);
refine_test_type get_refine_test();
void set_problem(const init_func_type&);
init_func_type get_problem();

#endif /* PROBLEM_HPP_ */
