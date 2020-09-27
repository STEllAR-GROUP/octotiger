#pragma once

//#define TVD_TEST

#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"

const recon_type<3>& reconstruct_experimental(const hydro::state_type &U_, const hydro::x_type &X, safe_real omega);

void reconstruct_ppm_experimental(std::vector<std::vector<safe_real>> &q, const std::vector<safe_real> &u, bool smooth, bool disc_detect, const std::vector<std::vector<double>> &disc);
