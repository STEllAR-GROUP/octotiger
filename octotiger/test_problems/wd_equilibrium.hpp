// Copyright (c) 2026 AUTHORS
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstddef>
#include <vector>

namespace octotiger::test_problems {

// Cold, nonrotating white dwarf with the repository's mu_e = 2 EOS.
// The code units have stellar mass = stellar radius = G = 1. The radial
// integration step is measured in sqrt(A_cgs / G_cgs) / B_cgs, not code units.
class wd_equilibrium_model {
public:
    static constexpr double solar_mass_cgs = 1.989e33;

    explicit wd_equilibrium_model(double radial_step = 2.0e-4);

    double mass_cgs() const;
    double radius_cgs() const;
    double code_a() const;
    double code_b() const;
    double central_density() const;
    double radial_step() const;
    size_t sample_count() const;

    // Radius is nonnegative and measured in stellar radii. Outside the star,
    // density and enthalpy are zero and enclosed mass is one. An atmosphere
    // belongs to the application initializer, not this equilibrium model.
    double density_at(double radius) const;
    double enthalpy_at(double radius) const;
    double enclosed_mass_at(double radius) const;

private:
    struct sample {
        double radius;
        double enthalpy;
        double enclosed_mass;
    };

    size_t interval_at(double radius) const;
    double density_from_enthalpy(double enthalpy) const;

    double radial_step_;
    double radius_cgs_;
    double code_a_;
    double code_b_;
    double central_density_;
    std::vector<sample> profile_;
};

wd_equilibrium_model const &one_solar_mass_wd();

} // namespace octotiger::test_problems
