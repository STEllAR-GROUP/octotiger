// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "octotiger/fld_material.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/options.hpp"
#include "octotiger/roe.hpp"

namespace octotiger::fld {

inline bool fld_fully_ionized_material() {
    auto const& model = opts().fld_material_model;
    if (model == "fully_ionized") {
        return true;
    }
    if (model == "constant_cv") {
        return false;
    }
    if (model == "auto") {
        return opts().eos == WD;
    }
    throw std::invalid_argument("Unknown FLD material model");
}

inline bool fld_physical_units() {
    auto const& units = opts().fld_unit_system;
    if (units == "physical") {
        return true;
    }
    if (units == "code") {
        return false;
    }
    if (units == "auto") {
        return opts().eos == WD || fld_fully_ionized_material() ||
            opts().fld_opacity_model == "free_free_thomson";
    }
    throw std::invalid_argument("Unknown FLD unit system");
}

inline MaterialUnits fld_unit_conversions() {
    return physical_units(opts().code_to_g, opts().code_to_cm, opts().code_to_s);
}

inline double fld_effective_c() {
    return fld_physical_units() ? fld_unit_conversions().c : double(opts().fld_c);
}

inline double fld_effective_ar() {
    return fld_physical_units() ? fld_unit_conversions().ar : double(opts().fld_ar);
}

// Getter(field, index) supports both grid::get_field and direct state vectors.
template <class Getter>
ThermalState fld_material_state_from_fields(Getter const& field, size_t const index) {
    if (opts().eos != IDEAL && opts().eos != WD) {
        throw std::invalid_argument("FLD material recovery supports only IDEAL and WD");
    }
    double const density = field(rho_i, index);
    if (!(density > 0.0) || !std::isfinite(density)) {
        throw std::runtime_error("FLD requires finite positive gas density");
    }
    long double kinetic = 0.0L;
    for (int dim = 0; dim < NDIM; ++dim) {
        double const momentum = field(sx_i + dim, index);
        if (!std::isfinite(momentum)) {
            throw std::runtime_error("FLD requires finite gas momentum");
        }
        kinetic += 0.5L * momentum * momentum / density;
    }
    double const cold = opts().eos == WD ? double(ztwd_energy(density)) : 0.0;
    CompositionMoments composition;
    bool const fullyIonized = fld_fully_ionized_material();
    bool const needsComposition = fullyIonized || opts().fld_opacity_model == "free_free_thomson";
    if (needsComposition) {
        size_t const count = opts().n_species;
        if (opts().n_species < 1 || opts().atomic_mass.size() < count ||
            opts().atomic_number.size() < count || opts().atomic_number_squared.size() != count) {
            throw std::runtime_error("FLD requires atomic masses and charge moments for every species");
        }
        long double speciesDensity = 0.0L;
        for (size_t species = 0; species < count; ++species) {
            double const value = field(spc_i + species, index);
            if (!(value >= 0.0) || !std::isfinite(value)) {
                throw std::runtime_error("FLD requires finite nonnegative species densities");
            }
            speciesDensity += value;
        }
        if (!(speciesDensity > 0.0L) || !std::isfinite(speciesDensity)) {
            throw std::runtime_error("FLD requires positive total species density");
        }
        // Match mean_ion_weight: normalize by the tracer total, not hydro rho.
        // The WD SCF currently adds its atmosphere tracer to stellar tracers,
        // so their total can differ from rho by the density floor.
        composition = composition_moments(count,
            [&](size_t const species) {
                return static_cast<double>(field(spc_i + species, index) / speciesDensity);
            },
            [&](size_t const species) { return double(opts().atomic_mass[species]); },
            [&](size_t const species) { return double(opts().atomic_number[species]); },
            [&](size_t const species) { return double(opts().atomic_number_squared[species]); });
    }
    double const gamma = grid::get_fgamma();
    double const cv = fullyIonized ? fully_ionized_cv(fld_unit_conversions().kbOverMh,
        gamma, composition.particleFraction) : double(opts().fld_cv);
    return recover_material_state(field(egas_i, index), kinetic, cold, field(tau_i, index),
        gamma, opts().dual_energy_sw1, density, cv,
        composition.electronFraction, composition.ionChargeSquared);
}

template <class Real>
ThermalState fld_material_state(
    std::vector<std::vector<Real>> const& state, size_t const index) {
    return fld_material_state_from_fields(
        [&](size_t const field, size_t const cell) { return double(state[field][cell]); }, index);
}

} // namespace octotiger::fld
