// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "octotiger/fld_closure.hpp"
#include "octotiger/fld_configuration.hpp"
#include "octotiger/fld_hydro.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/light_curve.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/roe.hpp"

namespace {

struct Material
{
    double temperature = 0.0;
    double extinction = 0.0;
    bool ready = false;
};

double material_temperature(
    std::vector<std::vector<safe_real>> const& state, integer const cell, double const density) {
    if (opts().eos == IPR) {
        // The IPR primitive update stores temperature in tau, not e^(1/gamma).
        return state[tau_i][cell];
    }

    if (opts().fld) {
        return octotiger::fld::fld_material_state(state, cell).temperature;
    }

    double const total = state[egas_i][cell];
    double const tau = state[tau_i][cell];
    if (!(total >= 0.0) || !std::isfinite(total) || !(tau >= 0.0) || !std::isfinite(tau)) {
        throw std::runtime_error("Light curve requires finite nonnegative gas energy and tau");
    }
    long double nonthermal = 0.0L;
    for (int dim = 0; dim < NDIM; ++dim) {
        double const momentum = state[sx_i + dim][cell];
        if (!std::isfinite(momentum)) {
            throw std::runtime_error("Light curve requires finite gas momentum");
        }
        nonthermal += 0.5L * momentum * momentum / density;
    }
    if (opts().eos == WD) {
        double const cold_energy = ztwd_energy(density);
        if (!(cold_energy >= 0.0) || !std::isfinite(cold_energy)) {
            throw std::runtime_error("Light curve requires finite nonnegative cold energy");
        }
        nonthermal += cold_energy;
    }
    if (!std::isfinite(nonthermal) || nonthermal > std::numeric_limits<double>::max()) {
        throw std::runtime_error("Light curve kinetic and cold energy exceeds the numeric range");
    }
    double const energy = octotiger::fld::recover_internal_energy(
        total, double(nonthermal), tau, grid::get_fgamma(), opts().dual_energy_sw1);
    if (!(energy >= 0.0) || !std::isfinite(energy)) {
        throw std::runtime_error("Light curve requires finite nonnegative thermal energy");
    }

    specie_state_t<real> species;
    double species_density = 0.0;
    for (integer species_index = 0; species_index < opts().n_species; ++species_index) {
        double const value = state[spc_i + species_index][cell];
        if (!(value >= 0.0) || !std::isfinite(value)) {
            throw std::runtime_error("Light curve requires finite nonnegative species densities");
        }
        species[species_index] = value;
        species_density += value;
    }
    if (!(species_density > 0.0) || !std::isfinite(species_density)) {
        throw std::runtime_error("Light curve requires positive total species density");
    }
    real mu = 0.0;
    real hydrogen = 0.0;
    real metals = 0.0;
    mean_ion_weight(species, mu, hydrogen, metals);
    if (!(mu > 0.0) || !std::isfinite(double(mu))) {
        throw std::runtime_error("Light curve requires a finite positive mean molecular weight");
    }
    return (grid::get_fgamma() - 1.0) * mu * physcon().mh / physcon().kb * energy / density;
}

double material_extinction(double const density, double const temperature) {
    double absorption = 0.0;
    if (opts().light_curve_absorption > 0.0) {
        if (temperature == 0.0 && opts().light_curve_temperature_exponent < 0.0) {
            throw std::runtime_error("Light curve absorption is singular at zero temperature");
        }
        if (temperature > 0.0 || opts().light_curve_temperature_exponent == 0.0) {
            absorption = octotiger::fld::absorption(density, temperature > 0.0 ? temperature : 1.0,
                opts().light_curve_absorption, opts().light_curve_density_exponent,
                opts().light_curve_temperature_exponent);
        }
    }
    double const extinction =
        absorption + octotiger::fld::scattering(density, opts().light_curve_scattering);
    if (!(extinction >= 0.0) || !std::isfinite(extinction)) {
        throw std::runtime_error("Light curve requires finite nonnegative extinction");
    }
    return extinction;
}

}    // namespace

octotiger::light_curve::Trace grid::trace_light_curve_columns(
    std::vector<octotiger::light_curve::Column> columns, bool const from_above) const {
    if (!(dx > 0.0) || !std::isfinite(double(dx))) {
        throw std::runtime_error("Light curve requires a finite positive cell width");
    }
    std::vector<Material> material(INX * INX * INX);
    octotiger::light_curve::Trace result;
    double const xmax = xmin[XDIM] + INX * dx;
    double const ymax = xmin[YDIM] + INX * dx;
    for (auto const& incoming : columns) {
        if (incoming.hit) {
            continue;
        }
        octotiger::light_curve::Column clipped;
        if (!octotiger::light_curve::intersect(
                incoming, xmin[XDIM], xmax, xmin[YDIM], ymax, clipped)) {
            continue;
        }
        integer const i_begin = std::clamp(
            integer(std::floor((clipped.x_lower - xmin[XDIM]) / dx)), integer(0), integer(INX - 1));
        integer const i_end = std::clamp(
            integer(std::ceil((clipped.x_upper - xmin[XDIM]) / dx)), integer(1), integer(INX));
        integer const j_begin = std::clamp(
            integer(std::floor((clipped.y_lower - xmin[YDIM]) / dx)), integer(0), integer(INX - 1));
        integer const j_end = std::clamp(
            integer(std::ceil((clipped.y_upper - xmin[YDIM]) / dx)), integer(1), integer(INX));
        for (integer i = i_begin; i < i_end; ++i) {
            for (integer j = j_begin; j < j_end; ++j) {
                octotiger::light_curve::Column column;
                if (!octotiger::light_curve::intersect(clipped, xmin[XDIM] + i * dx,
                        xmin[XDIM] + (i + 1) * dx, xmin[YDIM] + j * dx, xmin[YDIM] + (j + 1) * dx,
                        column)) {
                    continue;
                }
                for (integer offset = 0; offset < INX && !column.hit; ++offset) {
                    integer const k = from_above ? INX - 1 - offset : offset;
                    auto& cell_material = material[h0index(i, j, k)];
                    if (!cell_material.ready) {
                        integer const cell = hindex(i + H_BW, j + H_BW, k + H_BW);
                        double const density = U[rho_i][cell];
                        if (!(density >= 0.0) || !std::isfinite(density)) {
                            throw std::runtime_error(
                                "Light curve requires finite nonnegative density");
                        }
                        if (density > 0.0) {
                            cell_material.temperature = material_temperature(U, cell, density);
                            if (!(cell_material.temperature >= 0.0) ||
                                !std::isfinite(cell_material.temperature)) {
                                throw std::runtime_error(
                                    "Light curve requires finite nonnegative temperature");
                            }
                            if (opts().light_curve_opacity_model == "fld") {
                                auto const state = octotiger::fld::fld_material_state(U, cell);
                                cell_material.extinction = octotiger::fld::evaluate_opacity(
                                    octotiger::fld::fld_opacity_parameters(), density, state.temperature,
                                    state.electronFraction, state.ionChargeSquared).rosselandExtinction;
                            } else {
                                cell_material.extinction = material_extinction(density, cell_material.temperature);
                            }
                        }
                        cell_material.ready = true;
                    }
                    octotiger::light_curve::advance(
                        column, cell_material.extinction, cell_material.temperature, dx);
                    if (column.hit) {
                        octotiger::light_curve::Hit hit;
                        hit.x = xmin[XDIM] + (i + 0.5) * dx;
                        hit.y = xmin[YDIM] + (j + 0.5) * dx;
                        hit.z = xmin[ZDIM] + (k + 0.5) * dx;
                        hit.width = dx;
                        hit.temperature = column.temperature;
                        hit.x_lower = column.x_lower;
                        hit.x_upper = column.x_upper;
                        hit.y_lower = column.y_lower;
                        hit.y_upper = column.y_upper;
                        result.hits.push_back(hit);
                    }
                }
                if (!column.hit) {
                    result.columns.push_back(column);
                }
            }
        }
    }
    return result;
}
