// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <stdexcept>

#include "octotiger/fld_material_state.hpp"
#include "octotiger/fld_opacity.hpp"
#include "octotiger/options.hpp"

namespace octotiger::fld {

inline OpacityParameters fld_opacity_parameters() {
    OpacityParameters p;
    auto const &o = opts();
    if (o.fld_opacity_model == "legacy") {
        p.model = OpacityModel::legacy;
    } else if (o.fld_opacity_model == "independent") {
        p.model = OpacityModel::independent;
    } else if (o.fld_opacity_model == "free_free_thomson") {
        p.model = OpacityModel::free_free_thomson;
    } else {
        throw std::invalid_argument("Unknown FLD opacity provider");
    }
    p.absorption = o.fld_absorption;
    p.scattering = o.fld_scattering;
    p.densityExponent = o.fld_density_exponent;
    p.temperatureExponent = o.fld_temperature_exponent;
    p.rosseland = o.fld_rosseland;
    p.rosselandDensityExponent = o.fld_rosseland_density_exponent;
    p.rosselandTemperatureExponent = o.fld_rosseland_temperature_exponent;
    p.temperatureFloor = o.fld_opacity_temperature_floor;
    p.gaunt = o.fld_gaunt;
    if (p.model == OpacityModel::free_free_thomson) {
        auto const units = fld_unit_conversions();
        p.lengthUnit = units.lengthUnit;
        p.densityUnit = units.densityUnit;
    }
    return p;
}

} // namespace octotiger::fld
