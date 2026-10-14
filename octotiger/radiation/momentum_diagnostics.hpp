// Copyright (c) 2026 AUTHORS. Boost Software License, Version 1.0.
#pragma once

#include "octotiger/radiation/conservation.hpp"
#include <cmath>
#include <stdexcept>

namespace radiationMomentum {

struct Parameters {
    double c = 1, mass_unit = 1, length_unit = 1, time_unit = 1;
    bool fld = false, periodic = false, fixed_material = false, normalized = false;
};

inline void writeCsvHeader(std::ostream& out) {
    out << "time,physical_c,volume,gas_energy,radiation_energy,gas_px,gas_py,gas_pz,"
        "radiation_px,radiation_py,radiation_pz,gas_px_boundary,gas_py_boundary,gas_pz_boundary,"
        "radiation_boundary_available,radiation_px_boundary,radiation_py_boundary,radiation_pz_boundary,"
        "external_impulse_available,external_px,external_py,external_pz,gas_energy_boundary,radiation_energy_boundary,units\n";
}

inline void writeCsvRow(std::ostream& out, double time,
    const radiationConservation::Totals& sample, const Parameters& parameters) {
    FpeGuard guard{};
    for (const double value : {parameters.c, parameters.mass_unit, parameters.length_unit, parameters.time_unit}) {
        if (!(value > 0) || !std::isfinite(value)) {
            throw std::invalid_argument("Invalid momentum diagnostic unit conversion");
        }
    }
    const long double length = parameters.normalized ? 1 : parameters.length_unit;
    const long double duration = parameters.normalized ? 1 : parameters.time_unit;
    const long double mass = parameters.normalized ? 1 : parameters.mass_unit;
    const long double momentum_unit = mass * length / duration;
    const long double energy_unit = momentum_unit * length / duration;
    const long double c = parameters.c;
    const long double radiation_momentum_unit = momentum_unit / c / c;
    out << std::scientific << std::setprecision(std::numeric_limits<long double>::max_digits10)
        << time * duration << ',' << c * length / duration << ',' << sample.volume * length * length * length
        << ',' << sample.gas[0] * energy_unit << ',' << sample.value[0] * energy_unit;
    for (int d = 1; d < 4; ++d) {
        out << ',' << sample.gas[d] * momentum_unit;
    }
    for (int d = 1; d < 4; ++d) {
        out << ',' << sample.value[d] * radiation_momentum_unit;
    }
    for (int d = 1; d < 4; ++d) {
        out << ',' << sample.gas_boundary[d] * momentum_unit;
    }
    // FLD has no evolved radiation momentum equation. Its old zero-valued
    // flux-moment budgets are NOT measurements of boundary pressure traction.
    const bool radiation_boundary_available = !parameters.fld || parameters.periodic;
    out << ',' << radiation_boundary_available;
    for (int d = 1; d < 4; ++d) {
        out << ',';
        if (radiation_boundary_available) {
            out << (parameters.fld ? 0 : sample.boundary[d] * radiation_momentum_unit);
        }
    }
    // Fixed media explicitly omit material recoil. Their M1 source increment
    // is therefore an external impulse on the recorded gas+radiation system.
    // Coupled material recoil is internal and must not be subtracted twice.
    const bool external_available = !parameters.fixed_material || !parameters.fld;
    out << ',' << external_available;
    for (int d = 1; d < 4; ++d) {
        out << ',';
        if (external_available) {
            out << (parameters.fixed_material ? sample.source[d] * radiation_momentum_unit : 0);
        }
    }
    out << ',' << sample.gas_boundary[0] * energy_unit
        << ',' << sample.boundary[0] * energy_unit
        << ',' << (parameters.normalized ? "normalized" : "cgs") << '\n';
}

} // namespace radiationMomentum
