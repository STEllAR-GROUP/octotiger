// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace octotiger::light_curve {

// One rectangular bundle of vertical sightlines with a common optical depth.
// Split its footprint whenever it encounters finer transverse cells. The
// caller supplies cells in observer-to-interior order, independently for the
// +z and -z observers. Temperature is meaningful only after hit becomes true;
// optical_depth is then capped at one.
struct Column
{
    double x_lower = 0.0;
    double x_upper = 0.0;
    double y_lower = 0.0;
    double y_upper = 0.0;
    double optical_depth = 0.0;
    double temperature = 0.0;
    bool hit = false;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & x_lower;
        archive & x_upper;
        archive & y_lower;
        archive & y_upper;
        archive & optical_depth;
        archive & temperature;
        archive & hit;
    }
};

// The actual hydro cell containing tau_LOS=1, together with the visible
// subrectangle of that cell. Several disjoint footprints can hit one cell.
struct Hit
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double width = 0.0;
    double temperature = 0.0;
    double x_lower = 0.0;
    double x_upper = 0.0;
    double y_lower = 0.0;
    double y_upper = 0.0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & x;
        archive & y;
        archive & z;
        archive & width;
        archive & temperature;
        archive & x_lower;
        archive & x_upper;
        archive & y_lower;
        archive & y_upper;
    }
};

struct Trace
{
    std::vector<Column> columns;
    std::vector<Hit> hits;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & columns;
        archive & hits;
    }
};

// Clip a beam against a cell or node footprint, retaining its accumulated
// optical depth. Shared edges have zero area and do not produce another beam.
// A miss leaves clipped unchanged, so callers can reuse scratch storage.
inline bool intersect(Column const& source, double x_lower, double x_upper,
    double y_lower, double y_upper, Column& clipped) {
    auto const valid = [](double lower, double upper) {
        return std::isfinite(lower) && std::isfinite(upper) && lower < upper;
    };
    if (!valid(source.x_lower, source.x_upper) || !valid(source.y_lower, source.y_upper) ||
        !valid(x_lower, x_upper) || !valid(y_lower, y_upper)) {
        throw std::invalid_argument("Light-curve footprints require finite positive extents");
    }
    double const lower_x = std::max(source.x_lower, x_lower);
    double const upper_x = std::min(source.x_upper, x_upper);
    double const lower_y = std::max(source.y_lower, y_lower);
    double const upper_y = std::min(source.y_upper, y_upper);
    if (!(lower_x < upper_x && lower_y < upper_y)) {
        return false;
    }
    clipped = source;
    clipped.x_lower = lower_x;
    clipped.x_upper = upper_x;
    clipped.y_lower = lower_y;
    clipped.y_upper = upper_y;
    return true;
}

// Integrate one piecewise constant cell segment. Extinction is per unit
// length, and temperature is the gas temperature in that cell. This helper
// locates tau_LOS=1; it makes no assumption about a radiation energy field.
inline void advance(Column& column, double extinction, double temperature, double path_length) {
    if (!(extinction >= 0.0) || !(temperature >= 0.0) || !(path_length >= 0.0) ||
        !std::isfinite(extinction) || !std::isfinite(temperature) || !std::isfinite(path_length)) {
        throw std::invalid_argument(
            "Light-curve cell extinction, temperature, and length must be finite and nonnegative");
    }
    if (!(column.optical_depth >= 0.0 && column.optical_depth <= 1.0) ||
        !std::isfinite(column.optical_depth)) {
        throw std::invalid_argument("Light-curve column optical depth must be in [0,1]");
    }
    if (column.hit || extinction == 0.0 || path_length == 0.0) {
        return;
    }
    double const remaining = 1.0 - column.optical_depth;
    // Compare lengths first so an arbitrarily opaque, finite cell cannot
    // overflow extinction * path_length before recording the crossing.
    if (path_length >= remaining / extinction) {
        column.optical_depth = 1.0;
    } else {
        column.optical_depth = std::fma(extinction, path_length, column.optical_depth);
    }
    if (column.optical_depth >= 1.0) {
        column.optical_depth = 1.0;
        column.temperature = temperature;
        column.hit = true;
    }
}

}    // namespace octotiger::light_curve
