// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "octotiger/light_curve.hpp"

namespace {

using octotiger::light_curve::Column;
using octotiger::light_curve::advance;
using octotiger::light_curve::intersect;

void require(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, std::string const& message) {
    require(std::isfinite(actual) &&
            std::abs(actual - expected) <= 2.0e-15 * std::max(1.0, std::abs(expected)),
        message + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}

struct Cell
{
    double extinction;
    double temperature;
    double length;
};

double area(Column const& column) {
    return (column.x_upper - column.x_lower) * (column.y_upper - column.y_lower);
}

void rectangularFootprints() {
    Column source;
    source.x_lower = -1.0;
    source.x_upper = 1.0;
    source.y_lower = -2.0;
    source.y_upper = 2.0;
    source.optical_depth = 0.375;
    std::vector<Column> pieces;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            Column child;
            require(intersect(source, i - 1.0, double(i), 2.0 * j - 2.0, 2.0 * j, child),
                "each transverse child must intersect its parent beam");
            if (i == 0 && j == 0) {
                for (int ii = 0; ii < 2; ++ii) {
                    for (int jj = 0; jj < 2; ++jj) {
                        Column fine;
                        require(intersect(child, -1.0 + 0.5 * ii, -0.5 + 0.5 * ii,
                                    -2.0 + jj, -1.0 + jj, fine),
                            "fine transverse cells must partition the coarse beam");
                        pieces.push_back(fine);
                    }
                }
            } else {
                pieces.push_back(child);
            }
        }
    }
    require(pieces.size() == 7, "one refined quadrant must produce seven adaptive footprints");
    double total_area = 0.0;
    for (size_t i = 0; i < pieces.size(); ++i) {
        auto const& piece = pieces[i];
        total_area += area(piece);
        near(piece.optical_depth, source.optical_depth,
            "transverse refinement must preserve accumulated optical depth");
        require(!piece.hit, "splitting an unfinished beam must not create a hit");
        Column coarse;
        require(intersect(piece, source.x_lower, source.x_upper,
                    source.y_lower, source.y_upper, coarse),
            "fine beam must intersect a later coarse footprint");
        near(area(coarse), area(piece), "entering a coarser leaf must not expand a fine beam");
        for (size_t j = 0; j < i; ++j) {
            Column overlap;
            require(!intersect(piece, pieces[j].x_lower, pieces[j].x_upper,
                        pieces[j].y_lower, pieces[j].y_upper, overlap),
                "adaptive transverse footprints must not overlap at shared edges");
        }
    }
    near(total_area, area(source), "adaptive transverse partition must conserve visible area");
    Column untouched;
    untouched.optical_depth = 0.125;
    require(!intersect(source, 1.0, 2.0, -2.0, 2.0, untouched),
        "touching a footprint edge must not create a duplicate beam");
    near(untouched.optical_depth, 0.125, "intersection miss must leave output unchanged");
}

void partialOcclusion() {
    Column source;
    source.x_lower = source.y_lower = -1.0;
    source.x_upper = source.y_upper = 1.0;
    Column foreground;
    Column background;
    require(intersect(source, -1.0, 0.0, -1.0, 1.0, foreground) &&
            intersect(source, 0.0, 1.0, -1.0, 1.0, background),
        "foreground cells must partition the full footprint");
    advance(foreground, 2.0, 3.0, 0.5);
    advance(background, 0.5, 1.0, 0.5);
    require(foreground.hit && !background.hit,
        "only the opaque foreground half must complete its beam");
    Column coarse;
    require(intersect(background, -1.0, 1.0, -1.0, 1.0, coarse),
        "unfinished beam must reach the coarse background cell");
    near(coarse.optical_depth, 0.25,
        "entering a coarse background cell must preserve foreground attenuation");
    advance(coarse, 2.0, 5.0, 0.5);
    require(coarse.hit, "background must supply the uncovered half's photosphere");
    near(area(foreground) + area(coarse), area(source),
        "foreground and partially visible background must cover the image exactly once");
    near(area(coarse), 2.0,
        "a partially visible coarse cell must contribute only its uncovered area");
    near(area(foreground) * std::pow(foreground.temperature, 4) +
            area(coarse) * std::pow(coarse.temperature, 4),
        1412.0, "piecewise constant emission must use each visible footprint's own temperature");
}

void twoObservers() {
    // Ordered from -z to +z. Each observer crosses a different emitting cell
    // after passing through an optically thin outer cell.
    std::array<Cell, 5> const cells{{
        {1.0, 10.0, 0.25}, {3.0, 20.0, 0.5}, {0.2, 50.0, 1.0},
        {2.0, 80.0, 0.5}, {0.5, 100.0, 0.5}}};
    Column below;
    Column above;
    for (auto const& cell : cells) {
        advance(below, cell.extinction, cell.temperature, cell.length);
    }
    for (auto cell = cells.rbegin(); cell != cells.rend(); ++cell) {
        advance(above, cell->extinction, cell->temperature, cell->length);
    }
    require(below.hit && above.hit, "both opaque columns must have a photosphere");
    near(below.temperature, 20.0, "-z observer must retain its first crossing cell temperature");
    near(above.temperature, 80.0, "+z observer must retain its first crossing cell temperature");
    near(below.optical_depth, 1.0, "-z integration must stop at tau one");
    near(above.optical_depth, 1.0, "+z integration must stop at tau one");
}

void transparentColumns() {
    Column column;
    advance(column, 0.5, 3.0, 0.5);
    advance(column, 0.75, 7.0, 0.5);
    require(!column.hit, "an optically thin column must not invent a photosphere");
    near(column.optical_depth, 0.625, "thin column must retain its integrated optical depth");
    near(column.temperature, 0.0, "no-hit column must not retain an emitting temperature");
    advance(column, 0.0, 0.0, 1.0e100);
    advance(column, 1.0e100, 100.0, 0.0);
    near(column.optical_depth, 0.625, "vacuum and zero-length segments must add no optical depth");
    require(!column.hit, "vacuum must not create a photosphere");
}

void densityScaling() {
    double const scattering_opacity = 0.5;
    Column diffuse;
    Column dense;
    advance(diffuse, scattering_opacity * 1.0, 10.0, 0.5);
    advance(dense, scattering_opacity * 4.0, 10.0, 0.5);
    require(!diffuse.hit && dense.hit,
        "constant mass scattering opacity must cross sooner at larger density");
    near(diffuse.optical_depth, 0.25, "mass opacity times density gives extinction per length");
    advance(diffuse, 2.0, 20.0, 1.0);
    advance(dense, 2.0, 20.0, 1.0);
    near(diffuse.temperature, 20.0, "lower-density foreground reveals the deeper cell");
    near(dense.temperature, 10.0, "denser foreground hides the deeper cell");
}

void refinementInvariance() {
    for (int const subdivisions : std::array<int, 4>{{1, 2, 4, 8}}) {
        Column column;
        for (int segment = 0; segment < subdivisions; ++segment) {
            advance(column, 0.5, 3.0, 0.5 / subdivisions);
        }
        require(!column.hit, "refined thin foreground must remain thin");
        near(column.optical_depth, 0.25, "subdivision must preserve integrated optical depth");
        for (int segment = 0; segment < subdivisions; ++segment) {
            advance(column, 2.0, 7.0, 0.5 / subdivisions);
        }
        require(column.hit, "refined opaque layer must retain its crossing");
        near(column.temperature, 7.0, "piecewise constant subdivision must retain crossing temperature");
        near(column.optical_depth, 1.0, "refined column must stop at the same optical depth");
    }
    Column boundary;
    advance(boundary, 2.0, 4.0, 0.5);
    advance(boundary, 100.0, 9.0, 1.0);
    near(boundary.temperature, 4.0, "an exact cell-face crossing belongs to the first reached cell");
}

void finiteExtremes() {
    Column opaque;
    advance(opaque, std::numeric_limits<double>::max(), 8.0,
        std::numeric_limits<double>::max());
    require(opaque.hit, "finite opaque cells must cross without overflowing optical depth");
    near(opaque.optical_depth, 1.0, "opaque cell optical depth must remain finite");
    Column cold;
    advance(cold, 1.0, 0.0, 1.0);
    require(cold.hit && cold.temperature == 0.0, "zero-temperature cells remain valid crossings");
}

void invalidInputs() {
    std::array<double, 3> const invalid{{-1.0,
        std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}};
    for (int field = 0; field < 3; ++field) {
        for (double const value : invalid) {
            std::array<double, 3> arguments{{1.0, 2.0, 0.5}};
            arguments[field] = value;
            Column column;
            bool rejected = false;
            try {
                advance(column, arguments[0], arguments[1], arguments[2]);
            } catch (std::invalid_argument const&) {
                rejected = true;
            }
            require(rejected, "nonfinite or negative material/segment values must be rejected");
            require(!column.hit && column.optical_depth == 0.0 && column.temperature == 0.0,
                "invalid cell data must not change the column");
        }
    }
}

}    // namespace

int main() {
    try {
        rectangularFootprints();
        partialOcclusion();
        twoObservers();
        transparentColumns();
        densityScaling();
        refinementInvariance();
        finiteExtremes();
        invalidInputs();
        std::cout << "Light-curve numerical tests passed\n";
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
