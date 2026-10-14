// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/radiation/fld.hpp"
#include "octotiger/test_problems/radiation/profiles.hpp"

namespace radiation { namespace fld {

// The composite boundary unknown is a face trace. Average prescribed energy
// over the face, not a ghost-cell volume. Cartesian subfaces are squares.
template<class Function>
inline double boundary_face_average(const octotiger::elliptic::face& face, Function&& value) {
    constexpr double nodes[]{-.8611363115940526, -.3399810435848563,
        .3399810435848563, .8611363115940526};
    constexpr double weights[]{.3478548451374538, .6521451548625461,
        .6521451548625461, .3478548451374538};
    const double half_width = .5 * std::sqrt(face.area);
    const int a = (face.axis + 1) % 3, b = (face.axis + 2) % 3;
    double average = 0;
    for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) {
            auto position = face.center;
            position[a] += half_width * nodes[j];
            position[b] += half_width * nodes[k];
            average += .25 * weights[j] * weights[k] * value(position);
        }
    }
    return average;
}

inline std::vector<boundary_condition> sphere_boundary(const composite_mesh& mesh,
    const radiationTests::Parameters& parameters) {
    std::vector<boundary_condition> result(mesh.faces().size());
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        const auto& face = mesh.faces()[f];
        if (face.boundary_side != 0) {
            result[f] = {1, 0, boundary_face_average(face, [&](const auto& position) {
                return radiationTests::equilibriumSphere(position, parameters)[0];
            })};
        }
    }
    return result;
}

}} // namespace radiation::fld
