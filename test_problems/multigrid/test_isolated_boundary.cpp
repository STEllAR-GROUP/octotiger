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

#include "octotiger/multigrid/isolated_boundary.hpp"

namespace {

using octotiger::multigrid::IsolatedBoundaryMoments;
using octotiger::multigrid::SourceMoments;
using Position = std::array<double, 3>;

double const fourPi = 4.0 * std::acos(-1.0);

struct Particle {
    Position position;
    double mass;
};

void require(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance, std::string const& message) {
    require(std::isfinite(actual)
        && std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)),
        message + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}

template<class Function>
void rejects(Function function, std::string const& message) {
    bool rejected = false;
    try {
        function();
    } catch (std::invalid_argument const&) {
        rejected = true;
    }
    require(rejected, message);
}

Position difference(Position const& a, Position const& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

double norm(Position const& x) {
    return std::hypot(x[0], x[1], x[2]);
}

Position centerOf(std::vector<Particle> const& sources) {
    SourceMoments moments;
    for (auto const& source : sources) {
        moments.addSource(source.position, fourPi * source.mass, 1.0);
    }
    return moments.centerOfMass();
}

IsolatedBoundaryMoments accumulate(std::vector<Particle> const& sources, Position const& center) {
    IsolatedBoundaryMoments moments;
    for (auto const& source : sources) {
        moments.addSource(source.position, fourPi * source.mass, 1.0, center);
    }
    return moments;
}

// Independent reference: scalar Legendre polynomials and the addition theorem.
// It uses neither associated Legendre functions nor real/imaginary moments.
double legendreReference(std::vector<Particle> const& sources, Position const& field,
    Position const& center) {
    Position const fieldOffset = difference(field, center);
    double const radius = norm(fieldOffset);
    double result = 0.0;
    for (auto const& source : sources) {
        Position const sourceOffset = difference(source.position, center);
        double const sourceRadius = norm(sourceOffset);
        if (sourceRadius == 0.0) {
            result -= source.mass / radius;
            continue;
        }
        double const cosine = (sourceOffset[0] * fieldOffset[0]
            + sourceOffset[1] * fieldOffset[1] + sourceOffset[2] * fieldOffset[2])
            / (sourceRadius * radius);
        double const ratio = sourceRadius / radius;
        double previous = 1.0;
        double current = cosine;
        double power = ratio;
        double sum = 1.0 + power * current;
        for (int l = 2; l <= 4; ++l) {
            double const next = ((2 * l - 1) * cosine * current - (l - 1) * previous) / l;
            power *= ratio;
            sum += power * next;
            previous = current;
            current = next;
        }
        result -= source.mass * sum / radius;
    }
    return result;
}

double directPotential(std::vector<Particle> const& sources, Position const& field) {
    double result = 0.0;
    for (auto const& source : sources) {
        result -= source.mass / norm(difference(field, source.position));
    }
    return result;
}

void testMonopoleAndUnits() {
    Position const center{1.25, -2.0, 0.75};
    double const gravity = 2.5;
    double const density = 3.0;
    double const volume = 0.125;
    double const source = fourPi * gravity * density;
    double const mass = gravity * density * volume;
    SourceMoments com;
    com.addSource(center, source, volume);
    near(com.mass, mass, 1.0e-15, "source units include G and divide by 4*pi");
    for (int axis = 0; axis != 3; ++axis) {
        near(com.centerOfMass()[axis], center[axis], 1.0e-15, "point source COM");
    }
    IsolatedBoundaryMoments moments;
    moments.addSource(center, source, volume, center);
    Position const field{center[0] + 3.0, center[1] + 4.0, center[2]};
    near(moments.potential(field, center), -mass / 5.0, 1.0e-15, "centered point source");
    for (int p = 1; p != 25; ++p) {
        near(moments.real[p], 0.0, 0.0, "centered source has no higher real moments");
        near(moments.imaginary[p], 0.0, 0.0, "centered source has no higher imaginary moments");
    }
    rejects([&] { moments.potential(center, center); }, "nonempty field at COM is invalid");
}

void testEmptyAndInvalidInputs() {
    Position const origin{};
    SourceMoments com;
    require(com.centerOfMass() == origin, "empty COM is origin");
    IsolatedBoundaryMoments moments;
    moments.addSource(origin, 0.0, 1.0, origin);
    near(moments.potential(origin, origin), 0.0, 0.0, "empty potential at origin");
    Position const invalid{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
    rejects([&] { moments.potential(invalid, origin); }, "invalid empty field rejected");
    rejects([&] { moments.potential(origin, invalid); }, "invalid center rejected");
    rejects([&] { com.addSource(origin, -1.0, 1.0); }, "negative density rejected");
    rejects([&] { com.addSource(origin, 1.0, 0.0); }, "zero volume rejected");
    rejects([&] { moments.addSource(invalid, 1.0, 1.0, origin); }, "invalid source position rejected");
}

void testRotationAndHistoricalNormalization() {
    double const a = 0.5;
    double const radius = 2.0;
    double const q = a / radius;
    double const expected = -2.0 / radius * (1.0 + q * q + q * q * q * q);
    Position const origin{};
    for (int axis = 0; axis != 3; ++axis) {
        Position positive{};
        Position negative{};
        Position field{};
        positive[axis] = a;
        negative[axis] = -a;
        field[axis] = radius;
        std::vector<Particle> const sources{{positive, 1.0}, {negative, 1.0}};
        auto const moments = accumulate(sources, origin);
        near(moments.potential(field, origin), expected, 2.0e-15,
            "axis-rotated pair matches degree-four expansion");
        near(moments.potential(field, origin), legendreReference(sources, field, origin),
            2.0e-15, "axis rotation agrees with addition theorem");
    }
    // Historical q(2,-2)=2*q(2,2) weights its source-field product fourfold.
    // At equal equatorial directions the l=2 angular sum is 1/4+5*(3/8),
    // whereas the addition theorem requires P_2(1)=1 on every axis.
    double const historicalQuadrupole = 0.25 + 5.0 * 0.375;
    require(std::abs(historicalQuadrupole - 1.0) > 1.0,
        "historical negative-even-m factor violates the addition theorem");
}

void testAdditionTranslationAndReduction() {
    std::vector<Particle> sources{{{0.75, -0.25, 0.5}, 0.6},
        {{-0.5, 0.625, -0.25}, 1.4}, {{0.125, 0.25, 0.875}, 0.8},
        {{-0.375, -0.5, -0.625}, 0.4}};
    Position const center = centerOf(sources);
    auto const moments = accumulate(sources, center);
    std::array<Position, 4> const fields{{{3.0, -4.0, 5.0}, {-5.0, 2.0, -3.0},
        {1.5, 4.5, -2.5}, {-2.0, -3.0, 4.0}}};
    for (auto const& field : fields) {
        near(moments.potential(field, center), legendreReference(sources, field, center),
            3.0e-15, "arbitrary 3D addition theorem");
    }
    SourceMoments leftCom;
    SourceMoments rightCom;
    IsolatedBoundaryMoments left;
    IsolatedBoundaryMoments right;
    for (int i = 0; i != 4; ++i) {
        auto const& source = sources[i];
        auto& com = i % 2 == 0 ? leftCom : rightCom;
        auto& part = i % 2 == 0 ? left : right;
        com.addSource(source.position, fourPi * source.mass, 1.0);
        part.addSource(source.position, fourPi * source.mass, 1.0, center);
    }
    leftCom += rightCom;
    left += right;
    for (int axis = 0; axis != 3; ++axis) {
        near(leftCom.centerOfMass()[axis], center[axis], 2.0e-15, "partitioned COM reduction");
    }
    near(left.potential(fields[0], center), moments.potential(fields[0], center),
        2.0e-15, "partitioned multipole reduction");
    Position const shift{11.0, -7.5, 4.0};
    for (auto& source : sources) {
        for (int axis = 0; axis != 3; ++axis) {
            source.position[axis] += shift[axis];
        }
    }
    Position const translatedCenter = centerOf(sources);
    auto const translated = accumulate(sources, translatedCenter);
    Position translatedField = fields[0];
    for (int axis = 0; axis != 3; ++axis) {
        near(translatedCenter[axis], center[axis] + shift[axis], 2.0e-15, "translated COM");
        translatedField[axis] += shift[axis];
    }
    near(translated.potential(translatedField, translatedCenter),
        moments.potential(fields[0], center), 3.0e-15, "translation invariance");
}

void testFarFieldTruncation() {
    Position const origin{};
    std::vector<Particle> const sources{{{-0.5, 0.0, 0.0}, 1.0}, {{0.5, 0.0, 0.0}, 1.0}};
    auto const moments = accumulate(sources, origin);
    double const errorNear = std::abs(moments.potential({2.0, 0.0, 0.0}, origin)
        - directPotential(sources, {2.0, 0.0, 0.0}));
    double const errorFar = std::abs(moments.potential({4.0, 0.0, 0.0}, origin)
        - directPotential(sources, {4.0, 0.0, 0.0}));
    require(errorNear > 0.0 && errorFar > 0.0, "degree-four expansion has nonzero truncation error");
    // Symmetry removes odd l, so the first omitted term is l=6, proportional
    // to r^-7. The exact geometric tail predicts this ratio, close to 2^7.
    double const expectedRatio = 128.0 * (1.0 - 0.125 * 0.125) / (1.0 - 0.25 * 0.25);
    near(errorNear / errorFar, expectedRatio, 1.0e-9, "far-field degree-six error decay");
}

} // namespace

int main() {
    try {
        testMonopoleAndUnits();
        testEmptyAndInvalidInputs();
        testRotationAndHistoricalNormalization();
        testAdditionTranslationAndReduction();
        testFarFieldTruncation();
        std::cout << "Isolated boundary tests passed\n";
    } catch (std::exception const& error) {
        std::cerr << "Isolated boundary test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
