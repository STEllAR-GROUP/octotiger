// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "octotiger/fld_boundary.hpp"

namespace {

using octotiger::fld::vacuumConductance;
using octotiger::fld::vacuumFluxFactor;

void require(bool const condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void relative(double const actual, double const expected, double const tolerance,
    std::string const& message) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <= tolerance * std::abs(expected), message);
}

template <class Function>
void rejects(Function function, std::string const& message) {
    bool rejected = false;
    try {
        function();
    } catch (std::exception const&) {
        rejected = true;
    }
    require(rejected, message);
}

// Independent angular quadrature of <|mu|> for the normalized LP intensity
// psi(mu)=tanh(R)/(R*(1-mu*tanh(R))). The production beta uses a closed form.
long double angularFactor(long double const r) {
    constexpr int intervals = 16384;
    long double const tangent = std::tanh(r);
    long double sum = 0.0L;
    for (int index = 0; index <= intervals; ++index) {
        long double const mu = static_cast<long double>(index) / intervals;
        long double const value = 0.5L * mu * tangent / r *
            (1.0L / (1.0L - mu * tangent) + 1.0L / (1.0L + mu * tangent));
        int const weight = index == 0 || index == intervals ? 1 : index % 2 == 0 ? 2 : 4;
        sum += weight * value;
    }
    return sum / (3 * intervals);
}

void angular_limits() {
    relative(vacuumFluxFactor(0.0), 0.5, 0.0, "isotropic hemisphere has beta=1/2");
    relative(vacuumFluxFactor(1.0e-300), 0.5, 0.0, "tiny R has no cancellation");
    relative(vacuumFluxFactor(std::numeric_limits<double>::infinity()), 1.0, 0.0,
        "infinite R has the streaming limit");
    relative(vacuumFluxFactor(1000.0), 1.0 - std::log(2.0) / 1000.0, 2.0e-16,
        "large R retains its asymptotic correction without cosh overflow");
    for (double const r : {0.001, 0.03, 0.5, 1.0, 2.0}) {
        relative(vacuumFluxFactor(r), double(angularFactor(r)), 2.0e-12,
            "modified Marshak factor agrees with independent angular integration");
    }
    double previous = 0.5;
    for (int exponent = -300; exponent <= 300; ++exponent) {
        double const beta = vacuumFluxFactor(std::pow(10.0, exponent));
        require(std::isfinite(beta) && beta >= previous && beta <= 1.0,
            "vacuum factor is finite, bounded and monotone over extreme R");
        previous = beta;
    }
    for (double const transition : {0.001, 20.0}) {
        relative(vacuumFluxFactor(transition * (1.0 - 1.0e-10)),
            vacuumFluxFactor(transition * (1.0 + 1.0e-10)), 1.0e-10,
            "stable formula branches are continuous");
    }
}

void robin_relation() {
    relative(vacuumConductance(10.0, {{0.0, 0.0, 0.0}}, 8.0, 12.0, 0.25),
        1.5, 0.0, "diffusion limit includes the half-cell optical resistance");
    constexpr long double energy = 7.0L;
    constexpr long double chi = 2.0L;
    constexpr long double speed = 11.0L;
    constexpr long double distance = 0.37L;
    constexpr long double r = 5.0L / (chi * energy);
    long double const beta = angularFactor(r);
    long double const lambda = (2.0L + r) / (6.0L + 3.0L * r + r * r);
    long double const diffusion = speed * lambda / chi;
    long double const surfaceEnergy = diffusion * energy / (diffusion + beta * speed * distance);
    double const h = vacuumConductance(double(energy), {{3.0, -4.0, 0.0}},
        double(chi), double(speed), double(distance));
    relative(h * double(energy), double(beta * speed * surfaceEnergy), 2.0e-13,
        "boundary conductance satisfies the independent Robin surface relation");
    relative(h * double(energy), double(diffusion * (energy - surfaceEnergy) / distance), 2.0e-13,
        "boundary flux equals the center-to-face diffusive flux");
    relative(vacuumConductance(double(energy), {{0.0, 3.0, 4.0}},
                 double(chi), double(speed), double(distance)), h, 0.0,
        "interior closure factor is invariant under coordinate rotation");
    for (double const scale : {1.0e-200, 1.0e200}) {
        relative(vacuumConductance(double(energy) * scale, {{3.0 * scale, -4.0 * scale, 0.0}},
                     double(chi), double(speed), double(distance)), h, 5.0e-15,
            "conductance does not depend on radiation amplitude");
    }
}

void degenerate_and_extreme_states() {
    relative(vacuumConductance(4.0, {{0.0, 0.0, 0.0}}, 0.0, 6.0, 0.5),
        3.0, 0.0, "uniform exact vacuum has the isotropic limit");
    relative(vacuumConductance(4.0, {{3.0, -4.0, 0.0}}, 0.0, 6.0, 0.5),
        6.0 / (1.0 + 0.5 * 5.0 / 4.0), 3.0e-16,
        "gradient-bearing exact vacuum uses D=cE/gradient and beta=1");
    for (double const chi : {0.0, 2.0}) {
        require(vacuumConductance(0.0, {{1.0, 0.0, 0.0}}, chi, 6.0, 0.5) == 0.0,
            "zero energy with a nonzero gradient gives zero conductance");
        double const h = vacuumConductance(0.0, {{0.0, 0.0, 0.0}}, chi, 6.0, 0.5);
        require(std::isfinite(h) && h >= 0.0 && h * 0.0 == 0.0,
            "uniform zero energy has finite conductance and exactly zero escape");
    }
    relative(vacuumConductance(1.0, {{0.0, 0.0, 0.0}}, 1.0e300, 1.0e300, 1.0e-300),
        2.0e299, 5.0e-15, "large extinction and small cell width retain finite optical depth");
    relative(vacuumConductance(1.0e-300, {{1.0e300, 0.0, 0.0}},
                 1.0e-300, 1.0e300, 1.0e-300), 1.0, 5.0e-15,
        "unrepresentably large double R does not destroy a representable conductance");
    relative(vacuumConductance(1.0e300, {{1.0e-300, 0.0, 0.0}},
                 1.0e300, 5.0, 1.0e-300), 1.0, 5.0e-15,
        "unrepresentably small double R retains the diffusion limit");
    for (double const chi : {0.0, 1.0e-200, 1.0, 1.0e200}) {
        for (double const energy : {0.0, 1.0e-200, 1.0, 1.0e200}) {
            for (double const gradient : {0.0, 1.0e-200, 1.0, 1.0e200}) {
                double const h = vacuumConductance(energy, {{gradient, -gradient, gradient}},
                    chi, 3.0, 0.25);
                require(std::isfinite(h) && h >= 0.0 && h <= 3.0,
                    "outgoing conductance is positive and bounded by c for extreme states");
            }
        }
    }
}

void invalid_inputs() {
    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const infinity = std::numeric_limits<double>::infinity();
    for (double const bad : {-1.0, nan, -infinity}) {
        rejects([&] { vacuumFluxFactor(bad); }, "invalid gradient parameter is rejected");
    }
    for (double const bad : {-1.0, nan, infinity}) {
        rejects([&] { vacuumConductance(bad, {{0.0, 0.0, 0.0}}, 1.0, 1.0, 1.0); },
            "invalid energy is rejected");
        rejects([&] { vacuumConductance(1.0, {{0.0, 0.0, 0.0}}, bad, 1.0, 1.0); },
            "invalid extinction is rejected");
    }
    for (double const bad : {0.0, -1.0, nan, infinity}) {
        rejects([&] { vacuumConductance(1.0, {{0.0, 0.0, 0.0}}, 1.0, bad, 1.0); },
            "invalid transport speed is rejected");
        rejects([&] { vacuumConductance(1.0, {{0.0, 0.0, 0.0}}, 1.0, 1.0, bad); },
            "invalid half-cell distance is rejected");
    }
    for (double const bad : {nan, infinity}) {
        rejects([&] { vacuumConductance(1.0, {{0.0, bad, 0.0}}, 1.0, 1.0, 1.0); },
            "nonfinite gradient is rejected");
    }
}

} // namespace

int main() {
    try {
        angular_limits();
        robin_relation();
        degenerate_and_extreme_states();
        invalid_inputs();
        std::cout << "FLD vacuum boundary tests passed\n";
    } catch (std::exception const& error) {
        std::cerr << "FLD vacuum boundary test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
