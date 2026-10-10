// Copyright (c) 2026 AUTHORS. Boost Software License, Version 1.0.
#include "octotiger/amr/gradient.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}
void near(long double actual, long double expected, const char* message) {
    require(std::abs(actual - expected) <= 2e-12L * std::max(1.L, std::abs(expected)), message);
}
template<class Function>
void rejects(Function function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}
}

int main() {
    using namespace gradientRefinement;
    try {
        Parameters parameters;
        parameters.validate();
        for (double dx : {.125, .5, 2.}) {
            const double center = 20;
            const std::array<double, 6> linear{center - 2*dx, center + 2*dx,
                center + 4*dx, center - 4*dx, center - dx, center + dx};
            near(edgeGradient(center, linear, dx, Mode::absolute, 1e-20), 4,
                "Dimensional absolute linear gradient must be resolution independent and max-axis");
            const std::array<double, 6> exponential{7*std::exp(-dx), 7*std::exp(dx),
                7*std::exp(3*dx), 7*std::exp(-3*dx), 7, 7};
            near(edgeGradient(7, exponential, dx, Mode::relative, 1e-20), 3,
                "Dimensional logarithmic gradient must be resolution independent");
            auto scaled = exponential;
            for (double& value : scaled) {
                value *= 1e15;
            }
            near(edgeGradient(7e15, scaled, dx, Mode::relative, 1e-20), 3,
                "Relative gradient must be invariant under field-unit scaling away from floor");
        }
        std::array<double, 6> neighbors{};
        neighbors.fill(1);
        near(edgeGradient(2, neighbors, .25, Mode::absolute, 1e-20), 4,
            "An isolated peak must not cancel as it would with a centered derivative");
        for (int direction = 0; direction < 6; ++direction) {
            neighbors.fill(1);
            neighbors[direction] = 5;
            near(edgeGradient(1, neighbors, 2, Mode::absolute, 1e-20), 2,
                "Each forward and backward edge must be inspected");
        }
        neighbors.fill(0);
        near(edgeGradient(0, neighbors, 1, Mode::relative, 1e-6), 0,
            "Vacuum must have zero logarithmic gradient");
        neighbors[0] = 2e-6;
        near(edgeGradient(0, neighbors, .5, Mode::relative, 1e-6), 2*std::log(2.L),
            "Relative gradient must use the configured positive floor");
        near(edgeGradient(-1, neighbors, .5, Mode::relative, 1e-6), 2*std::log(2.L),
            "Values below the floor must use the same logarithm");
        near(edgeGradient(0, neighbors, 1, Mode::absolute, 10), 2e-6,
            "The logarithmic floor must not alter absolute gradients");

        parameters.threshold = 10;
        parameters.coarsenFactor = .8;
        near(parameters.effectiveThreshold(false), 10, "Leaf uses the refine threshold");
        near(parameters.effectiveThreshold(true), 8, "Refined parent uses the lower coarsen threshold");
        require(9 < parameters.effectiveThreshold(false) && 9 >= parameters.effectiveThreshold(true),
            "Hysteresis must retain an existing refinement without creating a new one");
        parameters.coarsenFactor = 1;
        near(parameters.effectiveThreshold(true), parameters.effectiveThreshold(false),
            "A factor of one disables hysteresis");

        const double infinity = std::numeric_limits<double>::infinity();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (double invalid : {0., -1., infinity, nan}) {
            rejects([&] { auto p = parameters; p.threshold = invalid; p.validate(); }, "Reject invalid threshold");
            rejects([&] { auto p = parameters; p.floor = invalid; p.validate(); }, "Reject invalid floor");
            rejects([&] { auto p = parameters; p.coarsenFactor = invalid; p.validate(); }, "Reject invalid coarsen factor");
            rejects([&] { edgeGradient(1, neighbors, invalid, Mode::absolute, 1e-6); }, "Reject invalid spacing");
        }
        rejects([&] { auto p = parameters; p.coarsenFactor = 1.01; p.validate(); }, "Reject coarsen factor above one");
        for (int invalid : {-1, 3}) {
            rejects([&] { auto p = parameters; p.buffer = invalid; p.validate(); }, "Reject unavailable halo width");
        }
        for (double invalid : {infinity, -infinity, nan}) {
            rejects([&] { edgeGradient(invalid, neighbors, 1, Mode::relative, 1e-6); }, "Reject nonfinite center");
            rejects([&] { auto bad = neighbors; bad[5] = invalid; edgeGradient(1, bad, 1, Mode::absolute, 1e-6); },
                "Reject nonfinite neighbor");
        }
        rejects([&] { edgeGradient(1, neighbors, 1, static_cast<Mode>(99), 1e-6); }, "Reject invalid mode");
        std::cout << "PASS: " << checks << " gradient refinement numerical checks\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
