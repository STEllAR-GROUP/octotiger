// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "octotiger/interpolated_cache.hpp"

namespace {

void require(bool const condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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

void reusable_interpolation() {
    using octotiger::InterpolatedCache;
    auto const polynomial = [](long double const coordinate) {
        return coordinate * coordinate * coordinate - 2.0L * coordinate - 3.0L;
    };
    InterpolatedCache signedValues(polynomial, 0.125L, 7);
    require(signedValues.statistics().samples == 0, "generic interpolation starts unsampled");
    signedValues.evaluate(0.025L);
    require(signedValues.statistics().intervals == 7,
        "generic interpolation honors the configured extension block size");
    for (int query = 0; query <= 200; ++query) {
        long double const coordinate = -3.0L + (query + 0.271828L) * 0.03L;
        long double const exact = polynomial(coordinate);
        require(std::abs(signedValues.evaluate(coordinate) - exact) <=
                2.0e-16L * std::max(1.0L, std::abs(exact)),
            "signed nonmonotone cubics are reproduced without endpoint clamping");
    }
    auto const warm = signedValues.statistics();
    for (long double const boundary : {warm.lower, warm.upper}) {
        for (long double const offset : {-1.0e-12L, 0.0L, 1.0e-12L}) {
            long double const coordinate = boundary + offset;
            require(std::abs(signedValues.evaluate(coordinate) - polynomial(coordinate)) <=
                    2.0e-16L * std::max(1.0L, std::abs(polynomial(coordinate))),
                "general cubic interpolation extends safely across either endpoint");
        }
    }
    require(signedValues.statistics().sampleEvaluations == signedValues.statistics().samples,
        "serial generic extension reuses every existing sample");

    // Unrelated positive function f(x) = 0.01*(1+x)^2, interpolated in log-log
    // coordinates. Its logarithm is negative over much of the sampled domain.
    auto const logFunction = [](long double const logX) {
        return std::log(0.01L) + 2.0L * std::log1p(std::exp(logX));
    };
    InterpolatedCache logLog(logFunction, 1.0L / 128.0L, 32);
    double maximumError = 0.0;
    for (int query = 0; query <= 800; ++query) {
        long double const logX = -4.0L + (query + 0.314159L) * 0.01L;
        long double const x = std::exp(logX);
        long double const exact = 0.01L * (1.0L + x) * (1.0L + x);
        double const error = double(std::abs(std::exp(logLog.evaluate(logX)) / exact - 1.0L));
        maximumError = std::max(maximumError, error);
        require(error <= 1.0e-10, "general log-log interpolation supports negative log ordinates");
    }
    auto const ready = logLog.statistics();
    for (int query = 0; query < 100; ++query) {
        logLog.evaluate(-3.0L + query * 0.06L);
    }
    require(logLog.statistics().sampleEvaluations == ready.sampleEvaluations,
        "warm generic log-log interpolation does not call its sampler");
    rejects([&] { InterpolatedCache bad({}, 0.125L); }, "empty sampler is rejected");
    for (long double const badSpacing : {0.0L, -1.0L,
             std::numeric_limits<long double>::infinity()}) {
        rejects([&] { InterpolatedCache bad(polynomial, badSpacing); },
            "invalid generic grid spacing is rejected");
    }
    rejects([&] { InterpolatedCache bad(polynomial, 0.125L, 0); },
        "empty extension block is rejected");
    rejects([&] { InterpolatedCache bad(polynomial, 0.125L, -1); },
        "negative extension block is rejected");
    rejects([&] { signedValues.evaluate(std::numeric_limits<long double>::quiet_NaN()); },
        "nonfinite generic input is rejected");
    rejects([&] { signedValues.evaluate(std::numeric_limits<long double>::max()); },
        "unrepresentable generic grid index is rejected before allocation");
    InterpolatedCache nonfinite([](long double) {
        return std::numeric_limits<long double>::quiet_NaN();
    }, 1.0L, 1);
    rejects([&] { nonfinite.evaluate(0.0L); }, "nonfinite sampled ordinate is rejected");
    require(nonfinite.statistics().samples == 0, "a failed extension does not publish a partial table");
    std::cout << "INTERPOLATOR nonopacity_loglog_max_relative_error=" << maximumError << '\n';
}

} // namespace

int main() {
    try {
        reusable_interpolation();
        std::cout << "Reusable interpolation tests passed\n";
    } catch (std::exception const& error) {
        std::cerr << "Interpolation test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
