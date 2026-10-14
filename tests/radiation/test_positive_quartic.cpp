// Standalone: c++ -std=c++17 -O3 -I. tests/radiation/test_positive_quartic.cpp -o /tmp/test-positive-quartic
// Add --benchmark to compare scalar root costs; no GPU performance is inferred.
#include "octotiger/radiation/positive_quartic.hpp"

#include <stddef.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Coefficients = std::pair<double, double>;

long double referenceRoot(double a, double b) {
    long double lo = 0, hi = 1;
    for (int iteration = 0; iteration < 100; ++iteration) {
        const long double y = 0.5L * (lo + hi);
        if (y == lo || y == hi) {
            break;
        }
        const long double y2 = y * y;
        if (static_cast<long double>(a) * y + static_cast<long double>(b) * y2 * y2 > 1) {
            hi = y;
        } else {
            lo = y;
        }
    }
    return 0.5L * (lo + hi);
}

// The normalized scalar iteration currently in RadiationM1::thermalExchange,
// including its 16-epsilon convergence test and safeguarded Newton step.
double iterativeRoot(double a, double b, int *iterations = nullptr) {
    double lo = 0, hi = 1, y = 1;
    for (int iteration = 0; iteration < 80; ++iteration) {
        const double y2 = y * y;
        const double residual = a * y + b * y2 * y2 - 1;
        if (std::abs(residual) <= 16 * std::numeric_limits<double>::epsilon()) {
            if (iterations != nullptr) {
                *iterations = iteration + 1;
            }
            return y;
        }
        if (residual > 0) {
            hi = y;
        } else {
            lo = y;
        }
        const double next = y - residual / (a + 4 * b * y2 * y);
        y = next > lo && next < hi ? next : 0.5 * (lo + hi);
    }
    throw std::runtime_error("Existing normalized Newton iteration did not converge");
}

std::vector<Coefficients> samples() {
    std::vector<Coefficients> result;
    auto add = [&](double other) {
        result.emplace_back(1., other);
        result.emplace_back(other, 1.);
    };
    add(0.);
    add(1.);
    add(std::numeric_limits<double>::denorm_min());
    add(std::numeric_limits<double>::min());
    add(std::nextafter(1., 0.));
    add(std::nextafter(std::numeric_limits<double>::min(), 0.));
    for (int exponent = -1074; exponent <= 0; ++exponent) {
        const double value = std::ldexp(1., exponent);
        add(value);
        add(std::nextafter(value, 0.));
        if (value < 1.) {
            add(std::nextafter(value, 1.));
        }
    }
    std::mt19937_64 random(0x71a471cULL);
    std::uniform_real_distribution<double> uniform(0., 1.);
    std::uniform_int_distribution<int> exponent(-1074, 0);
    for (int i = 0; i < 32768; ++i) {
        add(uniform(random));
        add(std::ldexp(0.5 + 0.5 * uniform(random), exponent(random)));
    }
    return result;
}

template <class Root> double benchmark(const std::vector<Coefficients> &inputs,
    Root root, double &checksum) {
    constexpr int repeats = 16;
    const auto start = std::chrono::steady_clock::now();
    double sum = 0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (const auto &ab : inputs) {
            sum += root(ab.first, ab.second);
        }
    }
    checksum = sum;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return 1e9 * seconds / (repeats * inputs.size());
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto inputs = samples();
        long double maxError = 0, maxResidual = 0, maxIterativeResidual = 0;
        long double maxFixedError = 0, maxFixedResidual = 0;
        Coefficients worstError{}, worstResidual{};
        int maxIterations = 0;
        size_t totalIterations = 0;
        const long double epsilon = std::numeric_limits<double>::epsilon();
        for (const auto &ab : inputs) {
            const double a = ab.first, b = ab.second;
            const double y = radiation::positiveQuarticRoot(a, b);
            const long double reference = referenceRoot(a, b);
            const long double yWide = y, y2 = yWide * yWide;
            const long double error = std::abs(yWide - reference) / reference;
            const long double residual = std::abs(a * yWide + b * y2 * y2 - 1);
            if (!std::isfinite(y) || y < 0.724 || y > 1 || error > 4 * epsilon || residual > 8 * epsilon) {
                std::cerr << std::setprecision(17) << "a=" << a << " b=" << b << " y=" << y <<
                    " error=" << error << " residual=" << residual << '\n';
                throw std::runtime_error("Positive quartic accuracy or physical interval check failed");
            }
            if ((a == 0 || b == 0) && y != 1.) {
                throw std::runtime_error("Degenerate quartic endpoint is not exact");
            }
            const double fixed = radiation::positiveQuarticRootFixedNewton(a, b);
            const long double fixedWide = fixed, fixed2 = fixedWide * fixedWide;
            const long double fixedError = std::abs(fixedWide - reference) / reference;
            const long double fixedResidual = std::abs(a * fixedWide + b * fixed2 * fixed2 - 1);
            if (!std::isfinite(fixed) || fixed < 0.724 || fixed > 1 ||
                fixedError > 4 * epsilon || fixedResidual > 8 * epsilon ||
                ((a == 0 || b == 0) && fixed != 1.)) {
                throw std::runtime_error("Fixed-count Newton accuracy or interval check failed");
            }
            maxFixedError = std::max(maxFixedError, fixedError);
            maxFixedResidual = std::max(maxFixedResidual, fixedResidual);
            if (error > maxError) {
                maxError = error;
                worstError = ab;
            }
            if (residual > maxResidual) {
                maxResidual = residual;
                worstResidual = ab;
            }
            int iterations = 0;
            const long double oldY = iterativeRoot(a, b, &iterations), oldY2 = oldY * oldY;
            maxIterativeResidual = std::max(maxIterativeResidual,
                std::abs(a * oldY + b * oldY2 * oldY2 - 1));
            maxIterations = std::max(maxIterations, iterations);
            totalIterations += static_cast<size_t>(iterations);
        }
        std::cout << std::setprecision(17) << "samples=" << inputs.size() <<
            " long_double_digits=" << std::numeric_limits<long double>::digits <<
            " max_relative_root_error=" << maxError << " at_a=" << worstError.first << " b=" << worstError.second <<
            "\nmax_wide_residual=" << maxResidual << " at_a=" << worstResidual.first << " b=" << worstResidual.second <<
            "\nfixed6_max_relative_root_error=" << maxFixedError << " fixed6_max_wide_residual=" << maxFixedResidual <<
            "\nexisting_max_wide_residual=" << maxIterativeResidual <<
            " existing_mean_iterations=" << double(totalIterations) / inputs.size() <<
            " existing_max_iterations=" << maxIterations << '\n';
        if (argc == 2 && std::string(argv[1]) == "--benchmark") {
            double analyticChecksum = 0, iterativeChecksum = 0, fixedChecksum = 0;
            const double analyticNs = benchmark(inputs, radiation::positiveQuarticRoot, analyticChecksum);
            const double iterativeNs = benchmark(inputs, [](double a, double b) { return iterativeRoot(a, b); }, iterativeChecksum);
            const double fixedNs = benchmark(inputs, radiation::positiveQuarticRootFixedNewton, fixedChecksum);
            std::cout << "analytic_ns_per_root=" << analyticNs << " existing_ns_per_root=" << iterativeNs <<
                " fixed6_ns_per_root=" << fixedNs << " analytic_checksum=" << analyticChecksum <<
                " existing_checksum=" << iterativeChecksum << " fixed6_checksum=" << fixedChecksum << '\n';
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
