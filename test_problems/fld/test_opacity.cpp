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

#include "octotiger/fld_opacity.hpp"
#include "octotiger/interpolated_cache.hpp"

namespace {

using octotiger::fld::OpacityModel;
using octotiger::fld::OpacityParameters;
using octotiger::fld::evaluate_opacity;

void require(bool const condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void relative(double const actual, double const expected, double const tolerance,
    std::string const& message) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <= tolerance * std::abs(expected),
        message + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
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

// Independent reference: composite Simpson integration on log frequency. The
// production calculation instead uses Gauss-Legendre rules on linear frequency.
// Keep the Planck and Rosseland weight normalizations numerical here as well.
std::array<long double, 2> reference_means(long double const amplitude,
    long double const scattering, int const intervals = 32768) {
    long double const lower = -30.0L;
    long double const upper = std::log(100.0L);
    long double const step = (upper - lower) / intervals;
    long double planckNumerator = 0.0L;
    long double planckDenominator = 0.0L;
    long double rosselandNumerator = 0.0L;
    long double rosselandDenominator = 0.0L;
    for (int index = 0; index <= intervals; ++index) {
        long double const frequency = std::exp(lower + index * step);
        long double const occupation = 1.0L / std::expm1(frequency);
        long double const stimulated = -std::expm1(-frequency);
        long double const absorption = amplitude * stimulated /
            (frequency * frequency * frequency);
        long double const planckWeight = std::pow(frequency, 3) * occupation;
        long double const rosselandWeight = std::pow(frequency, 4) * occupation * (1.0L + occupation);
        long double const simpson = (index == 0 || index == intervals) ? 1.0L :
            (index % 2 == 0 ? 2.0L : 4.0L);
        long double const jacobian = frequency * simpson;
        planckNumerator += jacobian * absorption * planckWeight;
        planckDenominator += jacobian * planckWeight;
        rosselandNumerator += jacobian * rosselandWeight;
        rosselandDenominator += jacobian * rosselandWeight / (absorption + scattering);
    }
    return {{planckNumerator / planckDenominator,
        rosselandNumerator / rosselandDenominator}};
}

void legacy_and_independent() {
    OpacityParameters p;
    p.absorption = 3.0;
    p.scattering = 2.0;
    p.densityExponent = 1.5;
    p.temperatureExponent = -2.0;
    auto const old = evaluate_opacity(p, 4.0, 2.0);
    relative(old.planckAbsorption, 6.0, 3.0e-15, "legacy power law retains inverse-length units");
    relative(old.scatteringExtinction, 8.0, 0.0, "mass scattering receives one density factor");
    relative(old.rosselandExtinction, 14.0, 3.0e-15, "legacy total extinction is unchanged");
    require(!old.regularized, "positive legacy temperature should not be regularized");

    p.model = OpacityModel::independent;
    p.rosseland = 1000.0;
    p.rosselandDensityExponent = 1.0;
    auto const independent = evaluate_opacity(p, 4.0, 2.0);
    relative(independent.planckAbsorption, old.planckAbsorption, 0.0, "independent Planck unchanged");
    relative(independent.rosselandExtinction, 4000.0, 3.0e-15,
        "independent Rosseland is total; scattering is not added twice");
    p.absorption = 1.0e9;
    auto const changedPlanck = evaluate_opacity(p, 4.0, 2.0);
    relative(changedPlanck.rosselandExtinction, independent.rosselandExtinction, 0.0,
        "Planck changes cannot change independent transport");
    p.rosseland = 1.0e-9;
    auto const changedRosseland = evaluate_opacity(p, 4.0, 2.0);
    relative(changedRosseland.planckAbsorption, changedPlanck.planckAbsorption, 0.0,
        "transport changes cannot change Planck absorption");
    p.absorption = 0.0;
    auto const pureScattering = evaluate_opacity(p, 4.0, 2.0);
    require(pureScattering.planckAbsorption == 0.0, "pure scattering creates no thermal source");
}

void harmonic_averaging() {
    using octotiger::fld::opacity_detail::rosseland_mean;
    double maximumError = 0.0;
    for (long double const amplitude : {0.0L, 1.0e-12L, 1.0e-8L, 1.0e-4L, 1.0L,
             1.0e4L, 1.0e8L, 1.0e12L, 1.0e32L}) {
        auto const coarse = reference_means(amplitude, 1.0L, 16384);
        auto const reference = reference_means(amplitude, 1.0L);
        relative(double(coarse[1]), double(reference[1]), 2.0e-12,
            "independent reference quadrature converges");
        double const actual = double(rosseland_mean(amplitude, 1.0L));
        relative(actual, double(reference[1]), 4.0e-11,
            "Rosseland mean agrees with independently integrated total extinction");
        maximumError = std::max(maximumError, std::abs(actual / double(reference[1]) - 1.0));
        require(actual >= 1.0 - 1.0e-14, "total Rosseland cannot be below constant scattering");
        if (amplitude > 0.0L) {
            relative(double(amplitude / octotiger::fld::opacity_detail::planckNormalization),
                double(reference[0]), 2.0e-12, "analytic Planck mean has correct normalization");
        }
    }

    long double const pi = std::acos(-1.0L);
    long double const zeta6 = std::pow(pi, 6) / 945.0L;
    long double const zeta7 = 1.00834927738192282683979754984979676L;
    long double const exactFreeFree = (4.0L * std::pow(pi, 4) / 15.0L) /
        (2520.0L * (zeta6 + zeta7));
    relative(double(rosseland_mean(1.0L, 0.0L)), double(exactFreeFree), 4.0e-12,
        "pure free-free Rosseland matches its analytic zeta-function integral");
    relative(double(rosseland_mean(0.0L, 7.0L)), 7.0, 0.0,
        "frequency-independent scattering mean is exact");
    double const mixed = double(rosseland_mean(1000.0L, 1.0L));
    double const naiveSum = double(rosseland_mean(1000.0L, 0.0L)) + 1.0;
    require(std::abs(mixed - naiveSum) > 0.05 * mixed,
        "mixed mean must expose the failure of separately averaging then adding");
    relative(double(rosseland_mean(1.0e-200L, 1.0e-201L)),
        double(rosseland_mean(10.0L, 1.0L) * 1.0e-201L), 4.0e-12,
        "harmonic averaging handles very small coefficients");
    relative(double(rosseland_mean(1.0e200L, 1.0e199L)),
        double(rosseland_mean(10.0L, 1.0L) * 1.0e199L), 4.0e-12,
        "harmonic averaging handles very large coefficients");
    std::cout << "FLD_OPACITY mean_max_relative_error=" << maximumError << '\n';
}

void physical_units_composition_and_floors() {
    OpacityParameters p;
    p.model = OpacityModel::free_free_thomson;
    p.temperatureFloor = 1.0e4;
    double const rho = 1.0e-5;
    double const temperature = 2.0e6;
    auto const helium = evaluate_opacity(p, rho, temperature, 0.5, 1.0);
    auto const carbon = evaluate_opacity(p, rho, temperature, 0.5, 3.0);
    relative(carbon.planckAbsorption, 3.0 * helium.planckAbsorption, 1.0e-14,
        "free-free absorption includes the ionic charge-squared composition moment");
    relative(carbon.scatteringExtinction, helium.scatteringExtinction, 0.0,
        "He and C have the same electrons per mass in this ionized model");
    relative(helium.scatteringExtinction, rho * 0.5 / 1.6733e-24 * 6.6524587321e-25,
        3.0e-15, "Thomson coefficient has the physical cross section and one density factor");

    long double const amplitude = 3.692e8L * std::pow(static_cast<long double>(rho) / 1.6733e-24L, 2) *
        0.5L * std::pow(6.6260755e-27L / 1.380658e-16L, 3) *
        std::pow(static_cast<long double>(temperature), -3.5L);
    auto const reference = reference_means(amplitude, helium.scatteringExtinction);
    relative(helium.planckAbsorption, double(reference[0]), 2.0e-12,
        "physical free-free normalization matches dimensional monochromatic reference");
    relative(helium.rosselandExtinction, double(reference[1]), 4.0e-11,
        "physical transport matches integrated free-free plus Thomson opacity");

    p.lengthUnit = 1.0e8;
    p.densityUnit = 1.0e6;
    auto const rescaled = evaluate_opacity(p, rho / p.densityUnit, temperature, 0.5, 1.0);
    relative(rescaled.planckAbsorption / p.lengthUnit, helium.planckAbsorption, 2.0e-14,
        "changing code units preserves physical Planck absorption");
    relative(rescaled.rosselandExtinction / p.lengthUnit, helium.rosselandExtinction, 2.0e-14,
        "changing code units preserves physical Rosseland extinction");
    relative(rescaled.scatteringExtinction / p.lengthUnit, helium.scatteringExtinction, 3.0e-15,
        "changing code units preserves physical scattering");

    double const coldTemperature = 0.0;
    auto const cold = evaluate_opacity(p, rho / p.densityUnit, coldTemperature);
    auto const cutoff = evaluate_opacity(p, rho / p.densityUnit, p.temperatureFloor);
    require(cold.regularized && !cutoff.regularized,
        "provider reports exactly when the opacity temperature cutoff is active");
    relative(cold.planckAbsorption, cutoff.planckAbsorption, 0.0,
        "cold opacity is finite and evaluated at its explicit cutoff");
    require(coldTemperature == 0.0 && cold.planckAbsorption * std::pow(coldTemperature, 4) == 0.0,
        "opacity regularization never turns zero-temperature material into an emitter");
    p.temperatureFloor *= 10.0;
    auto const warmerCutoff = evaluate_opacity(p, rho / p.densityUnit, coldTemperature);
    relative(warmerCutoff.planckAbsorption / cold.planckAbsorption, std::pow(10.0, -3.5),
        2.0e-14, "cutoff sensitivity follows the documented free-free law");
    require(warmerCutoff.planckAbsorption * std::pow(coldTemperature, 4) == 0.0,
        "a higher opacity cutoff still cannot create a thermal source at zero temperature");

    p.gaunt = 0.0;
    auto const scattering = evaluate_opacity(p, rho / p.densityUnit, coldTemperature);
    require(scattering.planckAbsorption == 0.0, "zero Gaunt factor selects scattering-only physics");
    relative(scattering.rosselandExtinction, scattering.scatteringExtinction, 0.0,
        "physical pure-scattering mean is exactly Thomson extinction");
}

void invalid_inputs() {
    OpacityParameters p;
    relative(evaluate_opacity(p, 1.0, 0.0).planckAbsorption, 1.0, 0.0,
        "constant opacity is defined at zero material temperature without a cutoff");
    p.temperatureExponent = -1.0;
    rejects([&] { evaluate_opacity(p, 1.0, 0.0); }, "a singular power law requires an explicit cutoff");
    p.temperatureFloor = 1.0;
    require(evaluate_opacity(p, 1.0, 0.0).regularized,
        "legacy mode accepts actual zero temperature only with a configured cutoff");
    rejects([&] { evaluate_opacity(p, 0.0, 1.0); }, "zero density is rejected");
    rejects([&] { evaluate_opacity(p, 1.0, -1.0); }, "negative material temperature is rejected");
    rejects([&] { evaluate_opacity(p, 1.0, std::numeric_limits<double>::quiet_NaN()); },
        "a cutoff must not hide a nonfinite material temperature");
    p.model = OpacityModel::free_free_thomson;
    p.temperatureFloor = 0.0;
    rejects([&] { evaluate_opacity(p, 1.0, 1.0e6); }, "physical mode requires a documented cutoff");
    p.temperatureFloor = 1.0e4;
    rejects([&] { evaluate_opacity(p, 1.0, 1.0e6, -0.5, 1.0); }, "negative composition is rejected");
    p.lengthUnit = 0.0;
    rejects([&] { evaluate_opacity(p, 1.0, 1.0e6); }, "invalid physical length units are rejected");
    p.lengthUnit = 1.0;
    p.gaunt = -1.0;
    rejects([&] { evaluate_opacity(p, 1.0, 1.0e6); }, "negative Gaunt factor is rejected");
    p.gaunt = 1.0;
    rejects([&] { evaluate_opacity(p, 1.0e200, 1.0e4); },
        "unrepresentable coefficients are reported instead of silently capped");
    p.model = static_cast<OpacityModel>(-1);
    rejects([&] { evaluate_opacity(p, 1.0, 1.0e6); }, "unknown opacity model is rejected");
}

void zero_temperature_power_laws() {
    OpacityParameters p;
    p.absorption = 0.0;
    p.scattering = 2.0;
    p.temperatureExponent = -3.5;
    auto const scattering = evaluate_opacity(p, 4.0, 0.0);
    require(scattering.planckAbsorption == 0.0 && !scattering.regularized,
        "zero absorption stays zero at zero temperature even with a negative exponent");
    relative(scattering.rosselandExtinction, 8.0, 0.0,
        "pure scattering works at zero temperature without an artificial cutoff");
    p.absorption = 3.0;
    p.temperatureExponent = 2.0;
    require(evaluate_opacity(p, 4.0, 0.0).planckAbsorption == 0.0,
        "positive temperature exponent has the analytic zero-temperature limit");
    p.temperatureExponent = 0.0;
    p.densityExponent = 1.0;
    relative(evaluate_opacity(p, 4.0, 0.0).planckAbsorption, 12.0, 3.0e-15,
        "temperature-independent absorption retains density dependence at zero temperature");
    p.model = OpacityModel::independent;
    p.rosseland = 7.0;
    relative(evaluate_opacity(p, 4.0, 0.0).rosselandExtinction, 7.0, 3.0e-15,
        "independent constant total transport needs no temperature cutoff");
    p.rosselandTemperatureExponent = 1.0;
    require(evaluate_opacity(p, 4.0, 0.0).rosselandExtinction == 0.0,
        "positive total transport exponent has a zero limit; solver validates positivity");
    p.rosselandTemperatureExponent = -1.0;
    rejects([&] { evaluate_opacity(p, 4.0, 0.0); },
        "independent singular transport requires its explicit cutoff");
    p.rosseland = 0.0;
    require(evaluate_opacity(p, 4.0, 0.0).rosselandExtinction == 0.0,
        "disabled independent transport remains zero despite a negative exponent");
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

void interpolation_cache() {
    using namespace octotiger::fld::opacity_detail;
    LogRosselandCache cache(log_rosseland_ratio);
    require(cache.statistics().samples == 0, "Rosseland cache starts without pretabulation");
    long double const first = cache.evaluate(0.25L);
    auto const initial = cache.statistics();
    require(initial.intervals == 128 && initial.sampleEvaluations == initial.samples,
        "first request builds only its padded anchored interval block");
    for (int repeat = 0; repeat < 20; ++repeat) {
        relative(double(cache.evaluate(0.25L)), double(first), 0.0,
            "repeated lookup is deterministic");
        cache.evaluate(0.75L);
    }
    require(cache.statistics().sampleEvaluations == initial.sampleEvaluations,
        "queries inside the cached range perform no quadrature sampling");
    cache.evaluate(-2.125L);
    cache.evaluate(3.875L);
    auto const expanded = cache.statistics();
    require(expanded.lower < initial.lower && expanded.upper > initial.upper &&
            expanded.sampleEvaluations == expanded.samples,
        "extensions on both ends reuse all existing quadrature samples");
    relative(double(cache.evaluate(0.25L)), double(first), 0.0,
        "extending the table cannot change existing polynomial coefficients");

    double maximumError = 0.0;
    long double previous = 0.0L;
    for (int index = 0; index <= 12000; ++index) {
        // Avoid only testing cached nodes: the irrational shift visits different
        // fractions of intervals while scanning both physical asymptotes.
        long double const coordinate = -40.0L + (index + 0.3141592653589793L) * 0.01L;
        long double const value = cache.evaluate(coordinate);
        long double const exact = log_rosseland_ratio(coordinate);
        double const error = double(std::abs(std::expm1(value - exact)));
        maximumError = std::max(maximumError, error);
        require(error <= 1.0e-10, "log-log Rosseland interpolation exceeds its relative error target");
        require(value >= 0.0L && value >= previous - 2.0e-13L,
            "interpolated total opacity is positive and monotone through both limits");
        previous = value;
    }
    auto const warm = cache.statistics();
    for (long double const boundary : {warm.lower, warm.upper - LogRosselandCache::spacing}) {
        for (long double const offset : {0.0L, 1.0e-12L, -1.0e-12L}) {
            long double const coordinate = boundary + offset;
            relative(double(std::exp(cache.evaluate(coordinate) -
                         log_rosseland_ratio(coordinate))), 1.0, 1.0e-10,
                "cache endpoints extend safely rather than using an out-of-range stencil");
        }
    }

    LogRosselandCache reverse(log_rosseland_ratio);
    reverse.evaluate(7.123L);
    reverse.evaluate(-7.123L);
    for (long double const coordinate : {-6.125L, -0.173L, 0.25L, 3.987L, 6.625L}) {
        relative(double(reverse.evaluate(coordinate)),
            double(cache.evaluate(coordinate)), 0.0,
            "anchored interpolation is independent of query order and cache history");
    }
    rejects([&] { cache.evaluate(std::numeric_limits<long double>::infinity()); },
        "nonfinite logarithmic cache coordinates are rejected");
    rejects([&] { rosseland_mean(-1.0L, 1.0L); }, "negative opacity amplitude is rejected");
    require(rosseland_mean(0.0L, 0.0L) == 0.0L, "zero amplitudes retain their exact transparent limit");
    // Ratio overflows double; the physical mean remains representable.
    relative(double(rosseland_mean(1.0e200L, 1.0e-200L)),
        double(rosseland_mean(1.0e200L, 0.0L)), 1.0e-10,
        "large opacity ratios never require forming or exponentiating the ratio");

    std::cout << "FLD_OPACITY interpolation_max_relative_error=" << maximumError
              << " cached_intervals=" << cache.statistics().intervals
              << " cache_sample_evaluations=" << cache.statistics().sampleEvaluations << '\n';
}

void concurrent_cache() {
    using namespace octotiger::fld::opacity_detail;
    LogRosselandCache cache(log_rosseland_ratio);
    std::array<std::exception_ptr, 2> errors{};
    std::array<std::thread, 2> workers;
    for (size_t worker = 0; worker < workers.size(); ++worker) {
        workers[worker] = std::thread([&, worker]() {
            try {
                for (int query = 0; query < 40; ++query) {
                    long double const coordinate = (worker == 0 ? -1.0L : 1.0L) *
                        (0.171L + query * 0.075L);
                    long double const actual = cache.evaluate(coordinate);
                    relative(double(std::exp(actual - log_rosseland_ratio(coordinate))),
                        1.0, 1.0e-10, "concurrent cache extension returns the correct coefficient");
                }
            } catch (...) {
                errors[worker] = std::current_exception();
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    for (auto const& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    auto const ready = cache.statistics();
    for (int query = 0; query < 100; ++query) {
        cache.evaluate(-2.9L + query * 0.058L);
    }
    require(cache.statistics().sampleEvaluations == ready.sampleEvaluations,
        "concurrent publication leaves a complete range with no interior cache holes");
}

void bounded_cache_timing() {
    using namespace octotiger::fld::opacity_detail;
    std::array<long double, 101> amplitudes{};
    for (size_t index = 0; index < amplitudes.size(); ++index) {
        amplitudes[index] = std::exp((static_cast<long double>(index) - 50.0L) * 0.1L);
        rosseland_mean(amplitudes[index], 1.0L);
    }
    constexpr int queries = 5000;
    auto const ready = rosseland_cache().statistics();
    long double cachedSum = 0.0L;
    auto const start = std::chrono::steady_clock::now();
    for (int query = 0; query < queries; ++query) {
        cachedSum += rosseland_mean(amplitudes[query % amplitudes.size()], 1.0L);
    }
    auto const middle = std::chrono::steady_clock::now();
    long double directSum = 0.0L;
    for (int query = 0; query < queries; ++query) {
        directSum += rosseland_mean_quadrature(amplitudes[query % amplitudes.size()], 1.0L);
    }
    auto const end = std::chrono::steady_clock::now();
    require(rosseland_cache().statistics().sampleEvaluations == ready.sampleEvaluations,
        "all bounded warm-cache queries avoid quadrature sampling");
    relative(double(cachedSum), double(directSum), 1.0e-10,
        "timed cached and direct means agree");
    std::cout << "FLD_OPACITY warm_queries=" << queries
              << " warm_cache_seconds=" << std::chrono::duration<double>(middle - start).count()
              << " direct_quadrature_seconds=" << std::chrono::duration<double>(end - middle).count()
              << " warm_cache_new_samples=0\n";
}

}    // namespace

int main() {
    try {
        legacy_and_independent();
        harmonic_averaging();
        physical_units_composition_and_floors();
        invalid_inputs();
        zero_temperature_power_laws();
        reusable_interpolation();
        interpolation_cache();
        concurrent_cache();
        bounded_cache_timing();
        std::cout << "FLD independent opacity tests passed\n";
    } catch (std::exception const& error) {
        std::cerr << "FLD opacity test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
