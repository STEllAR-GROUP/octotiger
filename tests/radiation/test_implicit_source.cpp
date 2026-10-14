// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
// Standalone: c++ -std=c++20 -O2 -I. tests/radiation/test_implicit_source.cpp -o /tmp/test_implicit_source
#include "octotiger/radiation/implicit_source.hpp"
#include "octotiger/radiation/m1.hpp"

#include <algorithm>
#include <stddef.h>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
namespace IS = radiation::implicit_source;
using State = IS::State<double>;
using Parameters = IS::Parameters<double>;
size_t checks = 0;
int maximum_iterations = 0;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}
void near(long double actual, long double expected, long double scale, const std::string& message,
    long double tolerance = 8192 * std::numeric_limits<double>::epsilon()) {
    ++checks;
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * std::max(scale, 1.e-280L)) {
        std::ostringstream error;
        error.precision(18);
        error << message << ": actual=" << actual << " expected=" << expected << " scale=" << scale;
        throw std::runtime_error(error.str());
    }
}
long double kinetic(const State& state, double density) {
    long double result = 0;
    for (int d = 0; d < 3; ++d) {
        result += static_cast<long double>(state.momentum[d]) * state.momentum[d] / (2 * density);
    }
    return result;
}
State state(double E = 1, double e = 2, double rho = 3, double c = 10) {
    State result{{E, .13 * E, -.09 * E, .05 * E}, {.003 * c * rho, -.002 * c * rho, .001 * c * rho}, 0, e};
    result.gas_energy = e + kinetic(result, rho);
    return result;
}

// Independent conserved-variable Levermore tensor, not the H/beta factorization
// used by the nonlinear solve and its analytic Jacobian.
void pressure_velocity(const State& y, const Parameters& p, long double* Av, long double& vQ) {
    long double Q2 = 0, velocity[3];
    vQ = 0;
    for (int d = 0; d < 3; ++d) {
        velocity[d] = static_cast<long double>(y.momentum[d]) / p.density;
        Q2 += static_cast<long double>(y.radiation[d + 1]) * y.radiation[d + 1];
        vQ += velocity[d] * y.radiation[d + 1];
    }
    const long double E = y.radiation[0];
    const long double f2 = E > 0 ? Q2 / (E * E) : 0;
    const long double chi = (3 + 4 * f2) / (5 + 2 * std::sqrt(4 - 3 * f2));
    for (int i = 0; i < 3; ++i) {
        Av[i] = E * velocity[i];
        for (int j = 0; j < 3; ++j) {
            const long double isotropic = i == j ? (1 - chi) / 2 : 0;
            const long double directed = Q2 > 0 ? (3 * chi - 1) / 2 * y.radiation[i + 1] * y.radiation[j + 1] / Q2 : 0;
            Av[i] += E * (isotropic + directed) * velocity[j];
        }
    }
}
void validate(const State& rhs, const Parameters& p, double h, const State& y, const std::string& label) {
    require(IS::detail::physical(y), label + " admissible output");
    const long double eps = p.reduced_c_factor;
    const long double chat = eps * p.c;
    const long double energy_scale = std::abs(rhs.gas_energy) + std::abs(y.gas_energy) +
        (std::abs(rhs.radiation[0]) + std::abs(y.radiation[0])) / eps;
    near(y.gas_energy + y.radiation[0] / eps, rhs.gas_energy + rhs.radiation[0] / eps,
        energy_scale, label + " weighted total energy");
    near(y.gas_energy - y.internal_energy - kinetic(y, p.density),
        rhs.gas_energy - rhs.internal_energy - kinetic(rhs, p.density), energy_scale,
        label + " dual-energy offset");
    for (int d = 0; d < 3; ++d) {
        const long double scale = std::abs(rhs.momentum[d]) + std::abs(y.momentum[d]) +
            (std::abs(rhs.radiation[d + 1]) + std::abs(y.radiation[d + 1])) / chat;
        near(y.momentum[d] + y.radiation[d + 1] / chat, rhs.momentum[d] + rhs.radiation[d + 1] / chat,
            scale, label + " weighted momentum");
    }
    long double Av[3], vQ;
    pressure_velocity(y, p, Av, vQ);
    const long double emission = p.marshak ? y.internal_energy :
        (y.internal_energy > 0 ? std::exp(static_cast<long double>(p.log_alpha) + 4 * std::log(static_cast<long double>(y.internal_energy))) : 0);
    const long double emitted = eps * h * p.c * p.absorption * emission;
    const long double absorbed = eps * h * p.c * p.absorption * y.radiation[0];
    const long double work = eps * h * (2 * static_cast<long double>(p.absorption) - p.extinction) * vQ;
    near(y.radiation[0] - static_cast<long double>(rhs.radiation[0]), emitted - absorbed + work,
        std::abs(y.radiation[0]) + std::abs(rhs.radiation[0]) + std::abs(emitted) + std::abs(absorbed) + std::abs(work),
        label + " implicit energy residual");
    for (int d = 0; d < 3; ++d) {
        const long double drag = eps * h * p.extinction * p.c * y.radiation[d + 1];
        const long double advection = eps * h * p.extinction * Av[d];
        near(y.radiation[d + 1] - static_cast<long double>(rhs.radiation[d + 1]), advection - drag,
            std::abs(y.radiation[d + 1]) + std::abs(rhs.radiation[d + 1]) + std::abs(drag) + std::abs(advection),
            label + " implicit momentum residual", 1.e-10L);
    }
}
State solve(const State& rhs, const Parameters& p, double h, const std::string& label) {
    State result{};
    IS::Diagnostics<double> diagnostics;
    const bool okay = IS::solve(rhs, p, h, result, &diagnostics);
    std::ostringstream detail;
    detail << label << " solve failed: status=" << int(diagnostics.failure) << " iterations=" << diagnostics.iterations
        << " residual=" << diagnostics.residual;
    require(okay, detail.str());
    maximum_iterations = std::max(maximum_iterations, diagnostics.iterations);
    validate(rhs, p, h, result, label);
    return result;
}
void thermal_cases() {
    for (double eps : {1., .1, .001}) {
        for (double coefficient : {1.e-80, 1.e-8, 1., 1.e8, 1.e80}) {
            for (double h : {1.e-10, .01, 10., 1.e10}) {
                Parameters p{3., 1., 1., std::log(coefficient), 10., eps};
                State rhs{{1., 0., 0., 0.}, {0., 0., 0.}, 2., 2.};
                const auto result = solve(rhs, p, h, "static thermal");
                // Bisection in long double is an independent positive-root oracle.
                const long double A = static_cast<long double>(h) * p.c * p.absorption;
                const long double w = A / (1 + eps * A);
                const long double total = rhs.internal_energy + w * rhs.radiation[0];
                long double lo = 0, hi = total;
                for (int iteration = 0; iteration < 600; ++iteration) {
                    const long double mid = (lo + hi) / 2;
                    if (mid + w * std::exp(static_cast<long double>(p.log_alpha)) * mid * mid * mid * mid > total) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                near(result.internal_energy, (lo + hi) / 2, (lo + hi) / 2, "quartic oracle");
            }
        }
    }
}
void moving_cases() {
    for (double eps : {1., .1, .003}) {
        for (double absorption : {0., .1, 1.}) {
            for (double h : {1.e-8, .01, 1., 100., 1.e8}) {
                Parameters p{3., absorption, 1., std::log(.2), 10., eps};
                auto rhs = state();
                rhs.gas_energy += .125; // Preserve a pre-existing dual-energy offset.
                solve(rhs, p, h, "moving stage eps=" + std::to_string(eps) + " a=" + std::to_string(absorption) + " h=" + std::to_string(h));
            }
        }
    }
    Parameters p{1., 1., 1., std::log(.0001), 100., .1};
    State rhs{{1., 2., -.5, .3}, {.01, -.02, .03}, 10.0007, 10.};
    require(!IS::detail::physical(rhs), "nonrealizable stage fixture");
    solve(rhs, p, 1., "nonrealizable stage RHS");
    rhs = {{-.1, 0., 0., 0.}, {0., 0., 0.}, 2., 2.};
    p = {1., 1., 1., 0., 1., 1.};
    solve(rhs, p, 1., "negative radiation RHS repaired by implicit emission");
    rhs = {{2., 0., 0., 0.}, {0., 0., 0.}, -.1, -.1};
    solve(rhs, p, 1., "negative gas RHS repaired by implicit absorption");
    p.marshak = true;
    solve(state(), p, .1, "linear emission moving stage");
}
void prescribed_scattering_cases() {
    const State rhs{{1., .4, -.3, .2}, {0., 0., 0.}, 2.125, 2.};
    const auto same_state = [](const State& actual, const State& expected, const std::string& label) {
        for (int d = 0; d < 4; ++d) {
            require(actual.radiation[d] == expected.radiation[d], label + " radiation unchanged");
            if (d < 3) {
                require(actual.momentum[d] == expected.momentum[d], label + " momentum unchanged");
            }
        }
        require(actual.gas_energy == expected.gas_energy, label + " gas energy unchanged");
        require(actual.internal_energy == expected.internal_energy, label + " internal energy unchanged");
    };
    for (double eps : {1., .1, .001}) {
        for (double chi : {0., 1.e-30, 1., 1.e30}) {
            for (double h : {0., 1.e-10, .1, 1.e10}) {
                Parameters p{3., 0., chi, 0., 10., eps, false, false, true};
                State result{};
                IS::Diagnostics<double> diagnostics;
                require(IS::solve(rhs, p, h, result, &diagnostics), "prescribed scattering stage succeeds");
                require(diagnostics.failure == IS::Failure::none && diagnostics.iterations == 0,
                    "prescribed scattering stage needs no nonlinear iteration");
                require(IS::detail::physical(result), "prescribed scattering stage admissible");
                require(result.radiation[0] == rhs.radiation[0], "prescribed scattering retains radiation energy");
                require(result.gas_energy == rhs.gas_energy && result.internal_energy == rhs.internal_energy,
                    "prescribed scattering retains both gas energies");
                const long double eta = static_cast<long double>(h) * p.c * chi * eps;
                for (int d = 0; d < 3; ++d) {
                    require(result.momentum[d] == rhs.momentum[d], "prescribed scattering has no recoil");
                    const long double expected = rhs.radiation[d + 1] / (1 + eta);
                    near(result.radiation[d + 1], expected, std::abs(expected),
                        "prescribed scattering independent backward-Euler oracle");
                }
                if (h == 0 || chi == 0) {
                    same_state(result, rhs, "prescribed scattering identity");
                }
            }
        }
    }
    Parameters p{3., 0., 2., 0., 10., .1, false, false, true};
    State explicit_rhs = rhs;
    explicit_rhs.radiation[1] = 2.;
    explicit_rhs.radiation[2] = -1.;
    require(!IS::detail::physical(explicit_rhs), "prescribed nonrealizable RHS fixture");
    State result{};
    require(IS::solve(explicit_rhs, p, 1., result), "prescribed damping repairs nonrealizable stage RHS");
    require(IS::detail::physical(result), "prescribed repaired stage is admissible");
    near(result.radiation[1], 2.L / 3, 1., "prescribed repaired stage flux");
    require(result.gas_energy == explicit_rhs.gas_energy && result.internal_energy == explicit_rhs.internal_energy,
        "prescribed repaired stage retains gas energies");

    // Each factor is finite, while their product cannot be represented in
    // double precision. Bounded stage weights must still return zero flux.
    p.c = p.extinction = std::numeric_limits<double>::max();
    p.reduced_c_factor = 1.;
    require(IS::solve(rhs, p, std::numeric_limits<double>::max(), result),
        "prescribed extreme stiffness succeeds without overflow");
    State equilibrium = rhs;
    for (int d = 1; d < 4; ++d) {
        equilibrium.radiation[d] = 0.;
    }
    same_state(result, equilibrium, "prescribed infinite-stiffness limit");

    const State sentinel = state(7., 9.);
    for (int policy = 0; policy < 3; ++policy) {
        for (double chi : {0., 1.}) {
            for (double h : {0., 1.}) {
                p = {3., 0., chi, 0., 10., .1, false, false, true};
                p.absorption = policy == 0 ? 1. : 0.;
                p.marshak = policy == 1;
                p.fixed_material = policy == 2;
                result = sentinel;
                IS::Diagnostics<double> diagnostics;
                require(!IS::solve(rhs, p, h, result, &diagnostics) &&
                        diagnostics.failure == IS::Failure::invalid_input,
                    "inconsistent prescribed policy rejected before no-source shortcut");
                same_state(result, sentinel, "invalid prescribed policy leaves output untouched");
            }
        }
    }
    for (int d = 0; d < 3; ++d) {
        for (double h : {0., 1.}) {
            State moving_rhs = rhs;
            moving_rhs.momentum[d] = .1;
            p = {3., 0., 1., 0., 10., .1, false, false, true};
            result = sentinel;
            IS::Diagnostics<double> diagnostics;
            require(!IS::solve(moving_rhs, p, h, result, &diagnostics) &&
                    diagnostics.failure == IS::Failure::invalid_input,
                "prescribed scattering rejects a moving medium");
            same_state(result, sentinel, "moving prescribed medium leaves output untouched");
        }
    }
    p = {3., 0., .01, 0., 10., .1, false, false, true};
    for (int invalid = 0; invalid < 3; ++invalid) {
        State bad_rhs = rhs;
        if (invalid == 0) {
            bad_rhs.radiation[1] = 2.;
        } else if (invalid == 1) {
            bad_rhs.radiation[0] = -1.;
        } else {
            bad_rhs.internal_energy = -1.;
        }
        result = sentinel;
        IS::Diagnostics<double> diagnostics;
        require(!IS::solve(bad_rhs, p, 1., result, &diagnostics) &&
                diagnostics.failure == IS::Failure::no_admissible_seed,
            "prescribed scattering rejects an inadmissible final stage");
        same_state(result, sentinel, "inadmissible prescribed stage leaves output untouched");
    }
}
void limits_and_failures() {
    Parameters p{3., 1., 1., 0., 10., .03};
    auto rhs = state();
    auto result = solve(rhs, p, 0., "zero timestep");
    require(result.radiation[1] == rhs.radiation[1] && result.gas_energy == rhs.gas_energy, "zero step identity");
    p.absorption = p.extinction = 0;
    result = solve(rhs, p, 1.e8, "zero opacity");
    require(result.radiation[1] == rhs.radiation[1] && result.gas_energy == rhs.gas_energy, "zero opacity identity");
    p.absorption = p.extinction = 1;
    rhs = {{0, 0, 0, 0}, {0, 0, 0}, 1., 1.};
    result = solve(rhs, p, 1., "emission into vacuum");
    require(result.radiation[0] > 0, "vacuum emits");
    rhs.gas_energy = rhs.internal_energy = 0;
    solve(rhs, p, 1., "zero total energy");
    State sentinel = state(7., 9.);
    result = sentinel;
    rhs.radiation[0] = -1;
    IS::Diagnostics<double> diagnostics;
    require(!IS::solve(rhs, p, 1., result, &diagnostics), "impossible negative energy rejected");
    require(result.radiation[0] == sentinel.radiation[0] && result.internal_energy == sentinel.internal_energy,
        "failed solve retains output");
    p.density = 0;
    require(!IS::solve(sentinel, p, 1., result, &diagnostics) && diagnostics.failure == IS::Failure::invalid_input,
        "invalid material rejected");

    // A spatial blast-stage RHS previously exceeded the realizability cone.
    // Its weak source (h*c*chi=1.23505e-4) cannot repair it: the unconstrained
    // local root has |Q|/E=1.04815. Do not turn a transport defect into a source
    // success by clipping moments or relaxing the physical-state constraint.
    rhs = {{.014494924247515289, -.015224722153663443,
                -.00037181710646810927, -.00037181710646810927},
        {-1.3060170511291032e-12, -1.4751455384201473e-14, -1.4751455384201454e-14},
        7.6338874272089026, 7.6338874272088324};
    p = {1.2129346010303019e-11, 4.8564155298321212e-12, 4.8564155298321212e-12,
        -9.2817719985559037, 29979245800., 1.};
    result = sentinel;
    require(!IS::solve(rhs, p, .00084829736124204509, result, &diagnostics),
        "weak source rejects the nonrealizable blast-stage RHS");
    for (int d = 0; d < 4; ++d) {
        require(result.radiation[d] == sentinel.radiation[d], "rejected blast stage retains radiation output");
        if (d < 3) {
            require(result.momentum[d] == sentinel.momentum[d], "rejected blast stage retains momentum output");
        }
    }
    require(result.gas_energy == sentinel.gas_energy && result.internal_energy == sentinel.internal_energy,
        "rejected blast stage retains gas energy output");
}
void jacobian() {
    IS::detail::Stage<double> stage;
    stage.rhs = state();
    stage.material = {3., .3, 1., std::log(.2), 10., .1};
    stage.h = .7;
    stage.chat = 1.;
    double weight;
    IS::detail::weights(stage.h, 10., .3, .1, stage.inverse_absorption, weight);
    stage.thermal_weight = weight / .1;
    IS::detail::weights(stage.h, 10., 1., .1, stage.inverse_extinction, stage.extinction_weight);
    for (int d = 0; d < 3; ++d) {
        stage.initial_velocity[d] = stage.rhs.momentum[d] / 3;
    }
    const double base[3] = {.04, -.03, .02};
    IS::detail::Evaluation<double> value;
    require(IS::detail::evaluate(stage, base, value), "Jacobian baseline");
    for (int j = 0; j < 3; ++j) {
        double plus[3], minus[3];
        for (int d = 0; d < 3; ++d) {
            plus[d] = minus[d] = base[d];
        }
        plus[j] += 1.e-6;
        minus[j] -= 1.e-6;
        IS::detail::Evaluation<double> a, b;
        require(IS::detail::evaluate(stage, plus, a) && IS::detail::evaluate(stage, minus, b), "Jacobian perturbations");
        for (int i = 0; i < 3; ++i) {
            const long double derivative = (a.residual[i] - b.residual[i]) / 2.e-6L;
            near(value.jacobian[i][j], derivative, std::max(1.L, std::abs(derivative)), "analytic reduced Jacobian", 1.e-8L);
        }
    }
}

void captured_shock_admissibility() {
    // First failed source stage from the revision-6 level-2 subcritical shock
    // at t=5541.4493523462497. These are the stage density and conservative RHS,
    // not the old cell's material coefficients. The desired Newton correction
    // points outside |Q|<=E even though the thermal solve and Jacobian are sound.
    const State rhs{{3.4275448779631369e-6, 3.4309218936214411e-6, 0., 0.},
        {-.00046602270468789378, 0., 0.}, 152.39264056379983, 12.820117104889135};
    const Parameters p{0x1.abb6eed8919f4p-31, 0x1.568063c90febcp-32,
        0x1.568063c90febcp-32, -0x1.9ed1c10e7c2b1p+4,
        0x1.beb9bf3a00000p+34, 0x1.eb851eb851eb8p-6};
    const double h = .081074606471773059;
    const State sentinel = state();
    State output = sentinel;
    IS::Diagnostics<double> diagnostics;
    require(!IS::solve(rhs, p, h, output, &diagnostics), "captured blocked source stage is rejected");
    require(diagnostics.failure == IS::Failure::inadmissible_iterate,
        "physical line-search obstruction is distinguished from ordinary no descent");
    near(diagnostics.residual, 2.0199589358223418e-7, 1.,
        "captured boundary residual remains unresolved", 1.e-11);
    for (int f = 0; f < 4; ++f) {
        near(output.radiation[f], sentinel.radiation[f], 1., "failed captured stage retains radiation", 0.);
    }
    for (int d = 0; d < 3; ++d) {
        near(output.momentum[d], sentinel.momentum[d], 1., "failed captured stage retains momentum", 0.);
    }
    near(output.gas_energy, sentinel.gas_energy, 1., "failed captured stage retains gas energy", 0.);
    near(output.internal_energy, sentinel.internal_energy, 1., "failed captured stage retains thermal energy", 0.);

    IS::detail::Stage<double> stage{};
    stage.rhs = rhs;
    stage.material = p;
    stage.h = h;
    stage.chat = p.c * p.reduced_c_factor;
    double weight;
    IS::detail::weights(h, p.c, p.absorption, p.reduced_c_factor, stage.inverse_absorption, weight);
    stage.thermal_weight = weight / p.reduced_c_factor;
    IS::detail::weights(h, p.c, p.extinction, p.reduced_c_factor,
        stage.inverse_extinction, stage.extinction_weight);
    for (int d = 0; d < 3; ++d) {
        stage.initial_velocity[d] = rhs.momentum[d] / p.density;
    }
    IS::detail::Evaluation<double> value{};
    double q[3] = {3.3547204537207397e-6, 0., 0.};
    bool inadmissible = false;
    require(!IS::detail::evaluate(stage, q, value, &inadmissible) && inadmissible,
        "finite out-of-cone candidate reports admissibility failure");
    q[0] = 0.;
    require(IS::detail::evaluate(stage, q, value, &inadmissible) && !inadmissible,
        "physical candidate clears the admissibility flag");
    inadmissible = true;
    q[0] = std::numeric_limits<double>::infinity();
    require(!IS::detail::evaluate(stage, q, value, &inadmissible) && !inadmissible,
        "nonfinite candidate is not an admissibility retry");
    q[0] = 0.;
    stage.rhs.internal_energy = -1.;
    inadmissible = true;
    require(!IS::detail::evaluate(stage, q, value, &inadmissible) && !inadmissible,
        "thermal evaluation failure is not misclassified from a partial state");

    // A nearby RHS remains outside the cone but has an admissible source root.
    // Retain that supported use case rather than rejecting or clipping all RHSs.
    State nearby = rhs;
    nearby.radiation[1] *= .99999;
    require(!IS::detail::physical(nearby), "nearby stage RHS still requires source repair");
    solve(nearby, p, h, "nearby repairable shock source");
}

void realizability_roundoff() {
    const double epsilon = std::numeric_limits<double>::epsilon();
    for (double energy : {1.e-200, 1., 1.e200}) {
        for (double excess : {0., 1024 * epsilon, 2048 * epsilon, 3072 * epsilon, 1.e-10}) {
            State rhs{{energy, energy * (1 + excess), 0., 0.}, {0., 0., 0.}, 1., 1.};
            const double normalized = rhs.radiation[1] / rhs.radiation[0];
            const bool expected = normalized * normalized <= 1 + 4096 * epsilon;
            require(IS::detail::physical(rhs) == expected, "source matches M1 squared realizability tolerance");
#ifndef NDEBUG
            bool m1_physical = true;
            try {
                RadiationM1<double>::ConservedState{rhs.radiation[0], rhs.radiation[1], 0., 0.}.checkState();
            } catch (const std::runtime_error&) {
                m1_physical = false;
            }
            require(IS::detail::physical(rhs) == m1_physical, "source and actual M1 checker agree on roundoff beams");
#endif
            Parameters p{1., 1., 1., 0., 3.e10, 1.};
            State output{};
            require(IS::solve(rhs, p, 0., output) == expected,
                "zero source step preserves the shared physical-state contract");
        }
    }
    State vacuum{{0., 0., 0., 0.}, {0., 0., 0.}, 1., 1.};
    require(IS::detail::physical(vacuum), "exact vacuum remains admissible");
    vacuum.radiation[1] = std::numeric_limits<double>::denorm_min();
    require(!IS::detail::physical(vacuum), "any nonzero flux in exact vacuum is rejected");
    vacuum.radiation[0] = std::numeric_limits<double>::denorm_min();
    vacuum.radiation[1] = std::numeric_limits<double>::max();
    require(!IS::detail::physical(vacuum), "huge flux over tiny energy is rejected without division overflow");
}

} // namespace

int main() {
    try {
#if defined(__GLIBC__)
        const int previous_traps = feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
#endif
        thermal_cases();
        moving_cases();
        prescribed_scattering_cases();
        limits_and_failures();
        jacobian();
        captured_shock_admissibility();
        realizability_roundoff();
#if defined(__GLIBC__)
        require(fegetexcept() == (previous_traps | FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW),
            "source solve retains the caller's floating-point trap settings");
        fedisableexcept(FE_ALL_EXCEPT);
        feenableexcept(previous_traps);
#endif
        std::cout << checks << " implicit source checks passed; maximum Newton iterations " << maximum_iterations << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
