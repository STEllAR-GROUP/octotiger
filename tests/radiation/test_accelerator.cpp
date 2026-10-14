// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
// Standalone CPU: c++ -std=c++20 -O2 -I. tests/radiation/test_accelerator.cpp
//     src/radiation/accelerator.cpp -o /tmp/test_radiation_accelerator
#include "octotiger/radiation/accelerator.hpp"
#include "octotiger/radiation/m1.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace Acc = radiation_accelerator;
using M1 = RadiationM1<double, 3>;
size_t checks = 0;

void require(bool valid, const char* message) {
    ++checks;
    if (!valid) {
        throw std::runtime_error(message);
    }
}
void near(double value, long double expected, long double scale, const char* message,
    long double epsilon_count = 2048) {
    ++checks;
    const long double tolerance = epsilon_count * std::numeric_limits<double>::epsilon() *
        std::max(std::abs(expected), std::abs(scale));
    if (!std::isfinite(value) || std::abs(value - expected) > tolerance) {
        std::ostringstream error;
        error.precision(19);
        error << message << ": got " << value << ", expected " << expected << ", tolerance " << tolerance;
        throw std::runtime_error(error.str());
    }
}
template <class Function>
void throws(Function function, const char* message) {
    bool caught = false;
    try {
        function();
    } catch (std::exception const&) {
        caught = true;
    }
    require(caught, message);
}
size_t cells(Acc::Grid g) {
    return size_t(g.n) * g.n * g.n;
}
size_t index(Acc::Grid g, int i, int j, int k) {
    return (size_t(i) * g.n + j) * g.n + k;
}
bool interior(Acc::Grid g, size_t r) {
    const int coord[3] = {int(r / (g.n * g.n)), int(r / g.n % g.n), int(r % g.n)};
    for (int d = 0; d < 3; ++d) {
        if (coord[d] < g.bw || coord[d] >= g.n - g.bw) {
            return false;
        }
    }
    return true;
}
M1::Primitives primitive(std::vector<double> const& u, size_t count, size_t r, double c) {
    M1::ConservedState state;
    state[0] = u[r];
    for (int f = 1; f < 4; ++f) {
        state[f] = u[size_t(f) * count + r] / c;
    }
    return state.toPrimitives();
}
std::vector<double> reference_flux(Acc::Grid g, std::vector<double> const& u,
    std::vector<double> const& x) {
    const size_t count = cells(g);
    std::vector<double> result(12 * count, 0);
    const int stride[3] = {g.n * g.n, g.n, 1};
    for (int normal = 0; normal < 3; ++normal) {
        int hi[3] = {g.n - g.bw, g.n - g.bw, g.n - g.bw};
        ++hi[normal];
        for (int i = g.bw; i < hi[0]; ++i) {
            for (int j = g.bw; j < hi[1]; ++j) {
                for (int k = g.bw; k < hi[2]; ++k) {
                    const size_t right = index(g, i, j, k), left = right - stride[normal];
                    const auto qleft = M1::reconstructPrimitives(
                        primitive(u, count, left - stride[normal], g.c), primitive(u, count, left, g.c),
                        primitive(u, count, right, g.c)).second;
                    const auto qright = M1::reconstructPrimitives(
                        primitive(u, count, left, g.c), primitive(u, count, right, g.c),
                        primitive(u, count, right + stride[normal], g.c)).first;
                    double vg = 0;
                    if (normal != 2) {
                        const int transverse = normal == 0 ? 1 : 0;
                        vg = (normal == 0 ? -g.omega : g.omega) * 0.5 *
                            (x[size_t(transverse) * count + left] + x[size_t(transverse) * count + right]);
                    }
                    const auto flux = M1::hll(qleft, qright, normal, g.c * g.reduced_c_factor, vg);
                    for (int f = 0; f < 4; ++f) {
                        result[size_t(4 * normal + f) * count + right] = flux[f] * (f ? g.c : 1);
                    }
                }
            }
        }
    }
    return result;
}
void transport(Acc::Backend backend, double c, double eps, double rotation, int mode) {
    Acc::Grid g{8, 2, 0.25, c, eps, 0.001 / (c * eps), rotation * c * eps};
    const size_t count = cells(g);
    std::vector<double> u(4 * count), x(3 * count);
    double energy_scale = mode == 0 ? 0 : (mode == 1 ? 1e-12 : (mode == 2 ? 1e12 : 1));
    for (int i = 0; i < g.n; ++i) {
        for (int j = 0; j < g.n; ++j) {
            for (int k = 0; k < g.n; ++k) {
                const size_t r = index(g, i, j, k);
                x[r] = (i - 3.5) * g.dx;
                x[count + r] = (j - 3.5) * g.dx;
                x[2 * count + r] = (k - 3.5) * g.dx;
                u[r] = energy_scale * (mode == 3 ? 1 + 0.1 * std::sin(i + 2 * j - k) : 1);
                if (mode == 2) {
                    // Exactly streaming, parallel and transverse to different faces.
                    u[count + r] = c * u[r];
                } else if (mode == 3) {
                    u[count + r] = c * u[r] * 0.25 * std::sin(0.3 * i + j);
                    u[2 * count + r] = c * u[r] * 0.2 * std::cos(0.2 * j - k);
                    u[3 * count + r] = c * u[r] * 0.15 * std::sin(0.6 * k + i);
                }
            }
        }
    }
    const auto expected_flux = reference_flux(g, u, x);
    std::vector<double> flux(1, -123);
    Acc::compute_flux(backend, g, u, x, flux);
    require(flux.size() == 12 * count, "flux output shape");
    for (int field = 0; field < 12; ++field) {
        const double scale = energy_scale * c * eps * (field % 4 ? c : 1);
        for (size_t r = 0; r < count; ++r) {
            near(flux[size_t(field) * count + r], expected_flux[size_t(field) * count + r], scale,
                "independent M1 reconstructed face flux");
        }
    }
    if (mode == 3) {
        // Mimic AMR reflux between flux construction and the update. A backend
        // that reuses its earlier device flux allocation would miss this change.
        const size_t face = index(g, g.bw + 1, g.bw + 1, g.bw + 1);
        for (int f = 0; f < 4; ++f) {
            flux[size_t(f) * count + face] += (f + 1) * 1e-5 * c * eps * (f ? c : 1);
        }
    }
    const auto original = u;
    std::vector<double> before(1, -456);
    Acc::advance(backend, g, u, flux, before);
    require(before.size() == 2 * count, "rotation output shape");
    for (size_t r = 0; r < count; ++r) {
        if (!interior(g, r)) {
            for (int f = 0; f < 4; ++f) {
                require(u[size_t(f) * count + r] == original[size_t(f) * count + r], "advance preserves ghosts");
            }
            continue;
        }
        long double expected[4];
        for (int f = 0; f < 4; ++f) {
            long double divergence = 0;
            const int stride[3] = {g.n * g.n, g.n, 1};
            for (int d = 0; d < 3; ++d) {
                const size_t base = size_t(4 * d + f) * count + r;
                divergence += static_cast<long double>(flux[base + stride[d]]) - flux[base];
            }
            expected[f] = original[size_t(f) * count + r] -
                static_cast<long double>(g.dt) / g.dx * divergence;
        }
        near(before[r], expected[1], energy_scale * c, "transported Fx before rotation");
        near(before[count + r], expected[2], energy_scale * c, "transported Fy before rotation");
        const long double angle = static_cast<long double>(g.omega) * g.dt;
        const long double fx = expected[1], fy = expected[2];
        expected[1] = std::cos(angle) * fx + std::sin(angle) * fy;
        expected[2] = std::cos(angle) * fy - std::sin(angle) * fx;
        for (int f = 0; f < 4; ++f) {
            near(u[size_t(f) * count + r], expected[f], energy_scale * (f ? c : 1),
                "finite volume update including reflux and rotation");
        }
    }
}

void compare_coupling(Acc::Backend backend, std::vector<Acc::CouplingInput> const& input,
    double dt, double c, double eps, bool marshak) {
    using Reference = RadiationM1<long double, 3>;
    std::vector<Acc::CouplingOutput> output;
    Acc::couple(backend, input, dt, c, eps, marshak, output);
    require(output.size() == input.size(), "coupling output shape");
    for (size_t i = 0; i < input.size(); ++i) {
        auto const& in = input[i];
        auto const& out = output[i];
        Reference::ConservedState radiation;
        Reference::SpatialVector momentum;
        for (int f = 0; f < 4; ++f) {
            radiation[f] = in.radiation[f];
        }
        for (int d = 0; d < 3; ++d) {
            momentum[d] = in.momentum[d];
        }
        const auto expected = radiation.couple(momentum, in.gas_energy, in.internal_energy,
            in.density, in.absorption, in.extinction, dt, c, in.log_alpha, marshak, eps);
        const long double energy_scale = in.internal_energy + in.radiation[0] / eps;
        for (int f = 0; f < 4; ++f) {
            near(out.radiation[f], expected.radiation[f],
                std::max(std::abs(expected.radiation[f]), 1e-290L), "long-double M1 coupling radiation", 4096);
        }
        for (int d = 0; d < 3; ++d) {
            near(out.momentum[d], expected.momentum[d],
                std::abs(in.momentum[d]) + in.radiation[0] / (eps * c), "long-double M1 coupling momentum");
            near(out.momentum[d] + out.radiation[d + 1] / (eps * c),
                static_cast<long double>(in.momentum[d]) + in.radiation[d + 1] / (eps * c),
                std::abs(in.momentum[d]) + in.radiation[0] / (eps * c), "RSLA momentum invariant");
        }
        near(out.gas_energy, expected.gasEnergy, energy_scale, "long-double M1 gas energy");
        near(out.internal_energy, expected.internalEnergy,
            std::max(std::abs(expected.internalEnergy), 1e-290L), "long-double M1 internal energy", 4096);
        near(out.gas_energy + out.radiation[0] / eps,
            static_cast<long double>(in.gas_energy) + in.radiation[0] / eps,
            energy_scale, "RSLA energy invariant");
    }
}
void coupling(Acc::Backend backend) {
    for (double c : {10.0, 2.99792458e10}) {
        for (double eps : {1.0, 0.03}) {
            for (bool marshak : {false, true}) {
                std::vector<Acc::CouplingInput> input;
                for (double opacity : {0.0, 1e-6, 1.0, 1e4, 1e10}) {
                    for (double log_alpha : {-1000.0, -10.0, 0.0, 10.0, 1000.0}) {
                        Acc::CouplingInput in{};
                        in.radiation[0] = 1;
                        in.density = 10;
                        in.internal_energy = 100;
                        in.gas_energy = 100;
                        in.absorption = opacity;
                        in.extinction = opacity;
                        in.log_alpha = log_alpha;
                        input.push_back(in);
                    }
                }
                Acc::CouplingInput vacuum{};
                vacuum.density = 1;
                input.push_back(vacuum);
                Acc::CouplingInput streaming{};
                streaming.radiation[0] = streaming.radiation[1] = 1;
                streaming.density = 10;
                streaming.gas_energy = streaming.internal_energy = 100;
                streaming.extinction = 1e8;
                input.push_back(streaming);
                Acc::CouplingInput moving{};
                moving.radiation[0] = 1;
                moving.radiation[1] = 0.2;
                moving.radiation[2] = -0.1;
                moving.radiation[3] = 0.05;
                moving.momentum[0] = 0.01;
                moving.momentum[1] = -0.02;
                moving.density = 10;
                moving.gas_energy = 100.000025;
                moving.internal_energy = 100;
                moving.absorption = 0.2;
                moving.extinction = 0.7;
                moving.log_alpha = -15;
                input.push_back(moving);
                Acc::CouplingInput equilibrium{};
                equilibrium.density = 1;
                equilibrium.radiation[0] = 1;
                equilibrium.gas_energy = equilibrium.internal_energy = 1;
                equilibrium.absorption = equilibrium.extinction = 1e12;
                equilibrium.log_alpha = 3 * std::numeric_limits<double>::epsilon();
                input.push_back(equilibrium);
                compare_coupling(backend, input, 0.03 / c, c, eps, marshak);
                compare_coupling(backend, input, 0, c, eps, marshak);
            }
        }
    }
    std::vector<Acc::CouplingOutput> output(1);
    Acc::couple(backend, {}, 1, 1, 1, false, output);
    require(output.empty(), "empty coupling batch");
}

void validate_implicit(Acc::CouplingInput const& in, Acc::CouplingOutput const& out,
    double dt, double c, double eps, bool marshak) {
    const long double E = out.radiation[0];
    const long double chat = static_cast<long double>(c) * eps;
    long double Q2 = 0, vQ = 0, kinetic = 0, initial_kinetic = 0;
    long double velocity[3];
    for (int d = 0; d < 3; ++d) {
        velocity[d] = static_cast<long double>(out.momentum[d]) / in.density;
        Q2 += static_cast<long double>(out.radiation[d + 1]) * out.radiation[d + 1];
        vQ += velocity[d] * out.radiation[d + 1];
        kinetic += static_cast<long double>(out.momentum[d]) * out.momentum[d] / (2 * in.density);
        initial_kinetic += static_cast<long double>(in.momentum[d]) * in.momentum[d] / (2 * in.density);
        near(out.momentum[d] + static_cast<long double>(out.radiation[d + 1]) / chat,
            in.momentum[d] + static_cast<long double>(in.radiation[d + 1]) / chat,
            std::abs(in.momentum[d]) + std::abs(out.momentum[d]) +
                (std::abs(in.radiation[d + 1]) + std::abs(out.radiation[d + 1])) / chat,
            "implicit source RSLA momentum invariant");
    }
    require(E >= 0 && out.internal_energy >= 0 && std::sqrt(Q2) <= E * (1 + 128e-16L),
        "implicit source physical state");
    const long double energy_scale = std::abs(in.gas_energy) + std::abs(out.gas_energy) +
        (std::abs(in.radiation[0]) + E) / eps;
    near(out.gas_energy + E / eps, in.gas_energy + static_cast<long double>(in.radiation[0]) / eps,
        energy_scale, "implicit source RSLA energy invariant");
    near(out.gas_energy - out.internal_energy - kinetic,
        in.gas_energy - in.internal_energy - initial_kinetic,
        energy_scale, "implicit source dual-energy offset");

    // Independent Levermore Eddington tensor in conserved variables. This
    // does not reuse the implicit solver's H/beta form or its Jacobian.
    const long double f2 = E > 0 ? Q2 / (E * E) : 0;
    const long double eddington = (3 + 4 * f2) / (5 + 2 * std::sqrt(4 - 3 * f2));
    for (int i = 0; i < 3; ++i) {
        long double advective = E * velocity[i];
        for (int j = 0; j < 3; ++j) {
            const long double isotropic = i == j ? (1 - eddington) / 2 : 0;
            const long double directed = Q2 > 0 ? (3 * eddington - 1) *
                out.radiation[i + 1] * out.radiation[j + 1] / (2 * Q2) : 0;
            advective += E * (isotropic + directed) * velocity[j];
        }
        const long double advection = static_cast<long double>(dt) * eps * in.extinction * advective;
        const long double drag = static_cast<long double>(dt) * chat * in.extinction * out.radiation[i + 1];
        // The nonlinear solver bounds the residual after division by the
        // implicit damping denominator, using E as its flux-vector scale.
        // Include that natural scale here, particularly when one Q component
        // is tiny; a small component alone is not the solver's error contract.
        const long double damping_scale = (1 + static_cast<long double>(dt) * chat * in.extinction) * E;
        near(out.radiation[i + 1] - static_cast<long double>(in.radiation[i + 1]), advection - drag,
            std::abs(out.radiation[i + 1]) + std::abs(in.radiation[i + 1]) +
                std::abs(advection) + std::abs(drag) + damping_scale,
            "implicit source full flux residual", 8192);
    }
    const long double emission = marshak ? out.internal_energy : (out.internal_energy > 0 ?
        std::exp(static_cast<long double>(in.log_alpha) + 4 * std::log(static_cast<long double>(out.internal_energy))) : 0);
    const long double emitted = static_cast<long double>(dt) * chat * in.absorption * emission;
    const long double absorbed = static_cast<long double>(dt) * chat * in.absorption * E;
    const long double work = static_cast<long double>(dt) * eps *
        (2 * static_cast<long double>(in.absorption) - in.extinction) * vQ;
    near(E - in.radiation[0], emitted - absorbed + work,
        E + std::abs(in.radiation[0]) + std::abs(emitted) + std::abs(absorbed) + std::abs(work),
        "implicit source full energy residual", 8192);
}

void compare_implicit(Acc::Backend selected, const std::vector<Acc::CouplingInput>& input,
    double dt, double c, double eps, bool marshak) {
    std::vector<Acc::CouplingOutput> result;
    require(Acc::couple_implicit(selected, input, dt, c, eps, marshak, result),
        "implicit coupling batch must converge");
    require(result.size() == input.size(), "implicit coupling shape");
    std::vector<Acc::CouplingOutput> cpu;
    if (selected != Acc::Backend::cpu) {
        require(Acc::couple_implicit(Acc::Backend::cpu, input, dt, c, eps, marshak, cpu),
            "implicit CPU parity batch must converge");
    }
    for (size_t cell = 0; cell < input.size(); ++cell) {
        try {
            validate_implicit(input[cell], result[cell], dt, c, eps, marshak);
        } catch (const std::exception& error) {
            std::ostringstream context;
            context << "implicit cell=" << cell << " c=" << c << " eps=" << eps << " dt=" << dt
                    << " absorption=" << input[cell].absorption << " extinction=" << input[cell].extinction
                    << " marshak=" << marshak << ": " << error.what();
            throw std::runtime_error(context.str());
        }
        if (!cpu.empty()) {
            const auto& actual = result[cell];
            const auto& expected = cpu[cell];
            const long double energy_scale = input[cell].internal_energy + input[cell].radiation[0] / eps;
            for (int f = 0; f < 4; ++f) {
                near(actual.radiation[f], expected.radiation[f], expected.radiation[0],
                    "implicit device/CPU radiation parity", 8192);
            }
            for (int d = 0; d < 3; ++d) {
                near(actual.momentum[d], expected.momentum[d],
                    std::abs(expected.momentum[d]) + expected.radiation[0] / (eps * c),
                    "implicit device/CPU momentum parity", 8192);
            }
            near(actual.gas_energy, expected.gas_energy, energy_scale, "implicit device/CPU gas parity", 8192);
            near(actual.internal_energy, expected.internal_energy, energy_scale,
                "implicit device/CPU internal energy parity", 8192);
        }
    }
}

bool same_coupling(const std::vector<Acc::CouplingOutput>& a, const std::vector<Acc::CouplingOutput>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t cell = 0; cell < a.size(); ++cell) {
        for (int f = 0; f < 4; ++f) {
            if (a[cell].radiation[f] != b[cell].radiation[f]) {
                return false;
            }
        }
        for (int d = 0; d < 3; ++d) {
            if (a[cell].momentum[d] != b[cell].momentum[d]) {
                return false;
            }
        }
        if (a[cell].gas_energy != b[cell].gas_energy || a[cell].internal_energy != b[cell].internal_energy) {
            return false;
        }
    }
    return true;
}

void implicit_coupling(Acc::Backend selected) {
    // Mixed opacities and states in the same launch, including a partial final
    // CUDA block. Repeated smaller batches exercise cached workspace extents.
    const double opacities[] = {0, 1e-6, 0.2, 1, 1e2, 1e6, 1e10};
    std::vector<Acc::CouplingInput> input;
    for (size_t cell = 0; cell < 257; ++cell) {
        Acc::CouplingInput in{};
        in.radiation[0] = 0.1 + 0.002 * cell;
        in.radiation[1] = in.radiation[0] * 0.15 * std::sin(cell + 0.1);
        in.radiation[2] = in.radiation[0] * 0.1 * std::cos(cell + 0.2);
        in.radiation[3] = in.radiation[0] * 0.05 * std::sin(cell + 0.3);
        in.density = 100 + cell % 11;
        in.internal_energy = 4 + 0.005 * cell;
        in.gas_energy = in.internal_energy + 0.125;
        for (int d = 0; d < 3; ++d) {
            in.momentum[d] = 0.1 * std::sin(0.1 * cell + d);
            in.gas_energy += in.momentum[d] * in.momentum[d] / (2 * in.density);
        }
        in.extinction = opacities[cell % 7];
        in.absorption = in.extinction * (0.3 + 0.1 * (cell % 4));
        in.log_alpha = -8 + 4 * double(cell % 4);
        input.push_back(in);
    }
    Acc::CouplingInput vacuum{};
    vacuum.density = 1;
    input.push_back(vacuum);
    for (double c : {10.0, 2.99792458e10}) {
        for (double eps : {1.0, 0.03}) {
            for (bool marshak : {false, true}) {
                compare_implicit(selected, input, 0.03 / c, c, eps, marshak);
                compare_implicit(selected, {input.front(), input[17]}, 0, c, eps, marshak);
            }
        }
    }
    std::vector<Acc::CouplingOutput> result(2);
    for (size_t cell = 0; cell < result.size(); ++cell) {
        for (int f = 0; f < 4; ++f) {
            result[cell].radiation[f] = 10 * cell + f + 1;
        }
        for (int d = 0; d < 3; ++d) {
            result[cell].momentum[d] = -10 * double(cell) - d - 1;
        }
        result[cell].gas_energy = 17 + cell;
        result[cell].internal_energy = 23 + cell;
    }
    const auto sentinel = result;
    for (int failure = 0; failure < 5; ++failure) {
        auto bad = input;
        auto& invalid = bad[129];
        if (failure == 0) {
            invalid.density = 0;
        } else if (failure == 1) {
            invalid.absorption = -1;
        } else if (failure == 2) {
            invalid.radiation[0] = std::numeric_limits<double>::quiet_NaN();
        } else if (failure == 3) {
            invalid.radiation[1] = 2 * invalid.radiation[0];
            invalid.absorption = invalid.extinction = 0;
        } else {
            invalid.internal_energy = -1e30;
        }
        require(!Acc::couple_implicit(selected, bad, 0.003, 10, 0.03, false, result),
            "invalid cell must reject complete implicit batch");
        require(same_coupling(result, sentinel), "failed implicit batch preserves every output");
    }
    require(!Acc::couple_implicit(selected, input, -1, 10, 1, false, result), "invalid implicit dt rejected");
    require(!Acc::couple_implicit(selected, input, 1, 0, 1, false, result), "invalid implicit c rejected");
    require(!Acc::couple_implicit(selected, input, 1, 10, 0, false, result), "invalid implicit factor rejected");
    require(same_coupling(result, sentinel), "invalid global parameters preserve implicit output");
    require(Acc::couple_implicit(selected, {}, 1, 10, 1, false, result), "empty implicit batch succeeds");
    require(result.empty(), "empty implicit batch clears output");
}

void invalid_input(Acc::Backend backend) {
    Acc::Grid grid{8, 2, 1, 1, 1, 0.001, 0};
    const size_t count = cells(grid);
    std::vector<double> state(4 * count, 0), coordinates(3 * count, 0);
    for (size_t r = 0; r < count; ++r) {
        state[r] = 1;
    }
    std::vector<double> output{17, 23};
    const auto sentinel = output;
    auto invalid = state;
    invalid[0] = -1;
    throws([&] { Acc::compute_flux(backend, grid, invalid, coordinates, output); }, "negative energy must fail");
    require(output == sentinel, "failed flux preserves output");
    invalid = state;
    invalid[count] = 1.01;
    throws([&] { Acc::compute_flux(backend, grid, invalid, coordinates, output); }, "nonrealizable state must fail");
    require(output == sentinel, "failed realizability preserves output");
    invalid = state;
    invalid[0] = std::numeric_limits<double>::quiet_NaN();
    throws([&] { Acc::compute_flux(backend, grid, invalid, coordinates, output); }, "nonfinite state must fail");
    require(output == sentinel, "failed finite check preserves output");
    std::vector<double> flux;
    Acc::compute_flux(backend, grid, state, coordinates, flux);
    const size_t r = index(grid, grid.bw, grid.bw, grid.bw);
    flux[r + grid.n * grid.n] = 1e100;
    const auto original = state;
    throws([&] { Acc::advance(backend, grid, state, flux, output); }, "negative transport update must fail");
    require(state == original && output == sentinel, "failed advance is transactional");
    Acc::CouplingInput in{};
    in.radiation[0] = 1;
    in.density = 1;
    in.internal_energy = in.gas_energy = 1;
    std::vector<Acc::CouplingOutput> result(1);
    result[0].gas_energy = 17;
    for (int failure = 0; failure < 4; ++failure) {
        auto bad = in;
        if (failure == 0) {
            bad.density = 0;
        } else if (failure == 1) {
            bad.internal_energy = -1;
        } else if (failure == 2) {
            bad.radiation[1] = 2;
        } else {
            bad.radiation[1] = 0.5;
            // Nonrelativistic gas (v/c=0.01), but the explicit scattering
            // work drains E by 5 during dt=1. Reject this oversized step.
            bad.momentum[0] = 0.01;
            bad.extinction = 1000;
            bad.internal_energy = 1e-20;
        }
        throws([&] { Acc::couple(backend, {bad}, 1, 1, 1, false, result); }, "invalid coupling must fail");
        require(result.size() == 1 && result[0].gas_energy == 17, "failed coupling preserves output");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string name = argc > 1 ? argv[1] : "cpu";
        const auto backend = Acc::backend(name);
        Acc::initialize(backend);
        for (double c : {1.0, 7.0, 2.99792458e10}) {
            for (double eps : {1.0, 0.03}) {
                for (double rotation : {0.0, 0.025}) {
                    for (int mode = 0; mode < 4; ++mode) {
                        transport(backend, c, eps, rotation, mode);
                    }
                }
            }
        }
        coupling(backend);
        implicit_coupling(backend);
        invalid_input(backend);
        Acc::finalize();
        std::cout << name << " radiation accelerator: " << checks << " numerical checks passed\n";
        return 0;
    } catch (std::exception const& error) {
        Acc::finalize();
        std::cerr << "radiation accelerator parity: " << error.what() << '\n';
        return 1;
    }
}
