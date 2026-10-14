// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
// Standalone: c++ -std=c++17 -O2 -I. tests/radiation/test_fld.cpp -o /tmp/test_fld
#include "octotiger/radiation/fld.hpp"
#include "octotiger/radiation/physical_opacity.hpp"

#include <cfenv>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace FLD = radiation::fld;
namespace MG = octotiger::elliptic;
size_t checks = 0;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}
void near(double actual, double expected, double tolerance, const std::string& message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
        message + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}
MG::composite_mesh uniform(int n, bool periodic = false) {
    std::vector<MG::cell> cells;
    const double h = 1.0 / n;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                cells.push_back({{{(i + .5) * h, (j + .5) * h, (k + .5) * h}}, h});
            }
        }
    }
    return MG::composite_mesh(std::move(cells), {{0, 0, 0}}, {{1, 1, 1}}, {{periodic, periodic, periodic}});
}
std::vector<FLD::boundary_condition> reflecting() {
    return std::vector<FLD::boundary_condition>(6, {0, 1, 0});
}
void limiter_tests() {
    near(FLD::limiter(0), 1.0 / 3, 1.e-15, "LP thick limit");
    near(FLD::limiter_flux_factor(std::numeric_limits<double>::infinity()), 1, 0, "LP streaming limit");
    for (double R : {0.0, 1.e-250, 1.e-10, 1.0, 10.0, 1.e250}) {
        require(FLD::limiter(R) >= 0, "nonnegative LP limiter");
        require(FLD::limiter_flux_factor(R) >= 0 && FLD::limiter_flux_factor(R) <= 1,
            "LP flux factor is bounded");
    }
    near(FLD::diffusion_coefficient(2, 100, 0, 30), .1, 1.e-15, "physical c thick diffusivity");
    near(FLD::diffusion_coefficient(2, 0, 3, 30) * 3, 60, 1.e-12, "zero-opacity streaming flux");
    near(FLD::diffusion_coefficient(0, 0, 0, 30), 0, 0, "empty vacuum convention");
    near(FLD::diffusion_coefficient(2, 0, 0, 30), 0, 0, "flat vacuum convention");
    for (double E : {1.e-250, 1.e-20, 1.0, 1.e200}) {
        for (double chi : {0.0, 1.e-100, 1.0, 1.e100}) {
            const double gradient = E * 2;
            const double D = FLD::diffusion_coefficient(E, chi, gradient, 10);
            require(std::isfinite(D) && D >= 0, "extreme coefficient finite");
            require(static_cast<long double>(D) * gradient <= 10.0L * E * (1 + 1.e-14),
                "extreme coefficient flux bound");
        }
    }
}
void opacity_unit_tests() {
    const double rho_cgs = 3.e-7, T = 2.e5, hydrogen = .7, metals = .02;
    const auto cgs = radiation::fitted_extinction(rho_cgs, T, hydrogen, metals, 1, 1);
    for (const auto& units : {std::array<double, 2>{{2.e33, 7.e10}},
            std::array<double, 2>{{1.e20, 1.e8}}}) {
        const double mass = units[0], length = units[1];
        const double density = rho_cgs * length * length * length / mass;
        const auto converted = radiation::fitted_extinction(density, T, hydrogen, metals, mass, length);
        near(converted[0] / length, cgs[0], 2.e-14 * cgs[0], "Rosseland CGS equivalence");
        near(converted[1] / length, cgs[1], 2.e-14 * cgs[1], "Planck CGS equivalence");
    }
    const double absorption = 4.e25 * (1 + hydrogen) * rho_cgs * std::pow(T, -3.5);
    const double scattering = (1 + hydrogen) * .2 * T * T /
        ((T * T + 2.7e11 * rho_cgs) * (1 + std::pow(T / 4.5e8, .86)));
    near(cgs[0], rho_cgs * (absorption * (metals + .001) + scattering),
        2.e-14 * cgs[0], "independent Rosseland inverse-length expression");
    near(cgs[1], rho_cgs * 30.262 * absorption * (metals + .0001),
        2.e-14 * cgs[1], "independent Planck inverse-length expression");
}
void backward_euler_order_test() {
    const int n = 4;
    const auto mesh = uniform(n, true);
    const double c = 3, chi_R = 100, chi_P = .3, final_time = 1, amplitude = 1.e-3;
    const double pi = std::acos(-1.0);
    const double diffusion_rate = c / (3 * chi_R) * 4 * n * n * std::pow(std::sin(pi / n), 2);
    const double absorption_rate = c * chi_P;
    const double root = std::sqrt(diffusion_rate * diffusion_rate / 4 + absorption_rate * absorption_rate);
    const double half_trace = -(diffusion_rate + 2 * absorption_rate) / 2;
    const double weight = .5 * (1 + (absorption_rate - diffusion_rate / 2) / root);
    const double exact = amplitude * (weight * std::exp((half_trace + root) * final_time) +
        (1 - weight) * std::exp((half_trace - root) * final_time));
    std::vector<double> errors;
    for (int steps : {4, 8, 16}) {
        std::vector<FLD::cell> state(mesh.cells().size());
        for (size_t i = 0; i < state.size(); ++i) {
            const double value = 10 + amplitude * std::cos(2 * pi * mesh.cells()[i].center[0]);
            state[i] = {value, value, chi_R, chi_P, 0, true};
        }
        FLD::settings options;
        options.nonlinear_tolerance = 1.e-13;
        options.linear_tolerance = 1.e-12;
        for (int step = 0; step < steps; ++step) {
            auto next = FLD::solve(mesh, state, final_time / steps, c, reflecting(), options);
            require(next.success, "BE temporal convergence solve: " + next.message);
            state = std::move(next.state);
        }
        double measured = 0, normalization = 0;
        for (size_t i = 0; i < state.size(); ++i) {
            const double mode = std::cos(2 * pi * mesh.cells()[i].center[0]);
            measured += (state[i].energy - 10) * mode;
            normalization += mode * mode;
        }
        errors.push_back(std::abs(measured / normalization - exact));
    }
    require(errors[0] / errors[1] > 1.8 && errors[0] / errors[1] < 2.2,
        "BE first-order coupled temporal convergence, coarse pair");
    require(errors[1] / errors[2] > 1.8 && errors[1] / errors[2] < 2.2,
        "BE first-order coupled temporal convergence, fine pair");
}
void homogeneous_tests() {
    const auto mesh = uniform(2);
    std::vector<FLD::cell> state(mesh.cells().size());
    for (auto& value : state) {
        value = {1, 2, 10, 3, 0, true};
    }
    const double dt = 2, c = 5;
    const auto linear = FLD::solve(mesh, state, dt, c, reflecting());
    require(linear.success, "linear-emission source solve: " + linear.message);
    const double expected_E = (1 + dt * c * 3 * 3) / (1 + 2 * dt * c * 3);
    for (const auto& value : linear.state) {
        near(value.energy, expected_E, 2.e-13, "analytic Marshak source energy");
        near(value.internal_energy + value.energy, 3, 2.e-13, "physical total conserved");
    }
    near(linear.conservation_error, 0, 2.e-13, "homogeneous conservation budget");
    near(linear.boundary_energy, 0, 0, "reflecting boundary budget");
    near(linear.source_energy, expected_E - 1, 2.e-13, "source budget is physical radiation gain");
    for (auto& value : state) {
        value = {0, 2, 1.e5, 1.e100, std::log(3.0), false};
    }
    const auto stiff = FLD::solve(mesh, state, 1.e100, 10, reflecting());
    require(stiff.success, "stiff nonlinear source solve: " + stiff.message);
    for (const auto& value : stiff.state) {
        near(value.energy + value.internal_energy, 2, 5.e-13, "stiff gas+radiation conservation");
        near(value.energy, 3 * std::pow(value.internal_energy, 4), 5.e-13, "stiff quartic LTE");
        require(value.energy >= 0 && value.internal_energy >= 0, "stiff positivity");
    }
    const std::vector<FLD::cell> empty(state.size(), {0, 0, 1, 1.e200, 0, false});
    const auto empty_result = FLD::solve(mesh, empty, 1.e200, 10, reflecting());
    require(empty_result.success, "infinitely stiff empty equilibrium remains admissible");
    near(empty_result.source_energy, 0, 0, "empty equilibrium has zero source budget");
    const auto equilibrium = FLD::solve(mesh, stiff.state, 1.e100, 10, reflecting());
    require(equilibrium.success, "LTE fixed point accepted");
    for (size_t i = 0; i < state.size(); ++i) {
        near(equilibrium.state[i].energy, stiff.state[i].energy, 2.e-13, "LTE fixed point radiation");
        near(equilibrium.state[i].internal_energy, stiff.state[i].internal_energy, 2.e-13, "LTE fixed point gas");
    }
}
void diffusion_and_heat_capacity_tests() {
    const int n = 4;
    const auto mesh = uniform(n, true);
    std::vector<FLD::cell> initial(mesh.cells().size());
    const double pi = std::acos(-1.0), epsilon = 1.e-5, chi_R = 1000, c = 3, dt = 100;
    const double base_u = 2, base_E = 16;
    for (size_t i = 0; i < initial.size(); ++i) {
        const double mode = std::cos(2 * pi * mesh.cells()[i].center[0]);
        const double E = base_E + epsilon * mode;
        initial[i] = {E, std::pow(E, .25), chi_R, 1.e12, 0, false};
    }
    FLD::settings options;
    options.nonlinear_tolerance = 1.e-13;
    options.linear_tolerance = 1.e-12;
    const auto result = FLD::solve(mesh, initial, dt, c, reflecting(), options);
    require(result.success, "coupled stiff diffusion solve: " + result.message);
    double amplitude = 0, normalization = 0;
    for (size_t i = 0; i < initial.size(); ++i) {
        const double mode = std::cos(2 * pi * mesh.cells()[i].center[0]);
        amplitude += (result.state[i].energy - base_E) * mode;
        normalization += mode * mode;
        near(result.state[i].energy, std::pow(result.state[i].internal_energy, 4), 1.e-10,
            "stiff diffusion remains LTE");
        require(FLD::detail::norm(result.flux[i]) <= c * result.state[i].energy * (1 + 1.e-13),
            "multidimensional cell diagnostic bound");
    }
    amplitude /= normalization;
    const double eigenvalue = 4 * n * n * std::pow(std::sin(pi / n), 2);
    const double heat_capacity = 1 + 1 / (4 * std::pow(base_u, 3));
    const double expected = epsilon / (1 + dt * c / (3 * chi_R) * eigenvalue / heat_capacity);
    near(amplitude, expected, 5.e-10, "stiff diffusion includes BOTH heat capacities");
    near(result.conservation_error, 0, 1.e-10, "periodic diffusion total energy conservation");
}
void boundary_tests() {
    const auto mesh = uniform(4);
    std::vector<FLD::cell> initial(mesh.cells().size());
    for (auto& value : initial) {
        value = {0, 0, 10, 10, 0, true};
    }
    const double c = 3.e10, dt = 1.e-10;
    auto bc = reflecting();
    bc[0] = {c / 2, 1, 2}; // Marshak incoming radiation flux is one.
    bc[1] = {c / 2, 1, 0}; // Vacuum Robin face on the opposite side.
    FLD::settings options;
    options.max_iterations = 100;
    const auto injected = FLD::solve(mesh, initial, dt, c, bc, options);
    require(injected.success, "dark Marshak boundary injection: " + injected.message + " iterations=" + std::to_string(injected.iterations) + " residual=" + std::to_string(injected.nonlinear_residual));
    require(injected.boundary_energy > 0, "incoming Marshak bath heats a dark domain");
    near(injected.conservation_error, 0, 1.e-19, "Marshak boundary energy balance");
    for (const auto& value : injected.state) {
        require(value.energy >= 0 && value.internal_energy >= 0, "dark front remains nonnegative");
    }
    const auto refreshed = FLD::refresh(mesh, injected.state, c, bc);
    require(refreshed.success, "Marshak flux diagnostic refresh");
    std::vector<double> D;
    std::vector<std::array<double, 3>> cell_flux;
    require(FLD::detail::coefficients(mesh, injected.state, c, bc, D, cell_flux),
        "boundary coefficients recomputed");
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        const auto& face = mesh.faces()[f];
        if (face.boundary_side != 0 && face.axis == 0 && D[f] > 0) {
            const auto& boundary = bc[face.boundary_side < 0 ? 0 : 1];
            const size_t i = face.lower == MG::boundary_index ? face.upper : face.lower;
            const double flux = refreshed.face_flux[f];
            const double Eb = (boundary.value + boundary.gamma * flux) / boundary.alpha;
            require(Eb >= 0, "nonnegative Robin boundary trace");
            near(injected.state[i].energy - Eb, flux * face.distance / D[f], 1.e-23,
                "Robin trace and numerical gradient agree");
            require(std::abs(flux) <= c * Eb * (1 + 1.e-12), "Robin face streaming bound");
        }
    }
}
void amr_cold_front_tests() {
    const int n = 4;
    const double h = 1.0 / n;
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<MG::cell> cells;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    const std::array<int, 3> index = {{i, j, k}};
                    const int parts = index[axis] >= n / 2 ? 2 : 1;
                    for (int a = 0; a < parts; ++a) {
                        for (int b = 0; b < parts; ++b) {
                            for (int d = 0; d < parts; ++d) {
                                cells.push_back({{{h * (i + (a + .5) / parts),
                                    h * (j + (b + .5) / parts), h * (k + (d + .5) / parts)}}, h / parts});
                            }
                        }
                    }
                }
            }
        }
        const MG::composite_mesh mesh(std::move(cells), {{0, 0, 0}}, {{1, 1, 1}});
        std::vector<FLD::cell> state(mesh.cells().size());
        for (auto& value : state) {
            value = {0, 0, 10, 10, 0, true};
        }
        auto bc = reflecting();
        bc[0] = {1.5e10, 1, 2};
        const auto next = FLD::solve(mesh, state, 1.e-10, 3.e10, bc);
        require(next.success, "AMR dark Marshak orientation " + std::to_string(axis) + ": " + next.message);
        near(next.conservation_error, 0, 1.e-20, "composite cold-front energy conservation");
        for (const auto& value : next.state) {
            require(value.energy >= 0 && value.internal_energy >= 0, "AMR cold-front positivity");
        }
        for (size_t f = 0; f < mesh.faces().size(); ++f) {
            const auto& face = mesh.faces()[f];
            if (face.boundary_side == 0) {
                const double Eface = std::max(next.state[face.lower].energy, next.state[face.upper].energy);
                require(std::abs(next.face_flux[f]) <= 3.e10 * Eface * (1 + 1.e-12),
                    "composite shared subface flux bound");
            }
        }
    }
}
void runtime_marshak_cold_tail_test() {
    // Match the application's actual initialization and leaf-packet ordering:
    // radiation starts at zero but material has a 1e-20 thermal seed. Its
    // homogeneous source gives a positive radiation tail far below Krylov
    // norm accuracy; that tail must not set the whole nonlinear step length.
    const double h = .125;
    std::vector<MG::cell> cells;
    for (int child = 0; child < 8; ++child) {
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                for (int k = 0; k < 8; ++k) {
                    cells.push_back({{{-1 + ((child & 1) * 8 + i + .5) * h,
                        -1 + (((child >> 1) & 1) * 8 + j + .5) * h,
                        -1 + (((child >> 2) & 1) * 8 + k + .5) * h}}, h});
                }
            }
        }
    }
    const MG::composite_mesh mesh(std::move(cells), {{-1, -1, -1}}, {{1, 1, 1}});
    std::vector<FLD::cell> state(mesh.cells().size(), {0, 1.e-20, 100, 100, 0, true});
    auto bc = reflecting();
    bc[0] = {.5, 1, 2};
    bc[1] = {.5, 1, 0};
    const auto callback = [](size_t, double, double& rosseland, double& planck) {
        rosseland = planck = 100;
        return true;
    };
    for (int step = 0; step < 2; ++step) {
        const auto next = FLD::solve(mesh, state, .125, 1, bc, {}, callback);
        require(next.success, "application-sized Marshak cold tail: " + next.message);
        require(next.nonlinear_residual <= 1.e-10, "cold-tail update satisfies full coupled residual");
        near(next.conservation_error, 0, 1.e-10, "cold-tail direction safeguard conserves energy");
        for (const auto& value : next.state) {
            require(value.energy >= 0 && value.internal_energy >= 0, "cold-tail iterate remains admissible");
        }
        state = next.state;
    }
}
void thin_boundary_tests() {
    const auto mesh = uniform(2);
    auto bc = reflecting();
    bc[0] = {5, 1, 2};
    for (double chi : {0.0, 1.e-30, 1.e30}) {
        std::vector<FLD::cell> state(mesh.cells().size(), {0, 0, chi, 0, 0, true});
        const auto refreshed = FLD::refresh(mesh, state, 10, bc);
        require(refreshed.success, "thin/thick empty incoming Robin constitutive solve");
        for (size_t f = 0; f < mesh.faces().size(); ++f) {
            if (mesh.faces()[f].axis == 0 && mesh.faces()[f].boundary_side == -1) {
                require(std::isfinite(refreshed.face_flux[f]) && refreshed.face_flux[f] < 0,
                    "finite incoming Robin flux over sixty opacity decades");
            }
        }
    }
}
void diagnostic_and_failure_tests() {
    const auto mesh = uniform(3);
    std::vector<FLD::cell> initial(mesh.cells().size());
    for (size_t i = 0; i < initial.size(); ++i) {
        const auto& x = mesh.cells()[i].center;
        initial[i] = {1.e-12 + x[0] + 2 * x[1] + 3 * x[2], 1, 1.e-50, 0, 0, true};
    }
    const auto diagnostics = FLD::refresh(mesh, initial, 10, reflecting());
    require(diagnostics.success, "diagnostic refresh: " + diagnostics.message);
    for (size_t i = 0; i < initial.size(); ++i) {
        require(FLD::detail::norm(diagnostics.flux[i]) <= 10 * initial[i].energy * (1 + 1.e-13),
            "full vector free-streaming bound");
        near(diagnostics.state[i].energy, initial[i].energy, 0, "refresh preserves radiation energy");
        near(diagnostics.state[i].internal_energy, initial[i].internal_energy, 0, "refresh preserves material energy");
    }
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        const auto& face = mesh.faces()[f];
        if (face.boundary_side == 0) {
            const double E_face = std::max(initial[face.lower].energy, initial[face.upper].energy);
            require(std::abs(diagnostics.face_flux[f]) <= 10 * E_face * (1 + 1.e-13),
                "shared face flux bound");
        }
    }
    size_t callback_calls = 0;
    const auto callback = [&callback_calls](size_t, double u, double& chi_R, double& chi_P) {
        ++callback_calls;
        chi_R = 100 + u;
        chi_P = .01 * (1 + u);
        return true;
    };
    const auto varying = FLD::solve(mesh, initial, .001, 10, reflecting(), {}, callback);
    require(varying.success, "temperature-dependent opacity refresh: " + varying.message);
    require(callback_calls > 2 * initial.size(), "opacity refreshed inside nonlinear iterations");
    for (const auto& value : varying.state) {
        near(value.chi_rosseland, 100 + value.internal_energy, 1.e-13, "accepted Rosseland temperature");
        near(value.chi_planck, .01 * (1 + value.internal_energy), 1.e-13, "accepted absorption temperature");
    }
    FLD::settings too_short;
    too_short.max_iterations = 1;
    too_short.nonlinear_tolerance = 1.e-15;
    const auto rejected = FLD::solve(mesh, initial, 1, 10, reflecting(), too_short);
    require(!rejected.success, "forced nonconvergence rejected");
    near(rejected.boundary_energy, 0, 0, "rejected boundary budget discarded");
    near(rejected.source_energy, 0, 0, "rejected source budget discarded");
    for (size_t i = 0; i < initial.size(); ++i) {
        near(rejected.state[i].energy, initial[i].energy, 0, "rejected state transactional");
    }
    auto invalid = initial;
    invalid[0].energy = -1;
    require(!FLD::solve(mesh, invalid, 1, 10, reflecting()).success, "negative input rejected, never clipped");
}
} // namespace

int main() {
#if defined(__GLIBC__)
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif
    try {
        limiter_tests();
        opacity_unit_tests();
        backward_euler_order_test();
        homogeneous_tests();
        diffusion_and_heat_capacity_tests();
        boundary_tests();
        amr_cold_front_tests();
        thin_boundary_tests();
        runtime_marshak_cold_tail_test();
        diagnostic_and_failure_tests();
        std::cout << "FLD checks passed: " << checks << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FLD test failure: " << error.what() << '\n';
        return 1;
    }
}
