// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
// Standalone: c++ -std=c++20 -O2 -I. tests/radiation/test_fld_hydro.cpp -o /tmp/test_fld_hydro
#include "octotiger/radiation/fld_hydro.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace H = radiation::fld::hydro;
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
    const double h = 1. / n;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                cells.push_back({{{(i + .5) * h, (j + .5) * h, (k + .5) * h}}, h});
            }
        }
    }
    return MG::composite_mesh(std::move(cells), {{0, 0, 0}}, {{1, 1, 1}}, {{periodic, periodic, periodic}});
}

H::state state(double rho, H::vector velocity, double internal, double radiation) {
    H::state result;
    result.density = rho;
    for (int d = 0; d < 3; ++d) {
        result.momentum[d] = rho * velocity[d];
    }
    result.gas_energy = internal + H::kinetic_energy(result);
    result.radiation_energy = radiation;
    return result;
}

long double total(const MG::composite_mesh& mesh, const std::vector<H::state>& values) {
    long double value = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        value += (static_cast<long double>(values[i].gas_energy) + values[i].radiation_energy) * mesh.volume(i);
    }
    return value;
}

void closure_test() {
    H::closure pressure;
    const H::vector velocity{{3, 4, 0}};
    const double c = 100, B = 7;
    const double h = 1 - 4. / 3. * H::dot(velocity, velocity) / (c * c);
    require(H::make_closure(B / h, 1.e10, {{0, 0, 0}}, c, pressure), "flat thick closure");
    near(pressure.lambda, 1. / 3., 0, "thick limiter");
    near(pressure.thermal_factor(velocity, c), h, 2.e-16, "boost thermal coefficient sign");
    near(pressure.thermal_target(velocity, c), B, 2.e-15, "uniform moving LTE target");
    const auto flux = pressure.lab_flux(velocity);
    for (int d = 0; d < 3; ++d) {
        near(flux[d], 4. / 3. * B / h * velocity[d], 2.e-14, "moving LTE advected enthalpy");
    }
    // This checks the retained dynamic-diffusion cancellation independently of
    // the thermal_target helper. Omitting h would leave a large stiff source.
    const double chi = 1.e12;
    const double emission_source = c * chi * (B - pressure.energy);
    const double boost_source = chi / c * H::dot(velocity, pressure.enthalpy_flux(velocity));
    near((emission_source + boost_source) / (c * chi * B), 0, 2.e-16,
        "leading dynamic-diffusion source cancels boosted equilibrium");

    const H::vector gradient{{2, -3, 6}};
    require(H::make_closure(4, .03, gradient, c, pressure), "oblique tensor closure");
    near(pressure.eddington[0][0] + pressure.eddington[1][1] + pressure.eddington[2][2],
        1, 3.e-16, "tensor trace");
    require(std::abs(pressure.eddington[0][1]) > .01, "oblique tensor is not isotropic");
    const auto enthalpy = pressure.enthalpy_flux(velocity);
    const double direct = pressure.energy - 2 * H::dot(velocity, pressure.diffusive_flux) / (c * c)
        - H::dot(velocity, enthalpy) / (c * c);
    near(pressure.thermal_target(velocity, c), direct, 2.e-15, "full tensor thermal target");
    require(H::norm(pressure.diffusive_flux) <= c * pressure.energy * (1 + 1.e-14),
        "relative flux bound");
    // Full lab flux is derived; it is not capped independently after adding H.
    const auto lab = pressure.lab_flux(velocity);
    for (int d = 0; d < 3; ++d) {
        near(lab[d], pressure.diffusive_flux[d] + enthalpy[d], 0, "derived lab flux identity");
    }
    for (double chiP : {.01, .03, 7.}) {
        const double gas_work = H::dot(velocity, pressure.force());
        const double thermal = c * chiP * (pressure.thermal_target(velocity, c) - B);
        const double independent = c * chiP * (pressure.energy - B)
            + pressure.lambda * (2 * chiP / pressure.chi_rosseland - 1) * H::dot(velocity, gradient)
            - chiP / c * H::dot(velocity, enthalpy);
        near(gas_work + thermal, independent, 1.e-11,
            "paired work plus thermal target matches mixed-frame energy equation");
    }
}

void constant_state_test() {
    const auto mesh = uniform(4, true);
    const auto initial = state(1, {{.2, -.1, .15}}, 3, 2);
    std::vector<H::state> values(mesh.cells().size(), initial);
    std::vector<double> chi(values.size(), 10);
    const auto probe = H::hyperbolic_step(mesh, values, chi, 0, 100, 5. / 3., {});
    require(probe.success && probe.stable_dt > 0, "CFL probe");
    const auto next = H::hyperbolic_step(mesh, values, chi, .8 * probe.stable_dt, 100, 5. / 3., {});
    require(next.success, "uniform moving state accepted: " + next.message);
    for (size_t i = 0; i < values.size(); ++i) {
        near(next.state_values[i].density, initial.density, 1.e-15, "constant density");
        near(next.state_values[i].gas_energy, initial.gas_energy, 2.e-15, "constant gas energy");
        near(next.state_values[i].radiation_energy, initial.radiation_energy, 2.e-15, "constant radiation energy");
        for (int d = 0; d < 3; ++d) {
            near(next.state_values[i].momentum[d], initial.momentum[d], 1.e-15, "constant momentum");
        }
        near(next.radiation_work_by_cell[i], 0, 0, "constant state force free");
    }
    near(next.conservation_error, 0, 3.e-15, "constant state conservation");
}

void impulse_and_rejection_test() {
    const auto mesh = uniform(4, true);
    const double pi = std::acos(-1.0);
    std::vector<H::state> values;
    for (const auto& cell : mesh.cells()) {
        values.push_back(state(1, {{0, 0, 0}}, 3,
            2 + .8 * std::sin(2 * pi * cell.center[0])));
    }
    const std::vector<double> chi(values.size(), 5);
    const auto probe = H::hyperbolic_step(mesh, values, chi, 0, 100, 5. / 3., {});
    require(probe.success, "impulse CFL probe");
    const auto next = H::hyperbolic_step(mesh, values, chi, .5 * probe.stable_dt, 100, 5. / 3., {});
    require(next.success, "impulse step accepted: " + next.message);
    double work = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        near(H::internal_energy(next.state_values[i]), 3, 2.e-15,
            "radiation force changes kinetic energy without artificial gas heat");
        const double kinetic = H::kinetic_energy(next.state_values[i]);
        near(next.radiation_work_by_cell[i], -kinetic * mesh.volume(i), 1.e-18,
            "radiation work equals negative kinetic impulse");
        work += next.radiation_work_by_cell[i];
    }
    require(work < 0, "radiation accelerates initially stationary gas");
    near(static_cast<double>(total(mesh, next.state_values) - total(mesh, values)), 0, 2.e-15,
        "periodic combined energy conserved");
    const auto rejected = H::hyperbolic_step(mesh, values, chi, 2 * probe.stable_dt, 100, 5. / 3., {});
    require(!rejected.success && rejected.stable_dt == probe.stable_dt, "CFL rejection reports usable bound");
    for (size_t i = 0; i < values.size(); ++i) {
        near(rejected.state_values[i].gas_energy, values[i].gas_energy, 0, "rejection preserves total gas energy");
        near(rejected.state_values[i].radiation_energy, values[i].radiation_energy, 0, "rejection preserves radiation");
        near(rejected.radiation_work_by_cell[i], 0, 0, "rejection does not commit work ledger");
        near(rejected.gas_boundary_by_cell[i], 0, 0, "rejection does not commit boundary ledger");
    }
}

void boundary_test() {
    const auto mesh = uniform(4);
    std::vector<H::state> values(mesh.cells().size(), state(1, {{-.2, 0, 0}}, 3, 2));
    const std::vector<double> chi(values.size(), 5);
    const auto boundary = [](const MG::face& face, const H::state& interior, const H::closure& pressure) {
        if (face.axis == 0 && face.boundary_side < 0) {
            return H::reflecting_boundary(face, interior, pressure);
        }
        if (face.axis == 0) {
            return H::boundary_state{state(1.2, {{-.25, 0, 0}}, 3.5, 2.3), 6, {{0, 0, 0}}};
        }
        return H::outflow_boundary(interior, pressure);
    };
    const auto probe = H::hyperbolic_step(mesh, values, chi, 0, 100, 5. / 3., boundary);
    require(probe.success, "boundary CFL probe");
    const auto next = H::hyperbolic_step(mesh, values, chi, .7 * probe.stable_dt, 100, 5. / 3., boundary);
    require(next.success, "reflecting/inflow step accepted: " + next.message);
    long double boundary_total = 0, boundary_mass = 0, mass_change = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        boundary_total += next.gas_boundary_by_cell[i] + next.radiation_boundary_by_cell[i];
        boundary_mass += next.mass_boundary_by_cell[i];
        mass_change += (static_cast<long double>(next.state_values[i].density) - values[i].density) * mesh.volume(i);
        if (mesh.cells()[i].center[0] < .25) {
            near(next.gas_boundary_by_cell[i], 0, 0, "reflecting wall has zero gas-energy inflow");
            near(next.radiation_boundary_by_cell[i], 0, 0, "reflecting wall has zero radiation enthalpy inflow");
        }
    }
    require(boundary_total > 0 && boundary_mass > 0, "reservoir provides nonzero energy and mass ledger");
    near(static_cast<double>(total(mesh, next.state_values) - total(mesh, values) - boundary_total),
        0, 2.e-15, "open-boundary combined energy accounting");
    near(static_cast<double>(mass_change - boundary_mass), 0, 2.e-16, "open-boundary mass accounting");
}

void radiation_sound_test() {
    const auto mesh = uniform(4, true);
    const std::vector<double> chi(mesh.cells().size(), 1000);
    const std::vector<H::state> gas(mesh.cells().size(), state(1, {{0, 0, 0}}, .001, 0));
    const std::vector<H::state> bright(mesh.cells().size(), state(1, {{0, 0, 0}}, .001, 1000));
    const auto gas_dt = H::hyperbolic_step(mesh, gas, chi, 0, 100, 5. / 3., {});
    const auto rad_dt = H::hyperbolic_step(mesh, bright, chi, 0, 100, 5. / 3., {});
    require(gas_dt.success && rad_dt.success, "radiation sound CFL probes");
    require(rad_dt.stable_dt < .01 * gas_dt.stable_dt,
        "radiation pressure enters timestep even for cold gas");
}

void compression_test() {
    const auto mesh = uniform(4, true);
    const double pi = std::acos(-1.0), amplitude = 1.e-5;
    std::vector<H::state> values;
    for (const auto& cell : mesh.cells()) {
        values.push_back(state(1, {{amplitude * std::sin(2 * pi * cell.center[0]), 0, 0}}, 3, 2));
    }
    const std::vector<double> chi(values.size(), 1.e10);
    const double dt = 1.e-3;
    const auto next = H::hyperbolic_step(mesh, values, chi, dt, 100, 5. / 3., {});
    require(next.success, "isotropic radiation compression step");
    for (size_t i = 0; i < values.size(); ++i) {
        const double divergence = amplitude * std::sin(2 * pi / 4) * 4 *
            std::cos(2 * pi * mesh.cells()[i].center[0]);
        // Uniform E has no diffusion force/work, so the enthalpy flux alone
        // gives dE/dt=-(4/3) E div(v), the radiation adiabatic law.
        near(next.state_values[i].radiation_energy, 2 - dt * (4. / 3.) * 2 * divergence,
            8.e-16, "radiation pressure compression counted once");
    }
}

void implicit_moving_test() {
    namespace F = radiation::fld;
    const auto mesh = uniform(4, true);
    const std::vector<F::boundary_condition> boundary(6);
    F::settings options;
    options.nonlinear_tolerance = 2.e-12;
    options.linear_tolerance = 1.e-12;
    const H::vector v{{.3, -.2, .1}};
    const double c = 10, h = 1 - 4. / 3. * H::dot(v, v) / (c * c);
    std::vector<F::cell> initial(mesh.cells().size(), {16 / h, 2, 1.e10, 1.e10, 0, false, v});
    const auto equilibrium = F::solve(mesh, initial, 1000, c, boundary, options);
    require(equilibrium.success, "stiff moving LTE solve: " + equilibrium.message);
    for (size_t i = 0; i < initial.size(); ++i) {
        near(equilibrium.state[i].energy, initial[i].energy, 2.e-12, "stiff boosted radiation remains in LTE");
        near(equilibrium.state[i].internal_energy, initial[i].internal_energy, 2.e-13,
            "stiff boosted gas remains in LTE");
    }

    const double pi = std::acos(-1.0);
    std::vector<double> energy(initial.size());
    for (size_t i = 0; i < initial.size(); ++i) {
        energy[i] = 2 + .4 * std::sin(2 * pi * mesh.cells()[i].center[0]);
    }
    for (size_t i = 0; i < initial.size(); ++i) {
        H::closure pressure;
        require(H::make_closure(energy[i], 3, mesh.cell_gradient(i, energy), c, pressure),
            "nonuniform moving input closure");
        initial[i] = {energy[i], std::pow(pressure.thermal_target(v, c), .25), 3, 2, 0, false, v};
    }
    const double dt = .003;
    const auto next = F::solve(mesh, initial, dt, c, boundary, options);
    require(next.success, "nonuniform frozen-velocity solve: " + next.message);
    for (size_t i = 0; i < initial.size(); ++i) {
        energy[i] = next.state[i].energy;
    }
    long double conservation = 0;
    for (size_t i = 0; i < initial.size(); ++i) {
        H::closure pressure;
        require(H::make_closure(energy[i], 3, mesh.cell_gradient(i, energy), c, pressure),
            "nonuniform accepted closure");
        const double target = pressure.thermal_target(v, c);
        const double u = next.state[i].internal_energy;
        near(u - initial[i].internal_energy, dt * c * 2 * (target - std::pow(u, 4)),
            3.e-13, "accepted thermal equation uses refreshed flux and velocity target");
        conservation += (static_cast<long double>(energy[i]) - initial[i].energy +
            static_cast<long double>(u) - initial[i].internal_energy) * mesh.volume(i);
    }
    near(static_cast<double>(conservation), 0, 2.e-11, "moving implicit energy conservation");
}

void radiation_acoustic_wave_test() {
    // Linear, optically thick, uncoupled-thermal radiation acoustics:
    // c_s^2=(gamma*p+4*E/9)/rho. This exercises transport plus force/work
    // together; a gas-only update would propagate at the wrong speed.
    const double gamma = 5. / 3., pi = std::acos(-1.0), amplitude = 1.e-6;
    const double exact_speed = std::sqrt(gamma + 4. / 3.);
    const double end = .1;
    std::vector<double> errors;
    double measured_speed = 0;
    for (int n : {8, 16}) {
        const auto mesh = uniform(n, true);
        const double cell_average = std::sin(pi / n) / (pi / n);
        std::vector<H::state> values;
        for (const auto& cell : mesh.cells()) {
            const double perturbation = amplitude * cell_average * std::sin(2 * pi * cell.center[0]);
            values.push_back(state(1 + perturbation, {{exact_speed * perturbation, 0, 0}},
                (1 + gamma * perturbation) / (gamma - 1), 3 + 4 * perturbation));
        }
        const std::vector<double> chi(values.size(), 1.e10);
        double time = 0;
        while (time < end) {
            const auto probe = H::hyperbolic_step(mesh, values, chi, 0, 100, gamma, {});
            require(probe.success, "radiation acoustic CFL probe");
            const double dt = std::min(probe.stable_dt, end - time);
            auto next = H::hyperbolic_step(mesh, values, chi, dt, 100, gamma, {});
            require(next.success, "radiation acoustic step: " + next.message);
            values = std::move(next.state_values);
            time += dt;
        }
        double error = 0, sine = 0, cosine = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            const double x = mesh.cells()[i].center[0];
            const double exact = 1 + amplitude * cell_average * std::sin(2 * pi * (x - exact_speed * end));
            error += std::abs(values[i].density - exact) * mesh.volume(i);
            sine += (values[i].density - 1) * std::sin(2 * pi * x) * mesh.volume(i);
            cosine += (values[i].density - 1) * std::cos(2 * pi * x) * mesh.volume(i);
        }
        errors.push_back(error);
        measured_speed = -std::atan2(cosine, sine) / (2 * pi * end);
    }
    require(errors[0] / errors[1] > 1.4 && errors[0] / errors[1] < 2.4,
        "radiation acoustic profile converges at first order");
    near(measured_speed, exact_speed, .05 * exact_speed,
        "radiation acoustic speed includes coupled radiation pressure");
}
}

int main() {
    try {
        closure_test();
        constant_state_test();
        impulse_and_rejection_test();
        boundary_test();
        radiation_sound_test();
        compression_test();
        implicit_moving_test();
        radiation_acoustic_wave_test();
        std::cout << "FLD hydro tests passed (" << checks << " checks)\n";
    } catch (const std::exception& error) {
        std::cerr << "FLD hydro test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
