// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "octotiger/elliptic/composite.hpp"

namespace {

using namespace octotiger::elliptic;
constexpr double pi = 3.1415926535897932384626433832795;

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<cell> make_cells(int n, int refinement_axis = -1) {
    std::vector<cell> result;
    double const h = 1.0 / n;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                std::array<int, 3> const index = {{i, j, k}};
                int const parts = refinement_axis >= 0 && index[refinement_axis] >= n / 2 ? 2 : 1;
                for (int a = 0; a < parts; ++a) {
                    for (int b = 0; b < parts; ++b) {
                        for (int c = 0; c < parts; ++c) {
                            result.push_back(
                                {{{h * (i + (a + 0.5) / parts), h * (j + (b + 0.5) / parts),
                                     h * (k + (c + 0.5) / parts)}},
                                    h / parts});
                        }
                    }
                }
            }
        }
    }
    return result;
}

composite_mesh make_mesh(
    int n, int refinement_axis = -1, std::array<bool, 3> periodic = {{false, false, false}}) {
    return composite_mesh(make_cells(n, refinement_axis), {{0, 0, 0}}, {{1, 1, 1}}, periodic);
}

double error(composite_mesh const& mesh, std::vector<double> const& numerical,
    std::vector<double> const& exact) {
    double result = 0.0;
    for (size_t i = 0; i < exact.size(); ++i) {
        double const delta = numerical[i] - exact[i];
        result += mesh.volume(i) * delta * delta;
    }
    return std::sqrt(result);
}

double affine(std::array<double, 3> const& position) {
    return 2.0 + 0.3 * position[0] - 0.2 * position[1] + 0.1 * position[2];
}

double sine_field(std::array<double, 3> const& position) {
    return std::sin(pi * position[0]) * std::sin(pi * position[1]) * std::sin(pi * position[2]);
}

void affine_and_conservation(int refinement_axis) {
    auto const mesh = make_mesh(4, refinement_axis);
    std::array<double, 3> const gradient = {{0.3, -0.2, 0.1}};
    std::vector<double> exact(mesh.cells().size());
    for (size_t i = 0; i < exact.size(); ++i) {
        exact[i] = affine(mesh.cells()[i].center);
    }
    std::vector<double> beta(mesh.faces().size(), 0.7);
    std::vector<boundary_condition> bc(mesh.faces().size());
    size_t split_faces = 0;
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        face const& item = mesh.faces()[f];
        if (item.boundary_side != 0) {
            // Cycle all three boundary families through the six orientations.
            int const kind = (2 * item.axis + (item.boundary_side > 0 ? 1 : 0)) % 3;
            bc[f].alpha = kind == 1 ? 0.0 : 1.0;
            bc[f].gamma = kind == 0 ? 0.0 : 0.4;
            bc[f].value = bc[f].alpha * affine(item.center) +
                bc[f].gamma * beta[f] * item.boundary_side * gradient[item.axis];
        } else {
            auto const reconstructed = mesh.face_gradient(f, exact);
            for (int d = 0; d < 3; ++d) {
                require(std::abs(reconstructed[d] - gradient[d]) < 1.0e-12,
                    "Coarse/fine face gradient did not preserve a linear field");
            }
            require(std::abs(mesh.face_value(f, exact) - affine(item.center)) < 1.0e-12,
                "Coarse/fine face value did not preserve a linear field");
            if (mesh.cells()[item.lower].width != mesh.cells()[item.upper].width) {
                ++split_faces;
            }
        }
    }
    if (refinement_axis >= 0) {
        require(split_faces > 0, "Composite test has no refinement interfaces");
    }
    std::vector<double> const reaction(exact.size(), 1.5);
    scalar_operator op(mesh, reaction, beta, bc);
    auto applied = op.apply(exact);
    std::vector<double> rhs(exact.size());
    for (size_t i = 0; i < exact.size(); ++i) {
        rhs[i] = 1.5 * exact[i];
        require(std::abs(applied[i] - op.boundary_rhs()[i] - rhs[i]) < 2.0e-11,
            "Affine manufactured operator is inconsistent");
    }
    auto fluxes = op.face_fluxes(exact);
    double volume_balance = 0.0;
    double boundary_balance = 0.0;
    for (size_t i = 0; i < exact.size(); ++i) {
        volume_balance += mesh.volume(i) * (applied[i] - op.boundary_rhs()[i] - rhs[i]);
    }
    for (size_t f = 0; f < fluxes.size(); ++f) {
        if (mesh.faces()[f].boundary_side != 0) {
            boundary_balance += mesh.faces()[f].area * fluxes[f];
        }
    }
    require(std::abs(volume_balance - boundary_balance) < 1.0e-12,
        "Composite flux divergence does not cancel internal subfaces");
    std::vector<double> x(exact.size(), 0.0);
    auto const report = solve(op, rhs, x);
    require(report.converged, "Affine mixed-boundary multigrid solve did not converge");
    require(error(mesh, x, exact) < 2.0e-9, "Affine multigrid solution is inaccurate");
    require(report.levels >= 2, "Solver did not build virtual coarse grids");
    std::cout << "affine axis=" << refinement_axis << " cells=" << x.size()
              << " iterations=" << report.iterations << " levels=" << report.levels
              << " residual=" << report.relative_residual << '\n';
}

double manufactured_error(int n, int refinement_axis) {
    auto const mesh = make_mesh(n, refinement_axis);
    std::vector<double> exact(mesh.cells().size()), rhs(exact.size());
    double const diffusion = 0.7;
    for (size_t i = 0; i < exact.size(); ++i) {
        exact[i] = sine_field(mesh.cells()[i].center);
        rhs[i] = (1.0 + 3.0 * diffusion * pi * pi) * exact[i];
    }
    scalar_operator op(mesh, std::vector<double>(exact.size(), 1.0),
        std::vector<double>(mesh.faces().size(), diffusion),
        std::vector<boundary_condition>(6, {1.0, 0.0, 0.0}));
    std::vector<double> x(exact.size(), 0.0);
    auto const report = solve(op, rhs, x);
    require(report.converged, "Manufactured sine solve did not converge");
    double const result = error(mesh, x, exact);
    std::cout << "sine n=" << n << " axis=" << refinement_axis << " cells=" << x.size()
              << " error=" << result << " iterations=" << report.iterations << '\n';
    return result;
}

double quadratic_error(int n, int refinement_axis) {
    auto const mesh = make_mesh(n, refinement_axis);
    auto quadratic = [](std::array<double, 3> const& p) {
        return 1.0 + p[0] * p[0] + 0.5 * p[1] * p[1] + 0.25 * p[2] * p[2] + p[0] * p[2];
    };
    std::vector<double> exact(mesh.cells().size()), rhs(exact.size());
    for (size_t i = 0; i < exact.size(); ++i) {
        exact[i] = quadratic(mesh.cells()[i].center);
        rhs[i] = exact[i] - 3.5;
    }
    std::vector<boundary_condition> bc(mesh.faces().size());
    for (size_t f = 0; f < bc.size(); ++f) {
        if (mesh.faces()[f].boundary_side != 0) {
            bc[f] = {1.0, 0.0, quadratic(mesh.faces()[f].center)};
        }
    }
    scalar_operator op(mesh, std::vector<double>(exact.size(), 1.0),
        std::vector<double>(mesh.faces().size(), 1.0), bc);
    std::vector<double> x(exact.size(), 0.0);
    auto const report = solve(op, rhs, x);
    require(report.converged, "Quadratic composite solve did not converge");
    double const result = error(mesh, x, exact);
    std::cout << "quadratic n=" << n << " axis=" << refinement_axis << " error=" << result << '\n';
    return result;
}

void constant_stiff_flux() {
    auto const mesh = make_mesh(4, 1);
    std::vector<double> const exact(mesh.cells().size(), 1.25);
    scalar_operator op(mesh, std::vector<double>(exact.size(), 2.0),
        std::vector<double>(mesh.faces().size(), 1.0e100));
    auto const applied = op.apply(exact);
    auto const fluxes = op.face_fluxes(exact);
    for (double value : applied) {
        require(value == 2.5, "Very stiff diffusion destroyed constant-state reaction");
    }
    for (double value : fluxes) {
        require(value == 0.0, "Very stiff diffusion produced spurious constant-state flux");
    }
    auto x = exact;
    auto const report = solve(op, std::vector<double>(exact.size(), 2.5), x);
    require(report.converged && report.iterations == 0,
        "Exact stiff constant state did not satisfy the initial residual");
}

void tiny_signed_rhs() {
    auto const mesh = make_mesh(4);
    std::vector<double> exact(mesh.cells().size());
    for (size_t i = 0; i < exact.size(); ++i) {
        exact[i] = 1.0e-200 * (mesh.cells()[i].center[0] - 0.5);
    }
    scalar_operator op(mesh, std::vector<double>(exact.size(), 1.0),
        std::vector<double>(mesh.faces().size(), 0.1));
    auto rhs = op.apply(exact);
    std::vector<double> x(exact.size(), 0.0);
    solver_options options;
    options.absolute_tolerance = 0.0;
    auto const report = solve(op, rhs, x, options);
    require(report.converged && report.iterations > 0,
        "Tiny finite right-hand side underflowed the residual norm");
    double relative_error = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        relative_error = std::max(relative_error, std::abs((x[i] - exact[i]) / 1.0e-200));
    }
    require(relative_error < 1.0e-9, "Tiny signed solve is inaccurate");
}

void signed_and_nullspace() {
    auto const mesh = make_mesh(8, -1, {{true, true, true}});
    std::vector<double> exact(mesh.cells().size());
    for (size_t i = 0; i < exact.size(); ++i) {
        auto const& p = mesh.cells()[i].center;
        exact[i] = std::sin(2.0 * pi * p[0]) + 0.4 * std::cos(2.0 * pi * p[1]);
    }
    scalar_operator op(mesh, std::vector<double>(exact.size(), 0.0),
        std::vector<double>(mesh.faces().size(), 1.0));
    auto const rhs = op.apply(exact);
    std::vector<double> x(exact.size(), 7.0);
    auto const report = solve(op, rhs, x);
    require(report.converged, "Periodic nullspace solve did not converge");
    require(error(mesh, x, exact) < 1.0e-9, "Signed periodic correction is inaccurate");
    require(*std::min_element(x.begin(), x.end()) < 0.0,
        "Multigrid improperly clipped signed corrections");
    auto incompatible = rhs;
    for (double& value : incompatible) {
        value += 1.0;
    }
    auto const rejected = solve(op, incompatible, x);
    require(!rejected.converged && rejected.reason.find("incompatible") != std::string::npos,
        "Poisson compatibility failure was not reported");
    auto const neumann_mesh = make_mesh(4);
    std::vector<double> neumann_exact(neumann_mesh.cells().size());
    for (size_t i = 0; i < neumann_exact.size(); ++i) {
        auto const& p = neumann_mesh.cells()[i].center;
        neumann_exact[i] = std::cos(pi * p[0]);
    }
    scalar_operator neumann_op(neumann_mesh, std::vector<double>(neumann_exact.size(), 0.0),
        std::vector<double>(neumann_mesh.faces().size(), 0.5));
    std::vector<double> neumann_x(neumann_exact.size(), 0.0);
    auto const neumann_report = solve(neumann_op, neumann_op.apply(neumann_exact), neumann_x);
    require(neumann_report.converged && error(neumann_mesh, neumann_x, neumann_exact) < 1.0e-9,
        "Reflecting nullspace solve failed");
    std::cout << "periodic signed Poisson iterations=" << report.iterations
              << " reflecting iterations=" << neumann_report.iterations << '\n';
}

void coefficient_jump() {
    auto const mesh = make_mesh(4, 0);
    std::vector<double> beta(mesh.faces().size());
    std::vector<double> cell_beta(mesh.cells().size());
    std::vector<double> exact(cell_beta.size());
    for (size_t i = 0; i < exact.size(); ++i) {
        cell_beta[i] = mesh.cells()[i].center[0] < 0.5 ? 1.0 : 1.0e-6;
        auto const& p = mesh.cells()[i].center;
        exact[i] = 0.2 + sine_field(p);
    }
    for (size_t f = 0; f < beta.size(); ++f) {
        auto const& item = mesh.faces()[f];
        if (item.boundary_side != 0) {
            beta[f] = cell_beta[item.lower == boundary_index ? item.upper : item.lower];
        } else {
            double const lo = mesh.cells()[item.lower].width;
            double const hi = mesh.cells()[item.upper].width;
            beta[f] = (lo + hi) / (lo / cell_beta[item.lower] + hi / cell_beta[item.upper]);
        }
    }
    scalar_operator op(mesh, std::vector<double>(exact.size(), 0.01), beta);
    auto const rhs = op.apply(exact);
    std::vector<double> x(exact.size(), 0.0);
    auto const report = solve(op, rhs, x);
    require(report.converged, "High-contrast coefficient solve did not converge");
    require(error(mesh, x, exact) < 1.0e-7, "High-contrast coefficient solve is inaccurate");
    auto const applied = op.apply(exact);
    double integral = 0.0;
    for (size_t i = 0; i < exact.size(); ++i) {
        integral += mesh.volume(i) * (applied[i] - 0.01 * exact[i]);
    }
    require(std::abs(integral) < 1.0e-13, "Coefficient jump violates leaf conservation");
    std::cout << "contrast=1e6 cells=" << x.size() << " iterations=" << report.iterations
              << " residual=" << report.relative_residual << '\n';
}

void topology_rejection() {
    auto cells = make_cells(4);
    cells.pop_back();
    bool threw = false;
    try {
        composite_mesh broken(std::move(cells), {{0, 0, 0}}, {{1, 1, 1}});
    } catch (std::invalid_argument const&) {
        threw = true;
    }
    require(threw, "Incomplete leaf topology was accepted");
}

}    // namespace

int main() {
    try {
        affine_and_conservation(-1);
        for (int axis = 0; axis < 3; ++axis) {
            affine_and_conservation(axis);
        }
        double const coarse = manufactured_error(8, -1);
        double const fine = manufactured_error(16, -1);
        require(coarse / fine > 3.7, "Uniform manufactured solution lost second-order convergence");
        for (int axis = 0; axis < 3; ++axis) {
            double const amr_coarse = manufactured_error(4, axis);
            double const amr_fine = manufactured_error(8, axis);
            require(amr_coarse / amr_fine > 3.2,
                "Composite manufactured solution lost second-order convergence");
            double const quadratic_coarse = quadratic_error(4, axis);
            double const quadratic_fine = quadratic_error(8, axis);
            require(quadratic_coarse / quadratic_fine > 3.2,
                "Composite quadratic solution lost second-order convergence");
        }
        signed_and_nullspace();
        coefficient_jump();
        constant_stiff_flux();
        tiny_signed_rhs();
        topology_rejection();
        std::cout << "elliptic tests passed\n";
    } catch (std::exception const& failure) {
        std::cerr << "elliptic test failure: " << failure.what() << '\n';
        return 1;
    }
    return 0;
}
