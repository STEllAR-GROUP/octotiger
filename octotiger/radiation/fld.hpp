// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/elliptic/composite.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stddef.h>
#include <string>
#include <vector>

namespace radiation { namespace fld {

// Physical-c, grey two-temperature FLD. The transport and
// absorption coefficients include density and have units inverse length.
// No radiation momentum is evolved and diagnostic flux never imparts recoil.
struct cell {
    double energy = 0;
    double internal_energy = 0;
    double chi_rosseland = 0;
    double chi_planck = 0;
    double log_alpha = 0; // B(u)=exp(log_alpha)*u^4.
    bool linear_emission = false; // Marshak verification: B(u)=u.
    std::array<double, 3> velocity{}; // Frozen during this implicit solve.
};

using opacity_callback = std::function<bool(size_t, double, double&, double&)>;
using boundary_condition = octotiger::elliptic::boundary_condition;
using composite_mesh = octotiger::elliptic::composite_mesh;

struct settings {
    double nonlinear_tolerance = 1.e-10;
    double linear_tolerance = 1.e-10;
    int max_iterations = 50;
    int max_linear_iterations = 200;
    int max_line_search = 24;
};

struct result {
    bool success = false;
    std::string message;
    std::vector<cell> state;
    std::vector<std::array<double, 3>> flux;
    std::vector<double> face_flux; // Physical per-area flux: +axis inside, outward at boundary.
    std::vector<double> boundary_energy_by_cell; // Positive into the domain, step integrated.
    std::vector<double> source_energy_by_cell; // Positive into radiation, step integrated.
    double boundary_energy = 0;
    double source_energy = 0;
    double conservation_error = 0;
    int iterations = 0;
    int linear_iterations = 0;
    int vcycles = 0;
    double nonlinear_residual = 0;
    double linear_residual = 0;
};

// lambda*R, evaluated without R^2 overflow and with its exact thick/streaming
// limits. The reciprocal form also works when R is +infinity.
inline double limiter_flux_factor(double R) {
    if (!(R >= 0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (R <= 1) {
        return R * (2 + R) / (6 + 3 * R + R * R);
    }
    const double inverse = 1 / R;
    return (1 + 2 * inverse) / (1 + 3 * inverse + 6 * inverse * inverse);
}

inline double limiter(double R) {
    if (!(R >= 0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (R <= 1) {
        return (2 + R) / (6 + 3 * R + R * R);
    }
    const double inverse = 1 / R;
    return inverse * (1 + 2 * inverse) / (1 + 3 * inverse + 6 * inverse * inverse);
}

// At E=0 there is no outgoing FLD flux. A flat vacuum (chi=|grad E|=0)
// also has zero flux and uses D=0; neither case divides by zero or inserts a
// fictitious opacity floor. Nonflat vacuum uses its finite streaming limit.
inline double diffusion_coefficient(double E, double chi, double gradient, double c) {
    if (!(E >= 0) || !(chi >= 0) || !(gradient >= 0) || !(c > 0) ||
        !std::isfinite(E) || !std::isfinite(chi) || !std::isfinite(gradient) || !std::isfinite(c)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (E == 0) {
        return 0;
    }
    if (gradient == 0) {
        return chi > 0 ? static_cast<double>(static_cast<long double>(c) / (3 * static_cast<long double>(chi))) : 0;
    }
    const long double g = gradient;
    const long double optical = static_cast<long double>(chi) * E;
    long double D;
    if (optical <= g) {
        const long double t = optical / g;
        D = static_cast<long double>(c) * E / g * (1 + 2 * t) / (1 + 3 * t + 6 * t * t);
    } else {
        const long double R = g / optical;
        D = static_cast<long double>(c) / chi * (2 + R) / (6 + 3 * R + R * R);
    }
    return static_cast<double>(D);
}

namespace detail {

inline double norm(const std::array<double, 3>& v) {
    return std::hypot(std::hypot(v[0], v[1]), v[2]);
}

inline void absorption_weights(double dt, double c, double chi, double& q, double& w) {
    if (dt == 0 || chi == 0) {
        q = 1;
        w = 0;
        return;
    }
    const double log_a = std::log(dt) + std::log(c) + std::log(chi);
    if (log_a >= 0) {
        const double reciprocal = std::exp(-log_a);
        w = 1 / (1 + reciprocal);
        q = reciprocal * w;
    } else {
        const double a = std::exp(log_a);
        q = 1 / (1 + a);
        w = a * q;
    }
}

// Positive root a*u + b*alpha*u^4=rhs. Both scaled polynomial
// coefficients lie in [0,1], so Newton starts above the positive root.
inline bool thermal_root(double a, double b, double log_alpha, bool linear,
    double rhs, double& u) {
    if (!(rhs >= 0) || !std::isfinite(rhs) || !(a >= 0) || !(b >= 0)) {
        return false;
    }
    if (rhs == 0) {
        u = 0;
        return true;
    }
    if (linear) {
        u = rhs / (a + b);
        return std::isfinite(u) && u >= 0;
    }
    if (b == 0) {
        u = rhs / a;
        return std::isfinite(u) && u >= 0;
    }
    const double lr = std::log(rhs);
    const double linear_scale = a > 0 ? lr - std::log(a) : std::numeric_limits<double>::infinity();
    const double quartic_scale = (lr - std::log(b) - log_alpha) / 4;
    const double log_scale = std::min(linear_scale, quartic_scale);
    const double scale = std::exp(log_scale);
    if (!(scale > 0) || !std::isfinite(scale)) {
        return false;
    }
    const double A = a > 0 ? std::exp(std::min(0.0, std::log(a) + log_scale - lr)) : 0;
    const double B = std::exp(std::min(0.0, std::log(b) + log_alpha + 4 * log_scale - lr));
    double y = 1;
    for (int iteration = 0; iteration < 20; ++iteration) {
        const double y2 = y * y;
        const double residual = A * y + B * y2 * y2 - 1;
        if (std::abs(residual) <= 16 * std::numeric_limits<double>::epsilon()) {
            u = scale * y;
            return std::isfinite(u) && u >= 0;
        }
        y -= residual / (A + 4 * B * y2 * y);
    }
    return false;
}

inline bool opacity(size_t i, double u, const cell& input, const opacity_callback& callback,
    double& chi_R, double& chi_P) {
    chi_R = input.chi_rosseland;
    chi_P = input.chi_planck;
    if (callback && !callback(i, u, chi_R, chi_P)) {
        return false;
    }
    return std::isfinite(chi_R) && chi_R >= 0 && std::isfinite(chi_P) && chi_P >= 0;
}

inline double material_derivative(double u, double q, double w, const cell& input) {
    if (w == 0) {
        return 0;
    }
    if (input.linear_emission) {
        return w / (q + w);
    }
    if (u == 0) {
        return q > 0 ? w / q : std::numeric_limits<double>::infinity();
    }
    const double log_bp = std::log(4.0) + input.log_alpha + 3 * std::log(u);
    const double first = q > 0 ? std::log(q) : -std::numeric_limits<double>::infinity();
    const double second = std::log(w) + log_bp;
    const double scale = std::max(first, second);
    const double denominator = scale + std::log(std::exp(first - scale) + std::exp(second - scale));
    return std::exp(std::log(w) - denominator);
}

// Mixed-frame thermal target after splitting off the explicit gas work
// v.f: h*E - 2*v.Fd/c^2, h=1-v.(I+f).v/c^2. Krumholz et al. (2007),
// equations 38--42. Retain the velocity-squared term for dynamic diffusion.
// The gradient/opacity/limiter are refreshed for EVERY nonlinear trial.
inline bool thermal_target(double E, double chi, const std::array<double, 3>& gradient,
    const std::array<double, 3>& velocity, double c, double& target, double& h) {
    const double g = norm(gradient);
    const double beta = norm(velocity) / c;
    if (!std::isfinite(beta) || beta > .1) {
        return false;
    }
    const long double optical = static_cast<long double>(chi) * E;
    const double R = g == 0 ? 0 : optical > 0 ? static_cast<double>(g / optical)
        : std::numeric_limits<double>::infinity();
    const double lambda = limiter(R);
    const double flux_factor = limiter_flux_factor(R);
    const double f = lambda + flux_factor * flux_factor;
    double projection = 0;
    if (g > 0) {
        for (int d = 0; d < 3; ++d) {
            projection += (velocity[d] / c) * (gradient[d] / g);
        }
    }
    h = 1 - .5 * (3 - f) * beta * beta - .5 * (3 * f - 1) * projection * projection;
    // Fd/c = -E*(lambda R)*n avoids overflow in c/chi in thin material.
    target = E * (h + 2 * flux_factor * projection);
    return std::isfinite(target) && target >= 0 && h > 0;
}

// Exact local elimination for fixed absorption. For fitted opacity, repeated
// local solves refresh absorption at the resulting material temperature;
// failure to converge is a rejected nonlinear evaluation, never a clipped
// or silently frozen-opacity accepted state.
inline bool material(size_t i, double E, double dt, double c, const cell& input,
    const opacity_callback& callback, double seed, cell& output, double& derivative,
    const std::array<double, 3>& gradient = {}) {
    output = input;
    output.energy = E;
    double u = seed;
    for (int iteration = 0; iteration < (callback ? 100 : 1); ++iteration) {
        double chi_R, chi_P;
        if (!opacity(i, u, input, callback, chi_R, chi_P)) {
            return false;
        }
        double q, w;
        absorption_weights(dt, c, chi_P, q, w);
        double target, h;
        if (!thermal_target(E, chi_R, gradient, input.velocity, c, target, h)) {
            return false;
        }
        const double rhs = q * input.internal_energy + w * target;
        double next;
        if (!thermal_root(q, w, input.log_alpha, input.linear_emission, rhs, next)) {
            return false;
        }
        const double scale = std::max({std::abs(next), std::abs(u), std::numeric_limits<double>::min()});
        double checked_R, checked_P;
        if (!opacity(i, next, input, callback, checked_R, checked_P)) {
            return false;
        }
        absorption_weights(dt, c, checked_P, q, w);
        if (!thermal_target(E, checked_R, gradient, input.velocity, c, target, h)) {
            return false;
        }
        double checked;
        if (!thermal_root(q, w, input.log_alpha, input.linear_emission,
                q * input.internal_energy + w * target, checked)) {
            return false;
        }
        if (!callback || std::abs(checked - next) <= 128 * std::numeric_limits<double>::epsilon() * scale) {
            output.internal_energy = next;
            output.chi_rosseland = checked_R;
            output.chi_planck = checked_P;
            // Lag closure/gradient derivatives only in the preconditioner.
            // The accepted residual always uses the full target above.
            derivative = h * material_derivative(next, q, w, input);
            return std::isfinite(derivative) && derivative >= 0;
        }
        // Underrelax coefficient iteration for temperature-dependent opacity.
        u = .5 * u + .5 * next;
    }
    return false;
}

inline const boundary_condition& condition(const composite_mesh& mesh,
    const std::vector<boundary_condition>& bc, size_t f) {
    const auto& face = mesh.faces()[f];
    if (bc.size() == 6) {
        return bc[2 * face.axis + (face.boundary_side > 0 ? 1 : 0)];
    }
    return bc[f];
}

inline std::vector<boundary_condition> scaled_conditions(const composite_mesh& mesh,
    const std::vector<boundary_condition>& bc, double dt, bool homogeneous) {
    std::vector<boundary_condition> result(mesh.faces().size());
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        if (mesh.faces()[f].boundary_side == 0) {
            continue;
        }
        result[f] = condition(mesh, bc, f);
        result[f].alpha *= dt;
        result[f].value = homogeneous ? 0 : result[f].value * dt;
    }
    return result;
}

inline bool coefficients(const composite_mesh& mesh, const std::vector<cell>& state,
    double c, const std::vector<boundary_condition>& bc, std::vector<double>& D,
    std::vector<std::array<double, 3>>& flux) {
    const auto& faces = mesh.faces();
    std::vector<double> E(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        E[i] = state[i].energy;
    }
    D.resize(faces.size());
    flux.resize(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        const auto gradient = mesh.cell_gradient(i, E);
        const double g = norm(gradient);
        const double d = diffusion_coefficient(E[i], state[i].chi_rosseland, g, c);
        if (!std::isfinite(d)) {
            return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
            flux[i][axis] = -d * gradient[axis];
        }
    }
    for (size_t f = 0; f < faces.size(); ++f) {
        const auto& face = faces[f];
        auto gradient = mesh.face_gradient(f, E);
        double face_E, chi;
        if (face.boundary_side == 0) {
            const size_t lo = face.lower, hi = face.upper;
            const double low_E = std::min(E[lo], E[hi]);
            const double high_E = std::max(E[lo], E[hi]);
            // Limit the reconstructed face value, never the cell energies.
            // A single positive face state bounds the shared vector flux.
            face_E = std::max(low_E, std::min(high_E, mesh.face_value(f, E)));
            // This distance-weighted extinction is harmonic transmission of
            // thick-limit diffusivity across a coefficient discontinuity.
            const double wl = mesh.cells()[lo].width;
            const double wh = mesh.cells()[hi].width;
            chi = static_cast<double>((static_cast<long double>(wl) * state[lo].chi_rosseland +
                static_cast<long double>(wh) * state[hi].chi_rosseland) / (wl + wh));
        } else {
            const size_t i = face.lower == octotiger::elliptic::boundary_index ? face.upper : face.lower;
            const auto& b = condition(mesh, bc, f);
            face_E = E[i];
            chi = state[i].chi_rosseland;
            if (b.gamma == 0) {
                face_E = b.value / b.alpha;
                gradient[face.axis] = face.boundary_side * (face_E - E[i]) / face.distance;
            } else if (b.alpha == 0 && b.value == 0) {
                gradient[face.axis] = 0;
            } else {
                // The boundary coefficient must satisfy the LP closure at
                // the ACTUAL Robin trace, not at an independently lagged
                // boundary energy. Solve its one-dimensional constitutive
                // relation here; the outer residual then uses the same D.
                const long double tangential_squared =
                    static_cast<long double>(gradient[(face.axis + 1) % 3]) * gradient[(face.axis + 1) % 3] +
                    static_cast<long double>(gradient[(face.axis + 2) % 3]) * gradient[(face.axis + 2) % 3];
                auto boundary_diffusion = [&](double trial_D, double& value_E, double& normal) {
                    const long double denominator = static_cast<long double>(b.gamma) * trial_D +
                        static_cast<long double>(b.alpha) * face.distance;
                    if (!(denominator > 0)) {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                    value_E = static_cast<double>((static_cast<long double>(b.gamma) * trial_D * E[i] +
                        static_cast<long double>(b.value) * face.distance) / denominator);
                    normal = static_cast<double>((static_cast<long double>(b.value) -
                        static_cast<long double>(b.alpha) * E[i]) / denominator);
                    if (value_E < 0 || !std::isfinite(value_E)) {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                    const double magnitude = static_cast<double>(std::sqrt(tangential_squared +
                        static_cast<long double>(normal) * normal));
                    return diffusion_coefficient(value_E, chi, magnitude, c);
                };
                if (E[i] == 0 && b.value == 0) {
                    D[f] = 0;
                    continue;
                }
                {
                    double lower = 0;
                    const double thick = chi > 0 ? diffusion_coefficient(1, chi, 0, c) :
                        std::numeric_limits<double>::max();
                    double upper = std::min(thick, c * face.distance);
                    double normal = 0;
                    if (!std::isfinite(upper) || !(upper > 0)) {
                        return false;
                    }
                    bool bracketed = false;
                    for (int growth = 0; growth < 1024; ++growth) {
                        const double closure = boundary_diffusion(upper, face_E, normal);
                        if (std::isfinite(closure) && closure <= upper) {
                            bracketed = true;
                            break;
                        }
                        lower = upper;
                        if (upper >= thick || upper >= std::numeric_limits<double>::max() / 2) {
                            break;
                        }
                        upper = std::min(thick, 2 * upper);
                    }
                    if (!bracketed) {
                        return false;
                    }
                    for (int iteration = 0; iteration < 64; ++iteration) {
                        const double trial = .5 * lower + .5 * upper;
                        const double closure = boundary_diffusion(trial, face_E, normal);
                        if (!std::isfinite(closure)) {
                            // Prescribed outward flux can make small-D
                            // boundary traces negative. Such D is inadmissible.
                            lower = trial;
                        } else if (closure > trial) {
                            lower = trial;
                        } else {
                            upper = trial;
                        }
                    }
                    const double converged_D = .5 * lower + .5 * upper;
                    const double closure = boundary_diffusion(converged_D, face_E, normal);
                    if (!std::isfinite(closure) || std::abs(closure - converged_D) >
                            1.e-10 * std::max(closure, converged_D)) {
                        return false;
                    }
                    gradient[face.axis] = face.boundary_side * normal;
                    D[f] = converged_D;
                    continue;
                }

            }
        }
        D[f] = diffusion_coefficient(face_E, chi, norm(gradient), c);
        if (!std::isfinite(D[f]) || D[f] < 0) {
            return false;
        }
    }
    return true;
}

struct evaluation {
    std::vector<cell> state;
    std::vector<double> reaction, diffusion, residual, physical_face_flux;
    std::vector<std::array<double, 3>> flux;
    double norm = 0;
};

inline bool evaluate(const composite_mesh& mesh, const std::vector<cell>& initial,
    const std::vector<double>& E, const std::vector<cell>& seed, double dt, double c,
    const std::vector<boundary_condition>& bc, const opacity_callback& callback,
    double scale, evaluation& output) {
    output.state.resize(initial.size());
    output.reaction.resize(initial.size());
    for (size_t i = 0; i < initial.size(); ++i) {
        if (!std::isfinite(E[i]) || E[i] < 0) {
            return false;
        }
        double derivative;
        if (!material(i, E[i], dt, c, initial[i], callback, seed[i].internal_energy,
                output.state[i], derivative, mesh.cell_gradient(i, E))) {
            return false;
        }
        output.reaction[i] = 1 + derivative;
    }
    if (!coefficients(mesh, output.state, c, bc, output.diffusion, output.flux)) {
        return false;
    }
    std::vector<double> beta = output.diffusion;
    for (auto& value : beta) {
        value *= dt;
    }
    octotiger::elliptic::scalar_operator transport(mesh, std::vector<double>(E.size(), 0),
        beta, scaled_conditions(mesh, bc, dt, false));
    output.residual = transport.apply(E);
    const auto boundary_rhs = transport.boundary_rhs();
    output.physical_face_flux = transport.face_fluxes(E);
    for (auto& value : output.physical_face_flux) {
        value /= dt;
    }
    output.norm = 0;
    long double integrated_residual = 0, integrated_energy = 0;
    for (size_t i = 0; i < E.size(); ++i) {
        output.residual[i] += (E[i] - initial[i].energy) +
            (output.state[i].internal_energy - initial[i].internal_energy) - boundary_rhs[i];
        output.norm = std::max(output.norm, std::abs(output.residual[i]) / scale);
        integrated_residual += std::abs(static_cast<long double>(output.residual[i])) * mesh.volume(i);
        integrated_energy += std::max(static_cast<long double>(initial[i].energy) + initial[i].internal_energy,
            static_cast<long double>(E[i]) + output.state[i].internal_energy) * mesh.volume(i);
    }
    // A hot localized cell must not hide a whole-domain energy imbalance.
    // Require both the cellwise residual and its volume-weighted L1 norm.
    if (integrated_energy > 0) {
        output.norm = std::max(output.norm, static_cast<double>(integrated_residual / integrated_energy));
    }
    return std::isfinite(output.norm);
}

inline bool valid_input(const composite_mesh& mesh, const std::vector<cell>& initial,
    double dt, double c, const std::vector<boundary_condition>& bc, const settings& options) {
    if (initial.size() != mesh.cells().size() || initial.empty() ||
        (bc.size() != 6 && bc.size() != mesh.faces().size()) ||
        !std::isfinite(dt) || dt < 0 || !std::isfinite(c) || c <= 0 ||
        !std::isfinite(options.nonlinear_tolerance) || options.nonlinear_tolerance <= 0 ||
        !std::isfinite(options.linear_tolerance) || options.linear_tolerance <= 0 ||
        options.max_iterations <= 0 || options.max_linear_iterations <= 0 || options.max_line_search <= 0) {
        return false;
    }
    for (const auto& input : initial) {
        if (!std::isfinite(input.energy) || input.energy < 0 ||
            !std::isfinite(input.internal_energy) || input.internal_energy < 0 ||
            !std::isfinite(input.chi_planck) || input.chi_planck < 0 ||
            !std::isfinite(input.chi_rosseland) || input.chi_rosseland < 0 ||
            !std::isfinite(input.log_alpha) || !std::isfinite(norm(input.velocity)) ||
            norm(input.velocity) / c > .1) {
            return false;
        }
    }
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        if (mesh.faces()[f].boundary_side == 0) {
            continue;
        }
        const auto& b = condition(mesh, bc, f);
        if (!std::isfinite(b.alpha) || b.alpha < 0 || !std::isfinite(b.gamma) || b.gamma < 0 ||
            !std::isfinite(b.value) || (b.alpha == 0 && b.gamma == 0) ||
            (b.gamma == 0 && b.value < 0)) {
            return false;
        }
    }
    return true;
}

} // namespace detail

// A failed attempt returns the original state and zero budgets. The caller
// can retry collectively at a smaller dt and commit only success=true.
inline result solve(const composite_mesh& mesh, const std::vector<cell>& initial,
    double dt, double c, const std::vector<boundary_condition>& bc,
    const settings& options = {}, const opacity_callback& callback = {}) {
    result answer;
    answer.state = initial;
    answer.boundary_energy_by_cell.assign(initial.size(), 0);
    answer.source_energy_by_cell.assign(initial.size(), 0);
    if (!detail::valid_input(mesh, initial, dt, c, bc, options)) {
        answer.message = "Invalid FLD state, boundary, or solver settings";
        return answer;
    }
    bool empty_equilibrium = true;
    for (const auto& value : initial) {
        empty_equilibrium = empty_equilibrium && value.energy == 0 && value.internal_energy == 0;
    }
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        if (mesh.faces()[f].boundary_side != 0) {
            empty_equilibrium = empty_equilibrium && detail::condition(mesh, bc, f).value == 0;
        }
    }
    if (dt == 0 || empty_equilibrium) {
        // In an unforced empty medium the exact solution stays empty even in
        // the infinitely stiff limit, whose eliminated derivative is singular.
        std::vector<double> diffusion;
        for (size_t i = 0; i < initial.size(); ++i) {
            if (!detail::opacity(i, initial[i].internal_energy, initial[i], callback,
                    answer.state[i].chi_rosseland, answer.state[i].chi_planck)) {
                answer.message = "Invalid FLD opacity during diagnostic refresh";
                return answer;
            }
        }
        if (!detail::coefficients(mesh, answer.state, c, bc, diffusion, answer.flux)) {
            answer.message = "FLD diagnostic reconstruction failed";
            return answer;
        }
        std::vector<double> E(initial.size());
        for (size_t i = 0; i < E.size(); ++i) {
            E[i] = initial[i].energy;
        }
        octotiger::elliptic::scalar_operator transport(mesh, std::vector<double>(E.size(), 0), diffusion, bc);
        answer.face_flux = transport.face_fluxes(E);
        answer.success = true;
        return answer;
    }
    std::vector<double> E(initial.size());
    double scale = std::numeric_limits<double>::min();
    // The incoming state may be identically dark (a Marshak front). Include
    // the prescribed bath or boundary fluence when normalizing residuals.
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        if (mesh.faces()[f].boundary_side != 0) {
            const auto& boundary = detail::condition(mesh, bc, f);
            if (boundary.alpha > 0) {
                scale = std::max(scale, std::abs(boundary.value / boundary.alpha));
            } else if (boundary.gamma > 0) {
                scale = std::max(scale, std::abs(dt * boundary.value /
                    (boundary.gamma * mesh.faces()[f].distance)));
            }
        }
    }
    for (size_t i = 0; i < initial.size(); ++i) {
        const auto& input = initial[i];
        scale = std::max({scale, input.energy, input.internal_energy});
        double chi_R, chi_P, q, w;
        if (!detail::opacity(i, input.internal_energy, input, callback, chi_R, chi_P)) {
            answer.message = "Invalid FLD opacity at incoming state";
            return answer;
        }
        detail::absorption_weights(dt, c, chi_P, q, w);
        double gas;
        if (!detail::thermal_root(1, w, input.log_alpha, input.linear_emission,
                input.internal_energy + w * input.energy, gas)) {
            answer.message = "FLD homogeneous source seed failed";
            return answer;
        }
        // Avoid subtracting nearly equal total and gas energies in a cold,
        // weakly coupled cell: recover E from the emission relation instead.
        long double emitted = 0;
        if (w > 0 && gas > 0) {
            emitted = input.linear_emission ? static_cast<long double>(w) * gas :
                std::exp(static_cast<long double>(std::log(w)) + input.log_alpha + 4 * std::log(static_cast<long double>(gas)));
        }
        E[i] = static_cast<double>(static_cast<long double>(q) * input.energy + emitted);
        if (detail::norm(input.velocity) > 0) {
            // The stationary homogeneous seed is not moving LTE. Starting
            // from the incoming radiation state preserves boosted equilibrium.
            E[i] = input.energy;
        }
        if (!std::isfinite(E[i]) || E[i] < 0) {
            answer.message = "FLD homogeneous source seed is nonfinite";
            return answer;
        }
    }
    // A positive starting guess in exactly dark cells keeps signed roundoff
    // in the iterative linear correction from making the admissible damping
    // identically zero at a cold front. This is ONLY an initial iterate: its
    // energy is included in the residual and never added to incoming state
    // or budgets. No accepted state is clipped or floored.
    if (scale > std::numeric_limits<double>::min()) {
        const double seed = scale * std::min(1.e-6, options.nonlinear_tolerance);
        for (auto& value : E) {
            if (value == 0) {
                value = seed;
            }
        }
    }
    detail::evaluation current;
    if (!detail::evaluate(mesh, initial, E, initial, dt, c, bc, callback, scale, current)) {
        answer.message = "FLD initial nonlinear evaluation failed";
        return answer;
    }
    for (int iteration = 0; iteration <= options.max_iterations; ++iteration) {
        answer.iterations = iteration;
        answer.nonlinear_residual = current.norm;
        if (current.norm <= options.nonlinear_tolerance) {
            answer.state = std::move(current.state);
            answer.flux = std::move(current.flux);
            answer.face_flux = std::move(current.physical_face_flux);
            long double total_change = 0, boundary_energy = 0, source_energy = 0;
            for (size_t i = 0; i < initial.size(); ++i) {
                const double volume = mesh.volume(i);
                answer.source_energy_by_cell[i] = (initial[i].internal_energy - answer.state[i].internal_energy) * volume;
                source_energy += answer.source_energy_by_cell[i];
                total_change += ((static_cast<long double>(answer.state[i].energy) - initial[i].energy) +
                    (static_cast<long double>(answer.state[i].internal_energy) - initial[i].internal_energy)) * volume;
            }
            for (size_t f = 0; f < mesh.faces().size(); ++f) {
                const auto& face = mesh.faces()[f];
                if (face.boundary_side == 0) {
                    continue;
                }
                const size_t i = face.lower == octotiger::elliptic::boundary_index ? face.upper : face.lower;
                const double incoming = -dt * answer.face_flux[f] * face.area;
                answer.boundary_energy_by_cell[i] += incoming;
                boundary_energy += incoming;
            }
            answer.source_energy = static_cast<double>(source_energy);
            answer.boundary_energy = static_cast<double>(boundary_energy);
            answer.conservation_error = static_cast<double>(total_change - boundary_energy);
            answer.success = true;
            return answer;
        }
        if (iteration == options.max_iterations) {
            break;
        }
        auto beta = current.diffusion;
        for (auto& value : beta) {
            value *= dt;
        }
        octotiger::elliptic::scalar_operator jacobian(mesh, current.reaction, beta,
            detail::scaled_conditions(mesh, bc, dt, true));
        auto rhs = current.residual;
        for (auto& value : rhs) {
            value = -value;
        }
        std::vector<double> direction(E.size(), 0);
        octotiger::elliptic::solver_options linear_options;
        linear_options.relative_tolerance = options.linear_tolerance;
        linear_options.absolute_tolerance = 0;
        linear_options.maximum_iterations = options.max_linear_iterations;
        const auto linear = octotiger::elliptic::solve(jacobian, rhs, direction, linear_options);
        answer.linear_iterations += linear.iterations;
        answer.vcycles += linear.vcycles;
        answer.linear_residual = linear.relative_residual;
        if (!linear.converged) {
            answer.message = "FLD scalar multigrid solve did not converge";
            return answer;
        }
        // Make the inexact Newton DIRECTION admissible in each cell before
        // the global residual line search. A norm-converged Krylov correction
        // can carry signed roundoff larger than a cold cell's energy; using
        // that cell to damp every other update stalls an advancing front.
        // This local fraction-to-boundary changes only the search direction.
        // Every candidate still recomputes the full coupled residual, and no
        // accepted energy or conservation budget is subsequently clipped.
        for (size_t i = 0; i < E.size(); ++i) {
            if (direction[i] < -.99 * E[i]) {
                direction[i] = -.99 * E[i];
            }
        }
        double damping = 1;
        bool accepted = false;
        for (int backtrack = 0; backtrack < options.max_line_search && damping > 0; ++backtrack) {
            std::vector<double> candidate(E.size());
            for (size_t i = 0; i < E.size(); ++i) {
                candidate[i] = E[i] + damping * direction[i];
            }
            detail::evaluation trial;
            if (detail::evaluate(mesh, initial, candidate, current.state, dt, c, bc, callback, scale, trial) &&
                (trial.norm <= options.nonlinear_tolerance || trial.norm < (1 - 1.e-4 * damping) * current.norm)) {
                E = std::move(candidate);
                current = std::move(trial);
                accepted = true;
                break;
            }
            damping *= .5;
        }
        if (!accepted) {
            answer.message = "FLD positivity-preserving nonlinear line search failed";
            return answer;
        }
    }
    answer.message = "FLD nonlinear iteration limit reached";
    return answer;
}

// A prescribed, time-independent emissivity enters the backward-Euler RHS.
// Shifting only incoming E by dt*q is exact: the material equation still uses
// its original u, and the coupled residual contains E_new-E_old+u_new-u_old+
// dt*div(F)-dt*q. The shifted state is scratch, never a committed source split.
inline result solve_with_source(const composite_mesh& mesh, const std::vector<cell>& initial,
    const std::vector<double>& emissivity, double dt, double c,
    const std::vector<boundary_condition>& bc, const settings& options = {},
    const opacity_callback& callback = {}) {
    result rejected;
    rejected.state = initial;
    rejected.source_energy_by_cell.assign(initial.size(), 0);
    rejected.boundary_energy_by_cell.assign(initial.size(), 0);
    rejected.message = "Invalid prescribed FLD emissivity or source interval";
    if (emissivity.size() != initial.size() || !std::isfinite(dt) || dt < 0 ||
        !detail::valid_input(mesh, initial, dt, c, bc, options)) {
        return rejected;
    }
    auto shifted = initial;
    std::vector<double> injected(initial.size());
    for (size_t i = 0; i < initial.size(); ++i) {
        if (!std::isfinite(emissivity[i]) || emissivity[i] < 0) {
            return rejected;
        }
        const long double energy = static_cast<long double>(initial[i].energy) +
            static_cast<long double>(dt) * emissivity[i];
        const long double budget = static_cast<long double>(dt) * emissivity[i] * mesh.volume(i);
        if (!std::isfinite(energy) || energy > std::numeric_limits<double>::max() ||
            !std::isfinite(budget) || budget > std::numeric_limits<double>::max()) {
            return rejected;
        }
        shifted[i].energy = static_cast<double>(energy);
        injected[i] = static_cast<double>(budget);
    }
    auto answer = solve(mesh, shifted, dt, c, bc, options, callback);
    if (!answer.success) {
        // A retry starts from the physical state, not this failed RHS shift.
        answer.state = initial;
        return answer;
    }
    long double balance = -static_cast<long double>(answer.boundary_energy);
    for (size_t i = 0; i < initial.size(); ++i) {
        answer.source_energy_by_cell[i] += injected[i];
        answer.source_energy += injected[i];
        balance += ((static_cast<long double>(answer.state[i].energy) - initial[i].energy) +
            (static_cast<long double>(answer.state[i].internal_energy) - initial[i].internal_energy)) *
            mesh.volume(i) - injected[i];
    }
    // Account for rounding of the shifted RHS in the physical energy ledger.
    answer.conservation_error = static_cast<double>(balance);
    return answer;
}

inline result refresh(const composite_mesh& mesh, const std::vector<cell>& state,
    double c, const std::vector<boundary_condition>& bc, const opacity_callback& callback = {}) {
    return solve(mesh, state, 0, c, bc, settings{}, callback);
}

}} // namespace radiation::fld
