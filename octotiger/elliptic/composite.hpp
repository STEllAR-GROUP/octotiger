// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

// CPU reference finite-volume operator and geometric correction multigrid.
// The physical unknowns are leaf cells only. Coarse meshes belong exclusively
// to the linear solver. There is one flux for every shared (possibly split)
// face, including during residual evaluation.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace octotiger {
namespace elliptic {

constexpr size_t boundary_index = std::numeric_limits<size_t>::max();

struct cell {
    std::array<double, 3> center;
    double width;
};

struct face {
    size_t lower = boundary_index;
    size_t upper = boundary_index;
    int axis = 0;
    // -1/+1 on an exterior face, zero on a shared or periodic face.
    int boundary_side = 0;
    std::array<double, 3> center = {{0.0, 0.0, 0.0}};
    double area = 0.0;
    double distance = 0.0;
};

using sparse_row = std::vector<std::pair<size_t, double>>;
using sparse_matrix = std::vector<sparse_row>;

namespace detail {

inline sparse_row compact(const std::map<size_t, double>& values) {
    sparse_row result;
    for (const auto& item : values) {
        if (item.second != 0.0) {
            result.push_back(item);
        }
    }
    return result;
}

inline void accumulate(std::map<size_t, double>& row, const sparse_row& other,
    double scale) {
    for (const auto& item : other) {
        row[item.first] += scale * item.second;
    }
}

inline double evaluate(const sparse_row& row, const std::vector<double>& x) {
    double result = 0.0;
    for (const auto& item : row) {
        result += item.second * x[item.first];
    }
    return result;
}

// Gradient stencils have zero row sum analytically. Evaluating differences
// preserves a constant field exactly even when beta or the time step is large.
inline double evaluate_derivative(const sparse_row& row, const std::vector<double>& x) {
    if (row.empty()) {
        return 0.0;
    }
    const double reference = x[row.front().first];
    double result = 0.0;
    for (const auto& item : row) {
        result += item.second * (x[item.first] - reference);
    }
    return result;
}

inline double evaluate_interpolation(const sparse_row& row, const std::vector<double>& x) {
    return row.empty() ? 0.0 : x[row.front().first] + evaluate_derivative(row, x);
}

inline std::vector<double> multiply(const sparse_matrix& matrix,
    const std::vector<double>& x) {
    std::vector<double> result(matrix.size());
    for (size_t i = 0; i < matrix.size(); ++i) {
        result[i] = evaluate(matrix[i], x);
    }
    return result;
}

inline long long lattice_coordinate(double value, double origin, double h) {
    const double position = (value - origin) / h;
    if (!std::isfinite(position) || std::abs(position) > 1.0e12) {
        throw std::invalid_argument("Elliptic mesh coordinate exceeds its lattice");
    }
    const long long integer = std::llround(position);
    if (std::abs(position - static_cast<double>(integer)) > 2.0e-7) {
        throw std::invalid_argument("Elliptic cells must align to a dyadic lattice");
    }
    return integer;
}

inline bool power_of_two(long long value) {
    return value > 0 && (value & (value - 1)) == 0;
}

} // namespace detail

class composite_mesh {
public:
    composite_mesh(std::vector<cell> cells, std::array<double, 3> lower,
        std::array<double, 3> upper,
        std::array<bool, 3> periodic = {{false, false, false}})
      : cells_(std::move(cells)), lower_(lower), upper_(upper), periodic_(periodic) {
        build_faces();
        build_gradients();
        build_face_stencils();
    }

    const std::vector<cell>& cells() const { return cells_; }
    const std::vector<face>& faces() const { return faces_; }
    const std::array<double, 3>& lower() const { return lower_; }
    const std::array<double, 3>& upper() const { return upper_; }
    const std::array<bool, 3>& periodic() const { return periodic_; }
    double minimum_width() const { return minimum_width_; }
    double volume(size_t i) const {
        const double h = cells_.at(i).width;
        return h * h * h;
    }

    const sparse_row& gradient_stencil(size_t i, int axis) const {
        return gradients_.at(i).at(axis);
    }

    const sparse_row& face_normal_stencil(size_t f) const {
        return normal_stencils_.at(f);
    }

    std::array<double, 3> cell_gradient(size_t i,
        const std::vector<double>& x) const {
        validate_size(x);
        std::array<double, 3> result;
        for (int d = 0; d < 3; ++d) {
            result[d] = detail::evaluate_derivative(gradients_.at(i)[d], x);
        }
        return result;
    }

    std::array<double, 3> face_gradient(size_t f,
        const std::vector<double>& x) const {
        validate_size(x);
        const face& item = faces_.at(f);
        if (item.boundary_side != 0) {
            return cell_gradient(item.lower == boundary_index ? item.upper : item.lower, x);
        }
        const auto lo = cell_gradient(item.lower, x);
        const auto hi = cell_gradient(item.upper, x);
        const double weight = cells_[item.upper].width /
            (cells_[item.lower].width + cells_[item.upper].width);
        std::array<double, 3> result;
        for (int d = 0; d < 3; ++d) {
            result[d] = weight * lo[d] + (1.0 - weight) * hi[d];
        }
        result[item.axis] = detail::evaluate_derivative(normal_stencils_[f], x);
        return result;
    }

    double face_value(size_t f, const std::vector<double>& x) const {
        validate_size(x);
        return detail::evaluate_interpolation(value_stencils_.at(f), x);
    }

private:
    // Face keys use normal plane, tangential lower corners and integer width.
    using face_key = std::tuple<int, long long, long long, long long, long long>;
    std::vector<cell> cells_;
    std::vector<face> faces_;
    std::array<double, 3> lower_, upper_;
    std::array<bool, 3> periodic_;
    double minimum_width_ = 0.0;
    std::vector<std::array<sparse_row, 3>> gradients_;
    std::vector<sparse_row> normal_stencils_, value_stencils_;

    void validate_size(const std::vector<double>& x) const {
        if (x.size() != cells_.size()) {
            throw std::invalid_argument("Elliptic vector has the wrong number of cells");
        }
    }

    void build_faces() {
        if (cells_.empty()) {
            throw std::invalid_argument("Elliptic mesh cannot be empty");
        }
        minimum_width_ = std::numeric_limits<double>::max();
        for (const cell& item : cells_) {
            if (!(item.width > 0.0) || !std::isfinite(item.width)) {
                throw std::invalid_argument("Elliptic cell width must be finite and positive");
            }
            minimum_width_ = std::min(minimum_width_, item.width);
        }
        std::array<long long, 3> extent;
        double domain_volume = 1.0;
        for (int d = 0; d < 3; ++d) {
            extent[d] = detail::lattice_coordinate(upper_[d], lower_[d], minimum_width_);
            if (extent[d] <= 0) {
                throw std::invalid_argument("Elliptic domain must have positive extent");
            }
            domain_volume *= upper_[d] - lower_[d];
        }
        std::map<face_key, size_t> low_faces, high_faces;
        std::vector<std::array<long long, 3>> corners(cells_.size());
        std::vector<long long> widths(cells_.size());
        double total_volume = 0.0;
        for (size_t i = 0; i < cells_.size(); ++i) {
            const cell& item = cells_[i];
            const long long width = detail::lattice_coordinate(item.width, 0.0, minimum_width_);
            if (!detail::power_of_two(width)) {
                throw std::invalid_argument("Elliptic cell widths must be dyadic");
            }
            widths[i] = width;
            for (int d = 0; d < 3; ++d) {
                corners[i][d] = detail::lattice_coordinate(item.center[d] - 0.5 * item.width,
                    lower_[d], minimum_width_);
                if (corners[i][d] < 0 || corners[i][d] + width > extent[d]
                    || corners[i][d] % width != 0) {
                    throw std::invalid_argument("Elliptic cell is outside or misaligned with its domain");
                }
            }
            total_volume += volume(i);
            for (int d = 0; d < 3; ++d) {
                const int a = (d + 1) % 3;
                const int b = (d + 2) % 3;
                const face_key lo(d, corners[i][d], corners[i][a], corners[i][b], width);
                const face_key hi(d, corners[i][d] + width, corners[i][a], corners[i][b], width);
                if (!low_faces.emplace(lo, i).second || !high_faces.emplace(hi, i).second) {
                    throw std::invalid_argument("Elliptic mesh contains duplicate cells");
                }
            }
        }
        if (std::abs(total_volume - domain_volume) > 2.0e-10 * domain_volume) {
            throw std::invalid_argument("Elliptic leaves do not cover their domain exactly once");
        }
        std::vector<std::array<double, 6>> covered(cells_.size());
        auto add_shared = [&](size_t lo, size_t hi, int d, long long plane,
                              long long a0, long long b0, long long width) {
            face item;
            item.lower = lo;
            item.upper = hi;
            item.axis = d;
            item.center[d] = lower_[d] + minimum_width_ * static_cast<double>(plane);
            const int a = (d + 1) % 3;
            const int b = (d + 2) % 3;
            item.center[a] = lower_[a] + minimum_width_ * (static_cast<double>(a0) + 0.5 * width);
            item.center[b] = lower_[b] + minimum_width_ * (static_cast<double>(b0) + 0.5 * width);
            const double h = minimum_width_ * static_cast<double>(width);
            item.area = h * h;
            item.distance = 0.5 * (cells_[lo].width + cells_[hi].width);
            faces_.push_back(item);
            covered[lo][2 * d + 1] += item.area;
            covered[hi][2 * d] += item.area;
        };
        auto add_boundary = [&](size_t i, int d, int side) {
            face item;
            item.lower = side > 0 ? i : boundary_index;
            item.upper = side < 0 ? i : boundary_index;
            item.axis = d;
            item.boundary_side = side;
            item.center = cells_[i].center;
            item.center[d] = side < 0 ? lower_[d] : upper_[d];
            item.area = cells_[i].width * cells_[i].width;
            item.distance = 0.5 * cells_[i].width;
            faces_.push_back(item);
            covered[i][2 * d + (side > 0 ? 1 : 0)] += item.area;
        };
        for (const auto& entry : high_faces) {
            const int d = std::get<0>(entry.first);
            const long long plane = std::get<1>(entry.first);
            const long long a0 = std::get<2>(entry.first);
            const long long b0 = std::get<3>(entry.first);
            const long long width = std::get<4>(entry.first);
            const size_t i = entry.second;
            if (plane == extent[d] && !periodic_[d]) {
                add_boundary(i, d, 1);
                continue;
            }
            const long long match_plane = plane == extent[d] ? 0 : plane;
            auto same = low_faces.find(face_key(d, match_plane, a0, b0, width));
            if (same != low_faces.end()) {
                add_shared(i, same->second, d, plane, a0, b0, width);
                continue;
            }
            const long long coarse_width = 2 * width;
            auto coarse = low_faces.find(face_key(d, match_plane,
                (a0 / coarse_width) * coarse_width,
                (b0 / coarse_width) * coarse_width, coarse_width));
            if (coarse != low_faces.end()) {
                add_shared(i, coarse->second, d, plane, a0, b0, width);
                continue;
            }
            if (width < 2) {
                throw std::invalid_argument("Elliptic mesh has a hole, overlap, or non-2:1 face");
            }
            const long long fine_width = width / 2;
            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    auto fine = low_faces.find(face_key(d, match_plane,
                        a0 + a * fine_width, b0 + b * fine_width, fine_width));
                    if (fine == low_faces.end()) {
                        throw std::invalid_argument("Elliptic mesh has a hole, overlap, or non-2:1 face");
                    }
                    add_shared(i, fine->second, d, plane,
                        a0 + a * fine_width, b0 + b * fine_width, fine_width);
                }
            }
        }
        for (size_t i = 0; i < cells_.size(); ++i) {
            for (int d = 0; d < 3; ++d) {
                if (corners[i][d] == 0 && !periodic_[d]) {
                    add_boundary(i, d, -1);
                }
            }
            for (int side = 0; side < 6; ++side) {
                const double area = cells_[i].width * cells_[i].width;
                if (std::abs(covered[i][side] - area) > 1.0e-12 * area) {
                    throw std::invalid_argument("Elliptic mesh face coverage is inconsistent");
                }
            }
        }
    }

    void build_gradients() {
        struct neighbor {
            size_t index;
            std::array<double, 3> displacement;
            double area;
        };
        std::vector<std::vector<neighbor>> neighbors(cells_.size());
        for (const face& item : faces_) {
            if (item.boundary_side != 0 || item.lower == item.upper) {
                continue;
            }
            std::array<double, 3> displacement;
            for (int d = 0; d < 3; ++d) {
                displacement[d] = cells_[item.upper].center[d] - cells_[item.lower].center[d];
            }
            // This also unwraps the displacement at a periodic domain face.
            displacement[item.axis] = item.distance;
            neighbors[item.lower].push_back({item.upper, displacement, item.area});
            for (double& value : displacement) {
                value = -value;
            }
            neighbors[item.upper].push_back({item.lower, displacement, item.area});
        }
        gradients_.resize(cells_.size());
        for (size_t i = 0; i < cells_.size(); ++i) {
            double augmented[3][6] = {};
            for (const neighbor& item : neighbors[i]) {
                double distance_squared = 0.0;
                for (double value : item.displacement) {
                    distance_squared += value * value;
                }
                const double weight = item.area / distance_squared;
                for (int d = 0; d < 3; ++d) {
                    for (int e = 0; e < 3; ++e) {
                        augmented[d][e] += weight * item.displacement[d] * item.displacement[e];
                    }
                }
            }
            double matrix_scale = 0.0;
            for (int d = 0; d < 3; ++d) {
                matrix_scale = std::max(matrix_scale, augmented[d][d]);
                augmented[d][d + 3] = 1.0;
            }
            // On one-cell-wide virtual grids the missing direction has zero
            // slope. The small regularizer only affects absent directions.
            for (int d = 0; d < 3; ++d) {
                if (augmented[d][d] <= 1.0e-14 * matrix_scale) {
                    augmented[d][d] = matrix_scale > 0.0 ? matrix_scale : 1.0;
                }
            }
            for (int d = 0; d < 3; ++d) {
                int pivot = d;
                for (int e = d + 1; e < 3; ++e) {
                    if (std::abs(augmented[e][d]) > std::abs(augmented[pivot][d])) {
                        pivot = e;
                    }
                }
                if (!(std::abs(augmented[pivot][d]) > 0.0)) {
                    throw std::invalid_argument("Elliptic gradient reconstruction is singular");
                }
                for (int e = 0; e < 6; ++e) {
                    std::swap(augmented[d][e], augmented[pivot][e]);
                }
                const double diagonal = augmented[d][d];
                for (int e = 0; e < 6; ++e) {
                    augmented[d][e] /= diagonal;
                }
                for (int e = 0; e < 3; ++e) {
                    if (e != d) {
                        const double scale = augmented[e][d];
                        for (int k = 0; k < 6; ++k) {
                            augmented[e][k] -= scale * augmented[d][k];
                        }
                    }
                }
            }
            for (int d = 0; d < 3; ++d) {
                std::map<size_t, double> stencil;
                for (const neighbor& item : neighbors[i]) {
                    double distance_squared = 0.0;
                    double coefficient = 0.0;
                    for (int e = 0; e < 3; ++e) {
                        distance_squared += item.displacement[e] * item.displacement[e];
                        coefficient += augmented[d][e + 3] * item.displacement[e];
                    }
                    coefficient *= item.area / distance_squared;
                    stencil[item.index] += coefficient;
                    stencil[i] -= coefficient;
                }
                gradients_[i][d] = detail::compact(stencil);
            }
        }
    }

    sparse_row trace(size_t i, const face& item) const {
        std::map<size_t, double> result;
        result[i] = 1.0;
        for (int d = 0; d < 3; ++d) {
            if (d != item.axis) {
                detail::accumulate(result, gradients_[i][d], item.center[d] - cells_[i].center[d]);
            }
        }
        return detail::compact(result);
    }

    void build_face_stencils() {
        normal_stencils_.reserve(faces_.size());
        value_stencils_.reserve(faces_.size());
        for (const face& item : faces_) {
            if (item.boundary_side != 0) {
                const size_t i = item.lower == boundary_index ? item.upper : item.lower;
                normal_stencils_.push_back(gradients_[i][item.axis]);
                value_stencils_.push_back({{i, 1.0}});
                continue;
            }
            const sparse_row lo = trace(item.lower, item);
            const sparse_row hi = trace(item.upper, item);
            std::map<size_t, double> gradient, value;
            detail::accumulate(gradient, lo, -1.0 / item.distance);
            detail::accumulate(gradient, hi, 1.0 / item.distance);
            const double weight = cells_[item.upper].width /
                (cells_[item.lower].width + cells_[item.upper].width);
            detail::accumulate(value, lo, weight);
            detail::accumulate(value, hi, 1.0 - weight);
            normal_stencils_.push_back(detail::compact(gradient));
            value_stencils_.push_back(detail::compact(value));
        }
    }
};

// alpha E_boundary + gamma beta dE/dn = value. Thus a prescribed
// OUTWARD flux q is {0,1,-q}; homogeneous Neumann is the default.
struct boundary_condition {
    double alpha = 0.0;
    double gamma = 1.0;
    double value = 0.0;
};

class scalar_operator {
public:
    scalar_operator(const composite_mesh& mesh, const std::vector<double>& reaction,
        const std::vector<double>& face_beta,
        const std::vector<boundary_condition>& boundaries = {})
      : mesh_(mesh), reaction_(reaction), beta_(face_beta), matrix_(mesh.cells().size()),
        boundary_rhs_(mesh.cells().size(), 0.0), boundary_diagonal_(mesh.faces().size(), 0.0),
        boundary_flux_(mesh.faces().size(), 0.0) {
        if (reaction.size() != mesh.cells().size() || face_beta.size() != mesh.faces().size()) {
            throw std::invalid_argument("Elliptic operator coefficients have the wrong size");
        }
        if (!boundaries.empty() && boundaries.size() != 6 && boundaries.size() != mesh.faces().size()) {
            throw std::invalid_argument("Elliptic boundary data must have six or face-count entries");
        }
        std::vector<std::map<size_t, double>> rows(mesh.cells().size());
        nullspace_ = true;
        for (size_t i = 0; i < reaction.size(); ++i) {
            if (!std::isfinite(reaction[i]) || reaction[i] < 0.0) {
                throw std::invalid_argument("Elliptic reaction must be finite and nonnegative");
            }
            rows[i][i] = reaction[i] * mesh.volume(i);
            nullspace_ = nullspace_ && reaction[i] == 0.0;
        }
        for (size_t f = 0; f < mesh.faces().size(); ++f) {
            const face& item = mesh.faces()[f];
            const double beta = face_beta[f];
            if (!std::isfinite(beta) || beta < 0.0) {
                throw std::invalid_argument("Elliptic face coefficient must be finite and nonnegative");
            }
            if (item.boundary_side == 0) {
                detail::accumulate(rows[item.lower], mesh.face_normal_stencil(f), -beta * item.area);
                detail::accumulate(rows[item.upper], mesh.face_normal_stencil(f), beta * item.area);
            } else {
                const boundary_condition bc = boundaries.empty() ? boundary_condition{} :
                    boundaries[boundaries.size() == 6 ?
                        static_cast<size_t>(2 * item.axis + (item.boundary_side > 0 ? 1 : 0)) : f];
                if (!(bc.alpha >= 0.0) || !(bc.gamma >= 0.0) ||
                    !std::isfinite(bc.alpha) || !std::isfinite(bc.gamma) ||
                    !std::isfinite(bc.value) || !(bc.alpha > 0.0 || bc.gamma > 0.0)) {
                    throw std::invalid_argument("Elliptic Robin data require finite nonnegative alpha/gamma");
                }
                const size_t i = item.lower == boundary_index ? item.upper : item.lower;
                const double transmission = beta / item.distance;
                double diagonal = 0.0;
                double forcing = 0.0;
                if (bc.alpha == 0.0) {
                    forcing = bc.value / bc.gamma;
                } else if (transmission > 0.0) {
                    const double denominator = bc.alpha + bc.gamma * transmission;
                    diagonal = transmission * bc.alpha / denominator;
                    forcing = transmission * bc.value / denominator;
                }
                rows[i][i] += item.area * diagonal;
                boundary_rhs_[i] += item.area * forcing / mesh.volume(i);
                boundary_diagonal_[f] = diagonal;
                boundary_flux_[f] = -forcing;
                nullspace_ = nullspace_ && diagonal == 0.0;
            }
        }
        for (size_t i = 0; i < rows.size(); ++i) {
            for (auto& item : rows[i]) {
                item.second /= mesh.volume(i);
                if (!std::isfinite(item.second)) {
                    throw std::invalid_argument("Elliptic matrix coefficient overflowed");
                }
            }
            matrix_[i] = detail::compact(rows[i]);
        }
    }

    const composite_mesh& mesh() const { return mesh_; }
    const sparse_matrix& matrix() const { return matrix_; }
    const std::vector<double>& boundary_rhs() const { return boundary_rhs_; }
    bool has_constant_nullspace() const { return nullspace_; }

    // Homogeneous linear operator; solve() adds boundary_rhs() to the RHS.
    std::vector<double> apply(const std::vector<double>& x) const {
        check_size(x);
        std::vector<double> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = reaction_[i] * x[i];
        }
        for (size_t f = 0; f < mesh_.faces().size(); ++f) {
            const face& item = mesh_.faces()[f];
            if (item.boundary_side == 0) {
                const double integrated_flux = -beta_[f] * item.area *
                    detail::evaluate_derivative(mesh_.face_normal_stencil(f), x);
                result[item.lower] += integrated_flux / mesh_.volume(item.lower);
                result[item.upper] -= integrated_flux / mesh_.volume(item.upper);
            } else {
                const size_t i = item.lower == boundary_index ? item.upper : item.lower;
                result[i] += item.area * boundary_diagonal_[f] * x[i] / mesh_.volume(i);
            }
        }
        return result;
    }

    // Flux per unit area: positive axis on interior faces, outward on
    // boundary faces. Boundary forcing is INCLUDED in these physical fluxes.
    std::vector<double> face_fluxes(const std::vector<double>& x) const {
        check_size(x);
        std::vector<double> result(mesh_.faces().size());
        for (size_t f = 0; f < result.size(); ++f) {
            const face& item = mesh_.faces()[f];
            if (item.boundary_side == 0) {
                result[f] = -beta_[f] * detail::evaluate_derivative(mesh_.face_normal_stencil(f), x);
            } else {
                const size_t i = item.lower == boundary_index ? item.upper : item.lower;
                result[f] = boundary_diagonal_[f] * x[i] + boundary_flux_[f];
            }
        }
        return result;
    }

private:
    const composite_mesh& mesh_;
    std::vector<double> reaction_, beta_;
    sparse_matrix matrix_;
    std::vector<double> boundary_rhs_, boundary_diagonal_, boundary_flux_;
    bool nullspace_;

    void check_size(const std::vector<double>& x) const {
        if (x.size() != matrix_.size()) {
            throw std::invalid_argument("Elliptic vector has the wrong size");
        }
    }
};

struct solver_options {
    double relative_tolerance = 1.0e-10;
    double absolute_tolerance = 1.0e-13;
    int maximum_iterations = 200;
    int restart = 30;
    int pre_smoothing = 3;
    int post_smoothing = 3;
    double smoothing_weight = 0.7;
};

struct solve_report {
    bool converged = false;
    int iterations = 0;
    int vcycles = 0;
    int levels = 0;
    double absolute_residual = std::numeric_limits<double>::infinity();
    double relative_residual = std::numeric_limits<double>::infinity();
    std::string reason;
};

namespace detail {

inline double dot(const std::vector<double>& x, const std::vector<double>& y,
    const std::vector<double>& volume) {
    long double result = 0.0;
    long double total_volume = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        result += static_cast<long double>(volume[i]) * x[i] * y[i];
        total_volume += volume[i];
    }
    return static_cast<double>(result / total_volume);
}

inline double norm(const std::vector<double>& x, const std::vector<double>& volume) {
    double scale = 0.0;
    for (double value : x) {
        if (!std::isfinite(value)) {
            return std::numeric_limits<double>::infinity();
        }
        scale = std::max(scale, std::abs(value));
    }
    if (scale == 0.0) {
        return 0.0;
    }
    long double sum = 0.0;
    long double total_volume = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const long double value = x[i] / scale;
        sum += volume[i] * value * value;
        total_volume += volume[i];
    }
    return scale * static_cast<double>(std::sqrt(sum / total_volume));
}

inline void mean_zero(std::vector<double>& x, const std::vector<double>& volume) {
    double total = 0.0;
    double total_volume = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        total += volume[i] * x[i];
        total_volume += volume[i];
    }
    const double mean = total / total_volume;
    for (double& value : x) {
        value -= mean;
    }
}

struct multigrid_level {
    sparse_matrix matrix;
    std::vector<double> volume;
    std::vector<double> diagonal;
    sparse_matrix prolongation;
    std::vector<size_t> parent;
};

// Restriction is volume averaging into geometric parents. Prolongation is
// linear reconstruction with signed corrections. R A P defines every coarse
// operator, preserving the fine interface fluxes and coefficient jumps.
class multigrid {
public:
    multigrid(const scalar_operator& op, const solver_options& options)
      : options_(options), nullspace_(op.has_constant_nullspace()) {
        composite_mesh mesh = op.mesh();
        sparse_matrix matrix = op.matrix();
        while (true) {
            multigrid_level current;
            current.matrix = std::move(matrix);
            current.volume.resize(mesh.cells().size());
            current.diagonal.resize(mesh.cells().size(), 0.0);
            for (size_t i = 0; i < mesh.cells().size(); ++i) {
                current.volume[i] = mesh.volume(i);
                for (const auto& item : current.matrix[i]) {
                    if (item.first == i) {
                        current.diagonal[i] = item.second;
                    }
                }
            }
            const double h = 2.0 * mesh.minimum_width();
            bool can_coarsen = mesh.cells().size() > 8;
            for (int d = 0; d < 3; ++d) {
                const double count = (mesh.upper()[d] - mesh.lower()[d]) / h;
                can_coarsen = can_coarsen && count >= 1.0 &&
                    std::abs(count - std::round(count)) < 1.0e-7;
            }
            if (!can_coarsen) {
                if (mesh.cells().size() > 512) {
                    throw std::invalid_argument("Elliptic mesh cannot be coarsened to a small direct solve");
                }
                levels_.push_back(std::move(current));
                break;
            }
            using key = std::tuple<long long, long long, long long, long long>;
            std::map<key, size_t> coarse_indices;
            std::vector<cell> coarse_cells;
            current.parent.resize(mesh.cells().size());
            for (size_t i = 0; i < mesh.cells().size(); ++i) {
                const cell& fine = mesh.cells()[i];
                cell coarse;
                coarse.width = std::max(h, fine.width);
                std::array<long long, 3> position;
                for (int d = 0; d < 3; ++d) {
                    position[d] = static_cast<long long>(std::floor(
                        (fine.center[d] - mesh.lower()[d]) / coarse.width));
                    coarse.center[d] = mesh.lower()[d] +
                        (static_cast<double>(position[d]) + 0.5) * coarse.width;
                }
                const key identity(position[0], position[1], position[2],
                    detail::lattice_coordinate(coarse.width, 0.0, h));
                auto found = coarse_indices.find(identity);
                if (found == coarse_indices.end()) {
                    const size_t index = coarse_cells.size();
                    coarse_cells.push_back(coarse);
                    found = coarse_indices.emplace(identity, index).first;
                }
                current.parent[i] = found->second;
            }
            composite_mesh coarse_mesh(std::move(coarse_cells), mesh.lower(), mesh.upper(), mesh.periodic());
            current.prolongation.resize(mesh.cells().size());
            for (size_t i = 0; i < mesh.cells().size(); ++i) {
                const size_t p = current.parent[i];
                std::map<size_t, double> interpolation;
                interpolation[p] = 1.0;
                for (int d = 0; d < 3; ++d) {
                    accumulate(interpolation, coarse_mesh.gradient_stencil(p, d),
                        mesh.cells()[i].center[d] - coarse_mesh.cells()[p].center[d]);
                }
                current.prolongation[i] = compact(interpolation);
            }
            std::vector<std::map<size_t, double>> coarse_rows(coarse_mesh.cells().size());
            for (size_t i = 0; i < mesh.cells().size(); ++i) {
                const size_t p = current.parent[i];
                const double weight = mesh.volume(i) / coarse_mesh.volume(p);
                for (const auto& item : current.matrix[i]) {
                    accumulate(coarse_rows[p], current.prolongation[item.first], weight * item.second);
                }
            }
            matrix.resize(coarse_rows.size());
            for (size_t i = 0; i < coarse_rows.size(); ++i) {
                matrix[i] = compact(coarse_rows[i]);
            }
            levels_.push_back(std::move(current));
            mesh = std::move(coarse_mesh);
        }
        factor_bottom();
    }

    int level_count() const { return static_cast<int>(levels_.size()); }
    const std::vector<double>& volume() const { return levels_.front().volume; }

    std::vector<double> apply(const std::vector<double>& rhs) const {
        std::vector<double> result(rhs.size(), 0.0);
        cycle(0, rhs, result);
        return result;
    }

private:
    solver_options options_;
    bool nullspace_;
    std::vector<multigrid_level> levels_;
    std::vector<std::vector<double>> bottom_lu_;
    std::vector<size_t> bottom_pivots_;

    void factor_bottom() {
        const auto& bottom = levels_.back();
        const size_t n = bottom.matrix.size();
        bottom_lu_.assign(n, std::vector<double>(n, 0.0));
        bottom_pivots_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            for (const auto& item : bottom.matrix[i]) {
                bottom_lu_[i][item.first] = item.second;
            }
        }
        if (nullspace_) {
            const double sum = std::accumulate(bottom.volume.begin(), bottom.volume.end(), 0.0);
            for (size_t j = 0; j < n; ++j) {
                bottom_lu_[0][j] = bottom.volume[j] / sum;
            }
        }
        for (size_t k = 0; k < n; ++k) {
            size_t pivot = k;
            for (size_t i = k + 1; i < n; ++i) {
                if (std::abs(bottom_lu_[i][k]) > std::abs(bottom_lu_[pivot][k])) {
                    pivot = i;
                }
            }
            bottom_pivots_[k] = pivot;
            if (!(std::abs(bottom_lu_[pivot][k]) > std::numeric_limits<double>::min())) {
                throw std::invalid_argument("Elliptic bottom operator is singular");
            }
            std::swap(bottom_lu_[k], bottom_lu_[pivot]);
            for (size_t i = k + 1; i < n; ++i) {
                bottom_lu_[i][k] /= bottom_lu_[k][k];
                for (size_t j = k + 1; j < n; ++j) {
                    bottom_lu_[i][j] -= bottom_lu_[i][k] * bottom_lu_[k][j];
                }
            }
        }
    }

    void direct_solve(const std::vector<double>& rhs, std::vector<double>& x) const {
        x = rhs;
        if (nullspace_) {
            x[0] = 0.0;
        }
        const size_t n = x.size();
        // Apply all pivot permutations before triangular substitution.
        for (size_t k = 0; k < n; ++k) {
            std::swap(x[k], x[bottom_pivots_[k]]);
        }
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < i; ++j) {
                x[i] -= bottom_lu_[i][j] * x[j];
            }
        }
        for (size_t k = n; k-- > 0;) {
            for (size_t j = k + 1; j < n; ++j) {
                x[k] -= bottom_lu_[k][j] * x[j];
            }
            x[k] /= bottom_lu_[k][k];
        }
        if (nullspace_) {
            mean_zero(x, levels_.back().volume);
        }
    }

    void smooth(const multigrid_level& level, const std::vector<double>& rhs,
        std::vector<double>& x, int sweeps, bool reverse) const {
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            for (size_t k = 0; k < x.size(); ++k) {
                const size_t i = reverse ? x.size() - k - 1 : k;
                if (level.diagonal[i] == 0.0) {
                    throw std::invalid_argument("Elliptic smoother encountered a zero diagonal");
                }
                x[i] += options_.smoothing_weight *
                    (rhs[i] - evaluate(level.matrix[i], x)) / level.diagonal[i];
            }
            if (nullspace_) {
                mean_zero(x, level.volume);
            }
        }
    }

    void cycle(size_t index, const std::vector<double>& rhs, std::vector<double>& x) const {
        if (index + 1 == levels_.size()) {
            direct_solve(rhs, x);
            return;
        }
        const auto& level = levels_[index];
        const auto& coarse = levels_[index + 1];
        smooth(level, rhs, x, options_.pre_smoothing, false);
        const auto applied = multiply(level.matrix, x);
        std::vector<double> coarse_rhs(coarse.matrix.size(), 0.0);
        for (size_t i = 0; i < x.size(); ++i) {
            const size_t p = level.parent[i];
            coarse_rhs[p] += level.volume[i] / coarse.volume[p] * (rhs[i] - applied[i]);
        }
        if (nullspace_) {
            mean_zero(coarse_rhs, coarse.volume);
        }
        std::vector<double> correction(coarse_rhs.size(), 0.0);
        cycle(index + 1, coarse_rhs, correction);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += evaluate_interpolation(level.prolongation[i], correction);
        }
        smooth(level, rhs, x, options_.post_smoothing, true);
    }
};

} // namespace detail

// Restarted right-preconditioned GMRES. Tangential coarse-face reconstruction
// and R A P need not be symmetric, so CG assumptions are deliberately absent.
inline solve_report solve(const scalar_operator& op, const std::vector<double>& density_rhs,
    std::vector<double>& x, const solver_options& options = {}) {
    if (density_rhs.size() != op.mesh().cells().size() ||
        x.size() != density_rhs.size()) {
        throw std::invalid_argument("Elliptic solve vectors have the wrong size");
    }
    if (!std::isfinite(options.relative_tolerance) || !std::isfinite(options.absolute_tolerance) ||
        !(options.relative_tolerance >= 0.0) || !(options.absolute_tolerance >= 0.0) ||
        options.maximum_iterations < 1 || options.restart < 1 ||
        options.pre_smoothing < 1 || options.post_smoothing < 1 ||
        !(options.smoothing_weight > 0.0 && options.smoothing_weight <= 1.0)) {
        throw std::invalid_argument("Invalid elliptic solver options");
    }
    solve_report report;
    std::vector<double> rhs = density_rhs;
    std::vector<double> volumes(rhs.size());
    double integral = 0.0;
    double magnitude = 0.0;
    for (size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] += op.boundary_rhs()[i];
        volumes[i] = op.mesh().volume(i);
        if (!std::isfinite(rhs[i]) || !std::isfinite(x[i])) {
            report.reason = "nonfinite linear input";
            return report;
        }
        integral += volumes[i] * rhs[i];
        magnitude += volumes[i] * std::abs(rhs[i]);
    }
    if (op.has_constant_nullspace()) {
        if (std::abs(integral) > 128.0 * std::numeric_limits<double>::epsilon() * magnitude) {
            report.reason = "incompatible periodic/Neumann Poisson right-hand side";
            return report;
        }
        detail::mean_zero(rhs, volumes);
        detail::mean_zero(x, volumes);
    }
    const double rhs_norm = detail::norm(rhs, volumes);
    const double target = options.absolute_tolerance + options.relative_tolerance * rhs_norm;
    auto residual = [&]() {
        auto result = op.apply(x);
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = rhs[i] - result[i];
        }
        return result;
    };
    auto update_report = [&](const std::vector<double>& r) {
        report.absolute_residual = detail::norm(r, volumes);
        report.relative_residual = report.absolute_residual / std::max(rhs_norm, 1.0e-300);
        report.converged = std::isfinite(report.absolute_residual) && report.absolute_residual <= target;
    };
    auto r = residual();
    update_report(r);
    if (report.converged) {
        report.reason = "initial residual satisfies tolerance";
        return report;
    }
    detail::multigrid preconditioner(op, options);
    report.levels = preconditioner.level_count();
    const int restart = std::min(options.restart, options.maximum_iterations);
    while (report.iterations < options.maximum_iterations) {
        const double beta = detail::norm(r, volumes);
        if (!(beta > 0.0) || !std::isfinite(beta)) {
            report.reason = "nonfinite or degenerate Krylov residual";
            break;
        }
        std::vector<std::vector<double>> basis(1, r), preconditioned;
        for (double& value : basis[0]) {
            value /= beta;
        }
        std::vector<std::vector<double>> hessenberg(restart + 1, std::vector<double>(restart, 0.0));
        std::vector<double> cosine(restart, 0.0), sine(restart, 0.0), projected(restart + 1, 0.0);
        projected[0] = beta;
        int steps = 0;
        for (int j = 0; j < restart && report.iterations < options.maximum_iterations; ++j) {
            preconditioned.push_back(preconditioner.apply(basis[j]));
            ++report.vcycles;
            auto w = op.apply(preconditioned.back());
            // Two modified Gram-Schmidt passes keep tight residual tolerances
            // reliable for ill-scaled reaction/diffusion coefficients.
            for (int pass = 0; pass < 2; ++pass) {
                for (int i = 0; i <= j; ++i) {
                    const double projection = detail::dot(w, basis[i], volumes);
                    hessenberg[i][j] += projection;
                    for (size_t k = 0; k < w.size(); ++k) {
                        w[k] -= projection * basis[i][k];
                    }
                }
            }
            hessenberg[j + 1][j] = detail::norm(w, volumes);
            const bool breakdown = !(hessenberg[j + 1][j] > std::numeric_limits<double>::min());
            if (!breakdown) {
                for (double& value : w) {
                    value /= hessenberg[j + 1][j];
                }
                basis.push_back(std::move(w));
            }
            for (int i = 0; i < j; ++i) {
                const double a = hessenberg[i][j];
                const double b = hessenberg[i + 1][j];
                hessenberg[i][j] = cosine[i] * a + sine[i] * b;
                hessenberg[i + 1][j] = -sine[i] * a + cosine[i] * b;
            }
            const double diagonal = std::hypot(hessenberg[j][j], hessenberg[j + 1][j]);
            if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
                report.reason = "Krylov Hessenberg breakdown";
                break;
            }
            cosine[j] = hessenberg[j][j] / diagonal;
            sine[j] = hessenberg[j + 1][j] / diagonal;
            hessenberg[j][j] = diagonal;
            hessenberg[j + 1][j] = 0.0;
            projected[j + 1] = -sine[j] * projected[j];
            projected[j] *= cosine[j];
            ++steps;
            ++report.iterations;
            if (std::abs(projected[j + 1]) <= target || breakdown) {
                break;
            }
        }
        if (steps == 0) {
            break;
        }
        std::vector<double> weights(projected.begin(), projected.begin() + steps);
        for (int j = steps; j-- > 0;) {
            for (int k = j + 1; k < steps; ++k) {
                weights[j] -= hessenberg[j][k] * weights[k];
            }
            weights[j] /= hessenberg[j][j];
        }
        for (int j = 0; j < steps; ++j) {
            for (size_t k = 0; k < x.size(); ++k) {
                x[k] += weights[j] * preconditioned[j][k];
            }
        }
        if (op.has_constant_nullspace()) {
            detail::mean_zero(x, volumes);
        }
        r = residual();
        update_report(r);
        if (report.converged) {
            report.reason = "converged";
            return report;
        }
        if (!std::isfinite(report.absolute_residual)) {
            report.reason = "nonfinite linear residual";
            return report;
        }
    }
    if (report.reason.empty()) {
        report.reason = "maximum linear iterations reached";
    }
    return report;
}

} // namespace elliptic
} // namespace octotiger
