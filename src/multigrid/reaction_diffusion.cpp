// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "octotiger/multigrid/reaction_diffusion.hpp"

namespace octotiger::multigrid {
namespace {

std::array<int, NDIM> childOrigin(int n, int octant) {
    assert(octant >= 0 && octant < NCHILD);
    return {{1 + (octant & 1) * (n / 2), 1 + ((octant >> 1) & 1) * (n / 2),
        1 + ((octant >> 2) & 1) * (n / 2)}};
}

double wallDiagonal(ReactionDiffusionBlock const& block, int i, int j, int k) {
    std::array<int, NDIM> const cell{{i, j, k}};
    double value = 0.0;
    for (int axis = 0; axis < NDIM; ++axis) {
        if (cell[axis] == 1) {
            value += block.wallConductance[axis](i, j, k);
        }
        if (cell[axis] == block.n) {
            auto face = cell;
            ++face[axis];
            value += block.wallConductance[axis](face[0], face[1], face[2]);
        }
    }
    return value / block.dx;
}

double localDiffusion(ReactionDiffusionBlock const& block, int i, int j, int k) {
    std::array<int, NDIM> const cell{{i, j, k}};
    double const center = block.solution(i, j, k);
    double value = 0.0;
    for (int axis = 0; axis < NDIM; ++axis) {
        auto lo = cell;
        auto hi = cell;
        --lo[axis];
        ++hi[axis];
        value += (block.diffusion[axis](i, j, k) *
                     (center - block.solution(lo[0], lo[1], lo[2])) +
                     block.diffusion[axis](hi[0], hi[1], hi[2]) *
                         (center - block.solution(hi[0], hi[1], hi[2]))) /
            (block.dx * block.dx);
    }
    return value + wallDiagonal(block, i, j, k) * center;
}

}    // namespace

ReactionDiffusionBlock::ReactionDiffusionBlock(int interiorSize, double cellWidth)
  : n(interiorSize)
  , dx(cellWidth)
  , solution(n)
  , rhs(n)
  , reaction(n)
  , residual(n)
  , saved(n)
  , fluxDefect(n) {
    for (int axis = 0; axis < NDIM; ++axis) {
        diffusion[axis] = Field(n);
        wallConductance[axis] = Field(n);
        flux[axis] = Field(n);
    }
    if (!(dx > 0.0) || !std::isfinite(dx)) {
        throw std::invalid_argument("Reaction-diffusion cell width must be finite and positive");
    }
}

double ReactionDiffusionBlock::applyLocal(int i, int j, int k) const {
    return reaction(i, j, k) * solution(i, j, k) + localDiffusion(*this, i, j, k);
}

double ReactionDiffusionBlock::fluxDivergence(int i, int j, int k) const {
    return (flux[0](i + 1, j, k) - flux[0](i, j, k) +
               flux[1](i, j + 1, k) - flux[1](i, j, k) +
               flux[2](i, j, k + 1) - flux[2](i, j, k)) /
        dx;
}

double ReactionDiffusionBlock::diagonal(int i, int j, int k) const {
    return reaction(i, j, k) +
        (diffusion[0](i, j, k) + diffusion[0](i + 1, j, k) +
            diffusion[1](i, j, k) + diffusion[1](i, j + 1, k) +
            diffusion[2](i, j, k) + diffusion[2](i, j, k + 1)) /
        (dx * dx) + wallDiagonal(*this, i, j, k);
}

void ReactionDiffusionBlock::computeFluxes() {
    for (int axis = 0; axis < NDIM; ++axis) {
        int const a = (axis + 1) % NDIM;
        int const b = (axis + 2) % NDIM;
        for (int f = 1; f <= n + 1; ++f) {
            for (int j = 1; j <= n; ++j) {
                for (int k = 1; k <= n; ++k) {
                    std::array<int, NDIM> hi{};
                    hi[axis] = f;
                    hi[a] = j;
                    hi[b] = k;
                    auto lo = hi;
                    --lo[axis];
                    double const coefficient = diffusion[axis](hi[0], hi[1], hi[2]);
                    if (!(coefficient >= 0.0) || !std::isfinite(coefficient)) {
                        throw std::runtime_error("Invalid reaction-diffusion face coefficient");
                    }
                    flux[axis](hi[0], hi[1], hi[2]) = -coefficient *
                        (solution(hi[0], hi[1], hi[2]) - solution(lo[0], lo[1], lo[2])) / dx;
                    double const conductance = wallConductance[axis](hi[0], hi[1], hi[2]);
                    if (!(conductance >= 0.0) || !std::isfinite(conductance) ||
                        (f != 1 && f != n + 1 && conductance != 0.0)) {
                        throw std::runtime_error("Invalid reaction-diffusion wall conductance");
                    }
                    if (f == 1) {
                        flux[axis](hi[0], hi[1], hi[2]) -=
                            conductance * solution(hi[0], hi[1], hi[2]);
                    } else if (f == n + 1) {
                        flux[axis](hi[0], hi[1], hi[2]) +=
                            conductance * solution(lo[0], lo[1], lo[2]);
                    }
                }
            }
        }
    }
}

void ReactionDiffusionBlock::updateFluxDefect() {
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            for (int k = 1; k <= n; ++k) {
                // The local operator includes wall leakage already; only
                // reconciled interface fluxes require a physical correction.
                fluxDefect(i, j, k) = fluxDivergence(i, j, k) -
                    localDiffusion(*this, i, j, k);
            }
        }
    }
}

void ReactionDiffusionBlock::computeResidual() {
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            for (int k = 1; k <= n; ++k) {
                residual(i, j, k) = rhs(i, j, k) -
                    reaction(i, j, k) * solution(i, j, k) - fluxDivergence(i, j, k);
            }
        }
    }
}

void ReactionDiffusionBlock::relax(double omega) {
    if (!(omega > 0.0 && omega <= 1.0) || !std::isfinite(omega)) {
        throw std::invalid_argument("Reaction-diffusion relaxation weight must be in (0,1]");
    }
    for (int color = 0; color < 2; ++color) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                for (int k = (i + j + color) % 2 + 1; k <= n; k += 2) {
                    double const d = diagonal(i, j, k);
                    if (!(d > 0.0) || !std::isfinite(d) || !(reaction(i, j, k) >= 0.0)) {
                        throw std::runtime_error("Invalid reaction-diffusion diagonal");
                    }
                    solution(i, j, k) += omega *
                        (rhs(i, j, k) - fluxDefect(i, j, k) - applyLocal(i, j, k)) / d;
                }
            }
        }
    }
}

void ReactionDiffusionBlock::saveSolution() {
    saved.values = solution.values;
}

void ReactionDiffusionBlock::formCoarseRhs(Field const& restrictedResidual) {
    assert(restrictedResidual.n == n);
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            for (int k = 1; k <= n; ++k) {
                rhs(i, j, k) = reaction(i, j, k) * solution(i, j, k) +
                    fluxDivergence(i, j, k) + restrictedResidual(i, j, k);
            }
        }
    }
}

std::vector<double> ReactionDiffusionBlock::restrictField(Field const& field) const {
    assert(field.n == n);
    std::vector<double> values;
    values.reserve(static_cast<size_t>(n / 2) * (n / 2) * (n / 2));
    for (int i = 1; i <= n; i += 2) {
        for (int j = 1; j <= n; j += 2) {
            for (int k = 1; k <= n; k += 2) {
                values.push_back(field.octetAverage(i, j, k));
            }
        }
    }
    return values;
}

std::vector<double> ReactionDiffusionBlock::childValues(Field const& field, int octant) const {
    assert(field.n == n);
    auto const origin = childOrigin(n, octant);
    std::vector<double> values;
    values.reserve(static_cast<size_t>(n / 2) * (n / 2) * (n / 2));
    for (int i = 0; i < n / 2; ++i) {
        for (int j = 0; j < n / 2; ++j) {
            for (int k = 0; k < n / 2; ++k) {
                values.push_back(field(origin[0] + i, origin[1] + j, origin[2] + k));
            }
        }
    }
    return values;
}

std::vector<double> ReactionDiffusionBlock::childCorrection(int octant) const {
    auto values = childValues(solution, octant);
    auto const original = childValues(saved, octant);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] -= original[i];
    }
    return values;
}

void ReactionDiffusionBlock::setChild(
    Field& field, int octant, std::vector<double> const& values) const {
    assert(field.n == n);
    assert(values.size() == static_cast<size_t>(n / 2) * (n / 2) * (n / 2));
    auto const origin = childOrigin(n, octant);
    size_t p = 0;
    for (int i = 0; i < n / 2; ++i) {
        for (int j = 0; j < n / 2; ++j) {
            for (int k = 0; k < n / 2; ++k) {
                field(origin[0] + i, origin[1] + j, origin[2] + k) = values[p++];
            }
        }
    }
}

void ReactionDiffusionBlock::addCoarseCorrection(std::vector<double> const& values) {
    assert(values.size() == static_cast<size_t>(n / 2) * (n / 2) * (n / 2));
    size_t p = 0;
    for (int i = 1; i <= n; i += 2) {
        for (int j = 1; j <= n; j += 2) {
            for (int k = 1; k <= n; k += 2) {
                double const increment = values[p++];
                for (int di = 0; di < 2; ++di) {
                    for (int dj = 0; dj < 2; ++dj) {
                        for (int dk = 0; dk < 2; ++dk) {
                            solution(i + di, j + dj, k + dk) += increment;
                        }
                    }
                }
            }
        }
    }
}

std::vector<double> ReactionDiffusionBlock::coarseFaceFluxes(int axis, int side) const {
    assert(axis >= 0 && axis < NDIM && side >= 0 && side < 2);
    int const face = side == 0 ? 1 : n + 1;
    int const a = (axis + 1) % NDIM;
    int const b = (axis + 2) % NDIM;
    std::vector<double> values;
    for (int j = 1; j <= n; j += 2) {
        for (int k = 1; k <= n; k += 2) {
            double sum = 0.0;
            for (int dj = 0; dj < 2; ++dj) {
                for (int dk = 0; dk < 2; ++dk) {
                    std::array<int, NDIM> cell{};
                    cell[axis] = face;
                    cell[a] = j + dj;
                    cell[b] = k + dk;
                    sum += flux[axis](cell[0], cell[1], cell[2]);
                }
            }
            values.push_back(sum / (NCHILD / 2));
        }
    }
    return values;
}

void ReactionDiffusionBlock::setRestrictedFlux(int axis, int faceIndex, int t0, int t1,
    std::vector<double> const& values) {
    assert(axis >= 0 && axis < NDIM && (faceIndex == 1 || faceIndex == n + 1));
    assert(t0 >= 0 && t0 <= n / 2 && t1 >= 0 && t1 <= n / 2);
    assert(values.size() == static_cast<size_t>(n / 2) * (n / 2));
    int const a = (axis + 1) % NDIM;
    int const b = (axis + 2) % NDIM;
    size_t p = 0;
    for (int j = 1; j <= n / 2; ++j) {
        for (int k = 1; k <= n / 2; ++k) {
            std::array<int, NDIM> cell{};
            cell[axis] = faceIndex;
            cell[a] = j + t0;
            cell[b] = k + t1;
            flux[axis](cell[0], cell[1], cell[2]) = values[p++];
        }
    }
}

namespace {

std::vector<double> restrictFaceField(Field const& field, int axis) {
    assert(axis >= 0 && axis < NDIM);
    int const n = field.n;
    int const a = (axis + 1) % NDIM;
    int const b = (axis + 2) % NDIM;
    std::vector<double> values;
    for (int j = 1; j <= n; j += 2) {
        for (int k = 1; k <= n; k += 2) {
            for (int face = 1; face <= n + 1; face += 2) {
                double sum = 0.0;
                for (int dj = 0; dj < 2; ++dj) {
                    for (int dk = 0; dk < 2; ++dk) {
                        std::array<int, NDIM> cell{};
                        cell[axis] = face;
                        cell[a] = j + dj;
                        cell[b] = k + dk;
                        sum += field(cell[0], cell[1], cell[2]);
                    }
                }
                values.push_back(sum / (NCHILD / 2));
            }
        }
    }
    return values;
}

void setChildFaceField(Field& field, int axis, int octant, std::vector<double> const& values) {
    assert(axis >= 0 && axis < NDIM);
    int const n = field.n;
    assert(values.size() == static_cast<size_t>(n / 2) * (n / 2) * (n / 2 + 1));
    auto const origin = childOrigin(n, octant);
    int const a = (axis + 1) % NDIM;
    int const b = (axis + 2) % NDIM;
    size_t p = 0;
    for (int j = 0; j < n / 2; ++j) {
        for (int k = 0; k < n / 2; ++k) {
            for (int face = 0; face <= n / 2; ++face) {
                auto cell = origin;
                cell[axis] += face;
                cell[a] += j;
                cell[b] += k;
                field(cell[0], cell[1], cell[2]) = values[p++];
            }
        }
    }
}

}    // namespace

std::vector<double> ReactionDiffusionBlock::restrictFaceCoefficients(int axis) const {
    assert(axis >= 0 && axis < NDIM);
    return restrictFaceField(diffusion[axis], axis);
}

void ReactionDiffusionBlock::setChildFaceCoefficients(
    int axis, int octant, std::vector<double> const& values) {
    assert(axis >= 0 && axis < NDIM);
    setChildFaceField(diffusion[axis], axis, octant, values);
}

std::vector<double> ReactionDiffusionBlock::restrictWallConductance(int axis) const {
    assert(axis >= 0 && axis < NDIM);
    return restrictFaceField(wallConductance[axis], axis);
}

void ReactionDiffusionBlock::setChildWallConductance(
    int axis, int octant, std::vector<double> const& values) {
    assert(axis >= 0 && axis < NDIM);
    setChildFaceField(wallConductance[axis], axis, octant, values);
}

}    // namespace octotiger::multigrid
