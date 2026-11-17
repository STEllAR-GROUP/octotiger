// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "octotiger/multigrid/block.hpp"

namespace octotiger::multigrid {
namespace {

    size_t fieldSize(int n) {
        assert(n >= 2 && n % 2 == 0 && n <= std::numeric_limits<int>::max() - 2);
        auto const extent = static_cast<size_t>(n) + 2;
        assert(extent <= std::numeric_limits<size_t>::max() / extent);
        assert(extent * extent <= std::numeric_limits<size_t>::max() / extent);
        return extent * extent * extent;
    }

    std::array<int, NDIM> childOrigin(int n, int octant) {
        assert(octant >= 0 && octant < NCHILD);
        return {{1 + (octant & 1) * (n / 2), 1 + ((octant >> 1) & 1) * (n / 2),
            1 + ((octant >> 2) & 1) * (n / 2)}};
    }

    [[maybe_unused]] bool validPatch(int n, int axis, int face, int t0, int t1, size_t size) {
        return axis >= 0 && axis < NDIM && (face == 1 || face == n + 1) && t0 >= 0 && t1 >= 0 &&
            t0 <= n / 2 && t1 <= n / 2 && size == static_cast<size_t>(n / 2) * (n / 2);
    }

    std::vector<double> coarseFaceDifferences(Field const& field, int axis, int side) {
        assert(axis >= 0 && axis < NDIM);
        assert(side == 0 || side == 1);
        int const n = field.n;
        int const face = side == 0 ? 1 : n + 1;
        int const a = (axis + 1) % NDIM;
        int const b = (axis + 2) % NDIM;
        std::vector<double> result;
        result.reserve(static_cast<size_t>(n / 2) * (n / 2));
        for (int j = 1; j <= n; j += 2) {
            for (int k = 1; k <= n; k += 2) {
                double value = 0.0;
                for (int dj = 0; dj < 2; ++dj) {
                    for (int dk = 0; dk < 2; ++dk) {
                        std::array<int, NDIM> hi{};
                        hi[axis] = face;
                        hi[a] = j + dj;
                        hi[b] = k + dk;
                        auto lo = hi;
                        --lo[axis];
                        value += field(hi[0], hi[1], hi[2]) - field(lo[0], lo[1], lo[2]);
                    }
                }
                result.push_back(0.5 * value);
            }
        }
        return result;
    }

}    // namespace

Field::Field(int interiorSize)
  : n(interiorSize)
  , values(fieldSize(n), 0.0) {}

double& Field::operator()(int i, int j, int k) {
    assert(i >= 0 && i < n + 2 && j >= 0 && j < n + 2 && k >= 0 && k < n + 2);
    size_t const extent = static_cast<size_t>(n) + 2;
    return values[(static_cast<size_t>(i) * extent + j) * extent + k];
}

double Field::operator()(int i, int j, int k) const {
    assert(i >= 0 && i < n + 2 && j >= 0 && j < n + 2 && k >= 0 && k < n + 2);
    size_t const extent = static_cast<size_t>(n) + 2;
    return values[(static_cast<size_t>(i) * extent + j) * extent + k];
}

void Field::fill(double value) {
    std::fill(values.begin(), values.end(), value);
}

double Field::laplacianNumerator(int i, int j, int k) const {
    double const center = (*this)(i, j, k);
    // Differences make an exactly constant field annihilate without cancellation.
    return ((*this)(i - 1, j, k) - center) + ((*this)(i + 1, j, k) - center) +
        ((*this)(i, j - 1, k) - center) + ((*this)(i, j + 1, k) - center) +
        ((*this)(i, j, k - 1) - center) + ((*this)(i, j, k + 1) - center);
}

double Field::octetAverage(int i, int j, int k) const {
    double value = 0.0;
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                value += (*this)(i + di, j + dj, k + dk);
            }
        }
    }
    return value / NCHILD;
}

Block::Block(int interiorSize, double cellWidth)
  : n(interiorSize)
  , dx(cellWidth)
  , phi(n)
  , source(n)
  , correction(n)
  , correctionRhs(n) {
    for (auto& component : force) {
        component = Field(n);
    }
    assert(dx > 0.0 && std::isfinite(dx));
}

void Block::initializeCorrection() {
    correction.fill(0.0);
    correctionRhs.fill(0.0);
    if (!refined) {
        double const scale = dx * dx / NFACE;
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                for (int k = 1; k <= n; ++k) {
                    correctionRhs(i, j, k) = scale * physicalResidual(i, j, k);
                }
            }
        }
    }
}

double Block::correctionResidual(int i, int j, int k) const {
    return correctionRhs(i, j, k) + correction.laplacianNumerator(i, j, k) / NFACE;
}

std::vector<double> Block::restrictCorrectionResidual() const {
    constexpr int refinementRatio = 2;
    constexpr double scale = double(refinementRatio * refinementRatio) / NCHILD;
    std::vector<double> result;
    result.reserve(static_cast<size_t>(n / 2) * (n / 2) * (n / 2));
    for (int i = 1; i <= n; i += refinementRatio) {
        for (int j = 1; j <= n; j += refinementRatio) {
            for (int k = 1; k <= n; k += refinementRatio) {
                double value = 0.0;
                for (int di = 0; di < refinementRatio; ++di) {
                    for (int dj = 0; dj < refinementRatio; ++dj) {
                        for (int dk = 0; dk < refinementRatio; ++dk) {
                            value += correctionResidual(i + di, j + dj, k + dk);
                        }
                    }
                }
                result.push_back(scale * value);
            }
        }
    }
    return result;
}

void Block::addCorrection(std::vector<double> const& values) {
    assert(values.size() == static_cast<size_t>(n) * n * n);
    size_t index = 0;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            for (int k = 1; k <= n; ++k) {
                correction(i, j, k) += values[index++];
            }
        }
    }
}

std::vector<double> Block::prolongatedCorrection(int octant) const {
    assert(octant >= 0 && octant < NCHILD);
    std::array<double, NDIM> offset{};
    for (int d = 0; d < NDIM; ++d) {
        offset[d] = ((octant >> d) & 1) * (n / 2) + 0.25;
    }
    std::vector<double> result;
    result.reserve(static_cast<size_t>(n) * n * n);
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            for (int k = 1; k <= n; ++k) {
                std::array<int, NDIM> const index{{i, j, k}};
                auto logical = offset;
                for (int d = 0; d < NDIM; ++d) {
                    logical[d] += 0.5 * index[d];
                }
                result.push_back(linearInterpolate(correction, logical));
            }
        }
    }
    return result;
}

void Block::beginDown() {
    for (int i = 0; i <= n + 1; ++i) {
        for (int j = 0; j <= n + 1; ++j) {
            correction(0, i, j) = correction(n + 1, i, j) = 0.0;
            correction(i, 0, j) = correction(i, n + 1, j) = 0.0;
            correction(i, j, 0) = correction(i, j, n + 1) = 0.0;
        }
    }
    if (refined) {
        return;
    }
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            for (int k = 1; k <= n; ++k) {
                correction(i, j, k) =
                    (phi.laplacianNumerator(i, j, k) - dx * dx * source(i, j, k)) / NFACE;
            }
        }
    }
}

void Block::saveCorrectionRhs() {
    correctionRhs.values = correction.values;
}

void Block::relax() {
    for (int color = 0; color < 2; ++color) {
        // Same checkerboard as Scorpio, traversed in this port's contiguous z
        // direction. Seven-point neighbors have the opposite color.
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                for (int k = (i + j + color) % 2 + 1; k <= n; k += 2) {
                    correction(i, j, k) +=
                        correctionRhs(i, j, k) + correction.laplacianNumerator(i, j, k) / NFACE;
                }
            }
        }
    }
}

void Block::retireCorrection() {
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            for (int k = 1; k <= n; ++k) {
                phi(i, j, k) += correction(i, j, k);
            }
        }
    }
}

void Block::computeForces() {
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
                    force[axis](hi[0], hi[1], hi[2]) =
                        -(phi(hi[0], hi[1], hi[2]) - phi(lo[0], lo[1], lo[2])) / dx;
                }
            }
        }
    }
}

double Block::physicalResidual(int i, int j, int k) const {
    return -(force[0](i + 1, j, k) - force[0](i, j, k) + force[1](i, j + 1, k) - force[1](i, j, k) +
               force[2](i, j, k + 1) - force[2](i, j, k)) /
        dx -
        source(i, j, k);
}

double Block::residualNorm() const {
    if (refined) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            for (int k = 1; k <= n; ++k) {
                double const value = physicalResidual(i, j, k);
                if (!std::isfinite(value)) {
                    return std::numeric_limits<double>::infinity();
                }
                sum += std::abs(value);
            }
        }
    }
    return sum * dx * dx * dx;
}

double Block::residualMaximum() const {
    if (refined) {
        return 0.0;
    }
    double result = 0.0;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            for (int k = 1; k <= n; ++k) {
                double const value = physicalResidual(i, j, k);
                if (!std::isfinite(value)) {
                    return std::numeric_limits<double>::infinity();
                }
                result = std::max(result, std::abs(value));
            }
        }
    }
    return result;
}

std::vector<double> Block::restrictField(Field const& field) const {
    std::vector<double> result;
    result.reserve(static_cast<size_t>(n / 2) * (n / 2) * (n / 2));
    for (int i = 1; i <= n; i += 2) {
        for (int j = 1; j <= n; j += 2) {
            for (int k = 1; k <= n; k += 2) {
                result.push_back(field.octetAverage(i, j, k));
            }
        }
    }
    return result;
}

std::vector<double> Block::childValues(Field const& field, int octant) const {
    auto const origin = childOrigin(n, octant);
    std::vector<double> result;
    result.reserve(static_cast<size_t>(n / 2) * (n / 2) * (n / 2));
    for (int i = 0; i < n / 2; ++i) {
        for (int j = 0; j < n / 2; ++j) {
            for (int k = 0; k < n / 2; ++k) {
                result.push_back(field(origin[0] + i, origin[1] + j, origin[2] + k));
            }
        }
    }
    return result;
}

void Block::setChild(Field& field, int octant, std::vector<double> const& values) const {
    auto const origin = childOrigin(n, octant);
    assert(values.size() == static_cast<size_t>(n / 2) * (n / 2) * (n / 2));
    size_t index = 0;
    for (int i = 0; i < n / 2; ++i) {
        for (int j = 0; j < n / 2; ++j) {
            for (int k = 0; k < n / 2; ++k) {
                field(origin[0] + i, origin[1] + j, origin[2] + k) = values[index++];
            }
        }
    }
}

void Block::replaceCorrectionMean(std::vector<double> const& parentValues) {
    assert(parentValues.size() == static_cast<size_t>(n / 2) * (n / 2) * (n / 2));
    size_t index = 0;
    for (int i = 1; i <= n; i += 2) {
        for (int j = 1; j <= n; j += 2) {
            for (int k = 1; k <= n; k += 2) {
                double const change = parentValues[index++] - correction.octetAverage(i, j, k);
                for (int di = 0; di < 2; ++di) {
                    for (int dj = 0; dj < 2; ++dj) {
                        for (int dk = 0; dk < 2; ++dk) {
                            correction(i + di, j + dj, k + dk) += change;
                        }
                    }
                }
            }
        }
    }
}

double Block::quadraticInterpolate(
    Field const& field, std::array<double, NDIM> const& logicalPosition) {
    std::array<int, NDIM> center{};
    std::array<std::array<double, 3>, NDIM> weights{};
    for (int d = 0; d < NDIM; ++d) {
        center[d] = static_cast<int>(std::floor(logicalPosition[d] + 0.5));
        assert(center[d] >= 1 && center[d] <= field.n);
        double const x = logicalPosition[d] - center[d];
        weights[d] = {{0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)}};
    }
    double result = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                result += weights[0][i] * weights[1][j] * weights[2][k] *
                    field(center[0] + i - 1, center[1] + j - 1, center[2] + k - 1);
            }
        }
    }
    return result;
}

double Block::sampleNearestCell(Field const& field, std::array<double, NDIM> const& logicalPosition) {
    std::array<int, NDIM> index{};
    for (int d = 0; d < NDIM; ++d) {
        index[d] = static_cast<int>(std::floor(logicalPosition[d] + 0.5));
        assert(index[d] >= 0 && index[d] <= field.n + 1);
    }
    return field(index[0], index[1], index[2]);
}

double Block::linearInterpolate(
    Field const& field, std::array<double, NDIM> const& logicalPosition) {
    std::array<int, NDIM> lower{};
    std::array<double, NDIM> fraction{};
    for (int d = 0; d < NDIM; ++d) {
        double const coordinate = logicalPosition[d];
        assert(std::isfinite(coordinate) && coordinate >= 0.0 && coordinate <= field.n + 1.0);
        lower[d] = std::min(static_cast<int>(std::floor(coordinate)), field.n);
        fraction[d] = coordinate - lower[d];
    }
    double result = 0.0;
    for (int octant = 0; octant < NCHILD; ++octant) {
        auto index = lower;
        double weight = 1.0;
        for (int d = 0; d < NDIM; ++d) {
            int const upper = (octant >> d) & 1;
            index[d] += upper;
            weight *= upper != 0 ? fraction[d] : 1.0 - fraction[d];
        }
        result += weight * field(index[0], index[1], index[2]);
    }
    return result;
}

std::vector<double> Block::coarseFaceJumps(int axis, int side) const {
    return coarseFaceDifferences(phi, axis, side);
}

std::vector<double> Block::coarseCorrectionFaceJumps(int axis, int side) const {
    return coarseFaceDifferences(correction, axis, side);
}

std::vector<double> Block::coarseFaceForces(int axis, int side) const {
    assert(axis >= 0 && axis < NDIM);
    assert(side == 0 || side == 1);
    int const face = side == 0 ? 1 : n + 1;
    int const a = (axis + 1) % NDIM;
    int const b = (axis + 2) % NDIM;
    std::vector<double> result;
    for (int j = 1; j <= n; j += 2) {
        for (int k = 1; k <= n; k += 2) {
            double value = 0.0;
            for (int dj = 0; dj < 2; ++dj) {
                for (int dk = 0; dk < 2; ++dk) {
                    std::array<int, NDIM> cell{};
                    cell[axis] = face;
                    cell[a] = j + dj;
                    cell[b] = k + dk;
                    value += force[axis](cell[0], cell[1], cell[2]);
                }
            }
            result.push_back(value / (NCHILD / 2));
        }
    }
    return result;
}

void Block::adjustFluxDefect(
    int axis, int faceIndex, int t0, int t1, std::vector<double> const& jumps) {
    assert(!refined);
    assert(validPatch(n, axis, faceIndex, t0, t1, jumps.size()));
    int const a = (axis + 1) % NDIM;
    int const b = (axis + 2) % NDIM;
    int const interior = faceIndex == 1 ? 1 : n;
    double const sign = faceIndex == 1 ? 1.0 : -1.0;
    size_t index = 0;
    for (int j = 1; j <= n / 2; ++j) {
        for (int k = 1; k <= n / 2; ++k) {
            std::array<int, NDIM> hi{};
            hi[axis] = faceIndex;
            hi[a] = j + t0;
            hi[b] = k + t1;
            auto lo = hi;
            --lo[axis];
            double const delta =
                (phi(hi[0], hi[1], hi[2]) - phi(lo[0], lo[1], lo[2]) - jumps[index++]) / NFACE;
            auto cell = hi;
            cell[axis] = interior;
            correction(cell[0], cell[1], cell[2]) += sign * delta;
        }
    }
}

void Block::addCorrectionFluxRhs(
    int axis, int faceIndex, int t0, int t1, std::vector<double> const& jumps) {
    assert(!refined);
    assert(validPatch(n, axis, faceIndex, t0, t1, jumps.size()));
    int const a = (axis + 1) % NDIM;
    int const b = (axis + 2) % NDIM;
    int const interior = faceIndex == 1 ? 1 : n;
    double const scale = (faceIndex == 1 ? -1.0 : 1.0) / NFACE;
    size_t index = 0;
    for (int j = 1; j <= n / 2; ++j) {
        for (int k = 1; k <= n / 2; ++k) {
            std::array<int, NDIM> cell{};
            cell[axis] = interior;
            cell[a] = j + t0;
            cell[b] = k + t1;
            correctionRhs(cell[0], cell[1], cell[2]) += scale * jumps[index++];
        }
    }
}

void Block::setRestrictedForce(
    int axis, int faceIndex, int t0, int t1, std::vector<double> const& forces) {
    assert(!refined);
    assert(validPatch(n, axis, faceIndex, t0, t1, forces.size()));
    int const a = (axis + 1) % NDIM;
    int const b = (axis + 2) % NDIM;
    size_t index = 0;
    for (int j = 1; j <= n / 2; ++j) {
        for (int k = 1; k <= n / 2; ++k) {
            std::array<int, NDIM> cell{};
            cell[axis] = faceIndex;
            cell[a] = j + t0;
            cell[b] = k + t1;
            force[axis](cell[0], cell[1], cell[2]) = forces[index++];
        }
    }
}

}    // namespace octotiger::multigrid
