// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "octotiger/fld_boundary.hpp"
#include "octotiger/fld_grid.hpp"

namespace {

using octotiger::multigrid::Field;

size_t hydroIndex(int i, int j, int k) {
    return static_cast<size_t>(hindex(i + H_BW - 1, j + H_BW - 1, k + H_BW - 1));
}

struct ThermalWeights
{
    double cOverH;
    double kOverH;
    double q;
    double g;
};

ThermalWeights thermalWeights(double capacity, double k, double planckDerivative) {
    ThermalWeights result{};
    if (k >= capacity) {
        double const ratio = capacity / k;
        double const denominator = ratio + planckDerivative;
        result.cOverH = ratio / denominator;
        result.kOverH = 1.0 / denominator;
        result.q = planckDerivative / denominator;
    } else {
        double const ratio = k / capacity;
        double const product = ratio * planckDerivative;
        double const denominator = 1.0 + product;
        result.cOverH = 1.0 / denominator;
        result.kOverH = ratio / denominator;
        result.q = product / denominator;
    }
    result.g = capacity * result.kOverH;
    if (!std::isfinite(result.g) || !std::isfinite(result.q) ||
        !std::isfinite(result.kOverH) || !std::isfinite(result.cOverH)) {
        throw std::runtime_error("Nonfinite FLD thermal linearization");
    }
    return result;
}

std::array<double, 2> planck(double temperature, double ar) {
    double const t2 = temperature * temperature;
    double const emission = ar * t2 * t2;
    double const derivative = 4.0 * ar * t2 * temperature;
    if (!std::isfinite(emission) || !std::isfinite(derivative)) {
        throw std::runtime_error("Nonfinite FLD Planck function");
    }
    return {{emission, derivative}};
}

void fillFaceCoefficients(fld_grid& grid, int axis, int firstFace, int lastFace,
    double neighborWidthRatio, int boundarySide) {
    int const n = grid.linear.n;
    double const dx = grid.linear.dx;
    int const a = (axis + 1) % NDIM;
    int const b = (axis + 2) % NDIM;
    // Normal distance is measured between centers. For ordinary faces both
    // widths are dx; at a fine/coarse face one width is neighborWidthRatio*dx.
    double const leftWidth = boundarySide == 0 ? neighborWidthRatio : 1.0;
    double const rightWidth = boundarySide == 1 ? neighborWidthRatio : 1.0;
    double const totalWidth = leftWidth + rightWidth;
    double const distance = 0.5 * dx * totalWidth;
    for (int face = firstFace; face <= lastFace; ++face) {
        for (int j = 1; j <= n; ++j) {
            for (int k = 1; k <= n; ++k) {
                std::array<int, NDIM> hi{};
                hi[axis] = face;
                hi[a] = j;
                hi[b] = k;
                auto lo = hi;
                --lo[axis];
                double const elo = grid.energy(lo[0], lo[1], lo[2]);
                double const ehi = grid.energy(hi[0], hi[1], hi[2]);
                double const energy = (rightWidth * elo + leftWidth * ehi) / totalWidth;
                // Integrate opacity over the two half-cell paths. In diffusion
                // this is the harmonic mean of the cell diffusion coefficients.
                double const chi =
                    (leftWidth * grid.extinction(lo[0], lo[1], lo[2]) +
                        rightWidth * grid.extinction(hi[0], hi[1], hi[2])) /
                    totalWidth;
                std::array<double, NDIM> gradient{};
                gradient[axis] = (ehi - elo) / distance;
                for (int tangent : {a, b}) {
                    auto ll = lo;
                    auto lh = lo;
                    auto hl = hi;
                    auto hh = hi;
                    --ll[tangent];
                    ++lh[tangent];
                    --hl[tangent];
                    ++hh[tangent];
                    gradient[tangent] =
                        (rightWidth * (grid.energy(lh[0], lh[1], lh[2]) -
                                          grid.energy(ll[0], ll[1], ll[2])) +
                            leftWidth * (grid.energy(hh[0], hh[1], hh[2]) -
                                            grid.energy(hl[0], hl[1], hl[2]))) /
                        (2.0 * dx * totalWidth);
                }
                auto const closure = octotiger::fld::closure(energy, gradient, chi);
                double const coefficient = grid.dt * grid.parameters.c * closure.lambda / chi *
                    (dx / distance);
                if (!(coefficient >= 0.0) || !std::isfinite(coefficient)) {
                    throw std::runtime_error("Nonfinite FLD face diffusivity");
                }
                grid.linear.diffusion[axis](hi[0], hi[1], hi[2]) = coefficient;
            }
        }
    }
}

}    // namespace

fld_grid::fld_grid(std::shared_ptr<std::vector<real>> radiationEnergy, double cellWidth)
  : sharedEnergy(std::move(radiationEnergy))
  , linear(INX, cellWidth)
  , energy(INX)
  , density(INX)
  , capacity(INX)
  , electronDensity(INX)
  , ionChargeSquaredDensity(INX)
  , temperature(INX)
  , energyStar(INX)
  , thermalStar(INX)
  , opacity(INX)
  , extinction(INX)
  , diffusionCell(INX)
  , savedEnergy(INX)
  , savedTemperature(INX)
  , proposedEnergy(INX)
  , proposedTemperature(INX) {
    if (!sharedEnergy || sharedEnergy->size() != H_N3) {
        throw std::invalid_argument("FLD requires a shared H_N3 radiation-energy vector");
    }
}

void fld_grid::initialize(double timestep, octotiger::fld::Parameters const& values,
    std::vector<real> const& rho, std::vector<real> const& internalEnergy,
    std::vector<real> const& cellCapacity, std::vector<real> const& cellElectronFraction,
    std::vector<real> const& cellIonChargeSquared) {
    if (rho.size() != H_N3 || internalEnergy.size() != H_N3 ||
        (!cellCapacity.empty() && cellCapacity.size() != H_N3) ||
        (!cellElectronFraction.empty() && cellElectronFraction.size() != H_N3) ||
        (!cellIonChargeSquared.empty() && cellIonChargeSquared.size() != H_N3) ||
        !(timestep > 0.0) || !std::isfinite(timestep) || !(values.cv > 0.0) ||
        !(values.c > 0.0) || !(values.ar > 0.0) ||
        !std::isfinite(values.cv) || !std::isfinite(values.c) || !std::isfinite(values.ar)) {
        throw std::invalid_argument("Invalid FLD implicit-stage parameters or layout");
    }
    dt = timestep;
    parameters = values;
    int const n = linear.n;
    for (int i = 0; i <= n + 1; ++i) {
        for (int j = 0; j <= n + 1; ++j) {
            for (int k = 0; k <= n + 1; ++k) {
                // Scratch ghosts start from the nearest interior state. The
                // distributed adapter fills their actual boundary values next.
                size_t const h = hydroIndex(std::clamp(i, 1, n), std::clamp(j, 1, n),
                    std::clamp(k, 1, n));
                double const r = rho[h];
                double const u = internalEnergy[h];
                double const e = (*sharedEnergy)[h];
                double const heatCapacity = cellCapacity.empty() ? r * parameters.cv : cellCapacity[h];
                double const ye = cellElectronFraction.empty() ? 0.5 : cellElectronFraction[h];
                double const z2 = cellIonChargeSquared.empty() ? 1.0 : cellIonChargeSquared[h];
                if (!(r > 0.0) || !(u >= 0.0) || !(e >= 0.0) ||
                    !std::isfinite(r) || !std::isfinite(u) || !std::isfinite(e)) {
                    throw std::runtime_error("FLD requires positive gas density and nonnegative thermal/radiation energy");
                }
                if (!(heatCapacity > 0.0) || !std::isfinite(heatCapacity) ||
                    !(ye >= 0.0) || !std::isfinite(ye) || !(z2 >= 0.0) || !std::isfinite(z2) ||
                    !std::isfinite(r * ye) || !std::isfinite(r * z2)) {
                    throw std::runtime_error("Invalid FLD heat capacity or composition");
                }
                double const t = u / heatCapacity;
                if (!(t >= 0.0) || !std::isfinite(t)) {
                    throw std::runtime_error("Invalid FLD material temperature");
                }
                density(i, j, k) = r;
                capacity(i, j, k) = heatCapacity;
                electronDensity(i, j, k) = r * ye;
                ionChargeSquaredDensity(i, j, k) = r * z2;
                temperature(i, j, k) = t;
                energy(i, j, k) = energyStar(i, j, k) = e;
                thermalStar(i, j, k) = u;
            }
        }
    }
    linear.solution.values = energy.values;
    linear.fluxDefect.fill(0.0);
}

void fld_grid::refreshCellCoefficients() {
    int const n = linear.n;
    regularizedCells = 0;
    for (int i = 0; i <= n + 1; ++i) {
        for (int j = 0; j <= n + 1; ++j) {
            for (int k = 0; k <= n + 1; ++k) {
                auto const coefficients = octotiger::fld::evaluate_opacity(parameters,
                    density(i, j, k), temperature(i, j, k),
                    electronDensity(i, j, k) / density(i, j, k),
                    ionChargeSquaredDensity(i, j, k) / density(i, j, k));
                double const chi = coefficients.rosselandExtinction;
                if (!(chi > 0.0) || !std::isfinite(chi)) {
                    throw std::runtime_error("FLD diffusion requires positive finite total extinction");
                }
                opacity(i, j, k) = coefficients.planckAbsorption;
                extinction(i, j, k) = chi;
                if (coefficients.regularized && i >= 1 && i <= n &&
                    j >= 1 && j <= n && k >= 1 && k <= n) {
                    ++regularizedCells;
                }
            }
        }
    }
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            for (int k = 1; k <= n; ++k) {
                double const scale = 0.5 / linear.dx;
                std::array<double, NDIM> const gradient{{
                    scale * (energy(i + 1, j, k) - energy(i - 1, j, k)),
                    scale * (energy(i, j + 1, k) - energy(i, j - 1, k)),
                    scale * (energy(i, j, k + 1) - energy(i, j, k - 1))}};
                diffusionCell(i, j, k) = parameters.c *
                    octotiger::fld::closure(energy(i, j, k), gradient, extinction(i, j, k)).lambda /
                    extinction(i, j, k);
            }
        }
    }
}

void fld_grid::refreshFaceCoefficients() {
    for (int axis = 0; axis < NDIM; ++axis) {
        linear.wallConductance[axis].fill(0.0);
        fillFaceCoefficients(*this, axis, 1, linear.n + 1, 1.0, -1);
    }
}

void fld_grid::refreshBoundaryFaceCoefficients(int axis, int side, double neighborWidthRatio) {
    if (axis < 0 || axis >= NDIM || side < 0 || side > 1 ||
        !(neighborWidthRatio > 0.0) || !std::isfinite(neighborWidthRatio)) {
        throw std::invalid_argument("Invalid FLD boundary face geometry");
    }
    int const face = side == 0 ? 1 : linear.n + 1;
    fillFaceCoefficients(*this, axis, face, face, neighborWidthRatio, side);
}

void fld_grid::refreshPhysicalBoundaryFaceCoefficients(int axis, int side, bool vacuum,
    std::array<bool, NFACE> const& physicalBoundaries) {
    if (axis < 0 || axis >= NDIM || side < 0 || side > 1) {
        throw std::invalid_argument("Invalid FLD physical boundary geometry");
    }
    int const a = (axis + 1) % NDIM;
    int const b = (axis + 2) % NDIM;
    for (int j = 1; j <= linear.n; ++j) {
        for (int k = 1; k <= linear.n; ++k) {
            std::array<int, NDIM> face{};
            face[axis] = side ? linear.n + 1 : 1;
            face[a] = j;
            face[b] = k;
            auto cell = face;
            cell[axis] = side ? linear.n : 1;
            double coefficient = 0.0;
            if (vacuum) {
                std::array<double, NDIM> gradient{};
                for (int d = 0; d < NDIM; ++d) {
                    auto lo = cell;
                    auto hi = cell;
                    if (cell[d] == 1 && physicalBoundaries[2 * d]) {
                        ++hi[d];
                    } else if (cell[d] == linear.n && physicalBoundaries[2 * d + 1]) {
                        --lo[d];
                    } else {
                        --lo[d];
                        ++hi[d];
                    }
                    // A mirrored physical ghost would halve a linear profile's
                    // derivative. Sample only interior cells in that direction.
                    gradient[d] = (energy(hi[0], hi[1], hi[2]) -
                        energy(lo[0], lo[1], lo[2])) / ((hi[d] - lo[d]) * linear.dx);
                }
                coefficient = dt * octotiger::fld::vacuumConductance(
                    energy(cell[0], cell[1], cell[2]), gradient,
                    extinction(cell[0], cell[1], cell[2]), parameters.c, 0.5 * linear.dx);
            }
            if (!(coefficient >= 0.0) || !std::isfinite(coefficient)) {
                throw std::runtime_error("Nonfinite FLD physical boundary conductance");
            }
            linear.diffusion[axis](face[0], face[1], face[2]) = 0.0;
            linear.wallConductance[axis](face[0], face[1], face[2]) = coefficient;
        }
    }
}

void fld_grid::prepareLinearization() {
    linear.solution.values = energy.values;
    linear.fluxDefect.fill(0.0);
    for (int i = 1; i <= linear.n; ++i) {
        for (int j = 1; j <= linear.n; ++j) {
            for (int k = 1; k <= linear.n; ++k) {
                double const heatCapacity = capacity(i, j, k);
                double const coupling = dt * parameters.c * opacity(i, j, k);
                if (!std::isfinite(coupling)) {
                    throw std::runtime_error("Nonfinite FLD coupling coefficient");
                }
                double const t = temperature(i, j, k);
                auto const b = planck(t, parameters.ar);
                auto const w = thermalWeights(heatCapacity, coupling, b[1]);
                double const tstar = thermalStar(i, j, k) / heatCapacity;
                linear.reaction(i, j, k) = 1.0 + w.g;
                linear.rhs(i, j, k) = energyStar(i, j, k) +
                    w.g * (b[0] + b[1] * (tstar - t));
                if (!std::isfinite(linear.rhs(i, j, k))) {
                    throw std::runtime_error("Nonfinite FLD linearized source");
                }
            }
        }
    }
}

double fld_grid::prepareUpdate() {
    savedEnergy.values = energy.values;
    savedTemperature.values = temperature.values;
    for (int i = 1; i <= linear.n; ++i) {
        for (int j = 1; j <= linear.n; ++j) {
            for (int k = 1; k <= linear.n; ++k) {
                double const t = temperature(i, j, k);
                double const heatCapacity = capacity(i, j, k);
                auto const b = planck(t, parameters.ar);
                auto const w = thermalWeights(heatCapacity,
                    dt * parameters.c * opacity(i, j, k), b[1]);
                double const linearEnergy = linear.solution(i, j, k);
                if (!std::isfinite(linearEnergy)) {
                    throw std::runtime_error("Nonfinite FLD nonlinear update");
                }
                // Tiny negative linear-solve errors must not pin the global
                // step at an empty radiation reservoir. This projection is
                // only a proposal; nonlinear convergence remains mandatory.
                double const nextEnergy = std::max(0.0, linearEnergy);
                // For B=a*T^4, kOverH*B=q*T/4. This nonnegative form avoids
                // cancellation in T+deltaT and in (Enew-E)+E when cold
                // radiation lies many orders below the preceding iterate.
                double const nextTemperature =
                    w.cOverH * (thermalStar(i, j, k) / heatCapacity) +
                    0.75 * w.q * t + w.kOverH * nextEnergy;
                if (!(nextTemperature >= 0.0) || !std::isfinite(nextTemperature)) {
                    throw std::runtime_error("Nonfinite FLD nonlinear update");
                }
                proposedEnergy(i, j, k) = nextEnergy;
                proposedTemperature(i, j, k) = nextTemperature;
            }
        }
    }
    return 1.0;
}

void fld_grid::applyUpdate(double damping) {
    if (!(damping > 0.0 && damping <= 1.0) || !std::isfinite(damping)) {
        throw std::invalid_argument("FLD update damping must be in (0,1]");
    }
    for (int i = 1; i <= linear.n; ++i) {
        for (int j = 1; j <= linear.n; ++j) {
            for (int k = 1; k <= linear.n; ++k) {
                double const e = (1.0 - damping) * savedEnergy(i, j, k) +
                    damping * proposedEnergy(i, j, k);
                double const t = (1.0 - damping) * savedTemperature(i, j, k) +
                    damping * proposedTemperature(i, j, k);
                if (!(e >= 0.0) || !(t >= 0.0) || !std::isfinite(e) || !std::isfinite(t)) {
                    throw std::runtime_error("FLD nonlinear update lost positivity");
                }
                energy(i, j, k) = e;
                temperature(i, j, k) = t;
            }
        }
    }
    linear.solution.values = energy.values;
}

octotiger::fld::NonlinearResidual fld_grid::computeNonlinearResidual() const {
    octotiger::fld::NonlinearResidual result;
    double const volume = linear.dx * linear.dx * linear.dx;
    for (int i = 1; i <= linear.n; ++i) {
        for (int j = 1; j <= linear.n; ++j) {
            for (int k = 1; k <= linear.n; ++k) {
                double const e = energy(i, j, k);
                double const t = temperature(i, j, k);
                double const heatCapacity = capacity(i, j, k);
                auto const b = planck(t, parameters.ar);
                auto const w = thermalWeights(heatCapacity,
                    dt * parameters.c * opacity(i, j, k), b[1]);
                double const thermalChange = heatCapacity * t - thermalStar(i, j, k);
                double const exchange = w.g * (e - b[0]);
                double const rr = e - energyStar(i, j, k) + linear.fluxDivergence(i, j, k) +
                    exchange + w.q * thermalChange;
                double const rt = w.cOverH * thermalChange - exchange;
                double const scale = std::max({energyStar(i, j, k), thermalStar(i, j, k),
                    e, heatCapacity * t, std::numeric_limits<double>::min()});
                if (!std::isfinite(rr) || !std::isfinite(rt) || !std::isfinite(scale)) {
                    throw std::runtime_error("Nonfinite FLD nonlinear residual");
                }
                result.radiationSquared += volume * rr * rr;
                result.materialSquared += volume * rt * rt;
                result.scaleSquared += volume * scale * scale;
                result.maximum = std::max(result.maximum, std::max(std::abs(rr), std::abs(rt)) / scale);
            }
        }
    }
    return result;
}

double fld_grid::computeLinearResidualMaximum() const {
    double result = 0.0;
    for (int i = 1; i <= linear.n; ++i) {
        for (int j = 1; j <= linear.n; ++j) {
            for (int k = 1; k <= linear.n; ++k) {
                double const t = temperature(i, j, k);
                double const heatCapacity = capacity(i, j, k);
                auto const b = planck(t, parameters.ar);
                auto const w = thermalWeights(heatCapacity,
                    dt * parameters.c * opacity(i, j, k), b[1]);
                double const e = linear.solution(i, j, k);
                // Back substitution of the frozen material row. Include its
                // new reservoir: real diffusion heating need not be bounded
                // by an initially vanishing material energy.
                double const thermal = w.q * heatCapacity * t +
                    w.cOverH * thermalStar(i, j, k) + w.g * (e - b[0]);
                double const scale = std::max({energyStar(i, j, k), thermalStar(i, j, k),
                    std::abs(e), std::abs(thermal), std::numeric_limits<double>::min()});
                double const residual = std::abs(linear.residual(i, j, k)) / scale;
                if (!std::isfinite(thermal) || !std::isfinite(residual)) {
                    throw std::runtime_error("Nonfinite locally scaled FLD linear residual");
                }
                result = std::max(result, residual);
            }
        }
    }
    return result;
}

std::vector<real> fld_grid::materialEnergyIncrement() const {
    std::vector<real> result(H_N3, 0.0);
    for (int i = 1; i <= linear.n; ++i) {
        for (int j = 1; j <= linear.n; ++j) {
            for (int k = 1; k <= linear.n; ++k) {
                result[hydroIndex(i, j, k)] = energyStar(i, j, k) - energy(i, j, k) -
                    linear.fluxDivergence(i, j, k);
            }
        }
    }
    return result;
}

std::vector<real> fld_grid::commit() {
    auto result = materialEnergyIncrement();
    for (int i = 1; i <= linear.n; ++i) {
        for (int j = 1; j <= linear.n; ++j) {
            for (int k = 1; k <= linear.n; ++k) {
                (*sharedEnergy)[hydroIndex(i, j, k)] = energy(i, j, k);
            }
        }
    }
    return result;
}
