#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "octotiger/fld_boundary.hpp"
#include "octotiger/fld_grid.hpp"

namespace {

void require(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double value, double expected, double tolerance, std::string const& message) {
    require(std::isfinite(value) &&
            std::abs(value - expected) <= tolerance * std::max(1.0, std::abs(expected)),
        message + ": " + std::to_string(value) + " != " + std::to_string(expected));
}

size_t hydroIndex(int i, int j, int k) {
    return static_cast<size_t>(hindex(i + H_BW - 1, j + H_BW - 1, k + H_BW - 1));
}

double referenceTemperature(
    double rho, double thermal, double radiation, double heatCapacity,
    double dt, octotiger::fld::Parameters const& p) {
    long double const capacity = heatCapacity;
    long double const total = static_cast<long double>(thermal) + radiation;
    long double lower = 0.0L;
    long double upper = total / capacity;
    for (int iteration = 0; iteration < 160; ++iteration) {
        long double const t = (lower + upper) / 2.0L;
        long double const opacityTemperature = std::max(t, static_cast<long double>(p.temperatureFloor));
        long double const alpha = p.absorption *
            std::pow(static_cast<long double>(rho), p.densityExponent) *
            std::pow(opacityTemperature, p.temperatureExponent);
        long double const residual = capacity * t - thermal -
            dt * p.c * alpha * (total - capacity * t - p.ar * t * t * t * t);
        if (residual > 0.0L) {
            upper = t;
        } else {
            lower = t;
        }
    }
    return static_cast<double>((lower + upper) / 2.0L);
}

double refreshResidual(fld_grid& grid) {
    grid.refreshCellCoefficients();
    grid.refreshFaceCoefficients();
    // Each cell is isolated in this fixture; the actual thermal wrapper and
    // scalar Schur equation are retained, while all diffusion faces are closed.
    for (auto& coefficient : grid.linear.diffusion) {
        coefficient.fill(0.0);
    }
    grid.linear.solution.values = grid.energy.values;
    grid.linear.computeFluxes();
    grid.linear.updateFluxDefect();
    auto const residual = grid.computeNonlinearResidual();
    return std::max(std::sqrt((residual.radiationSquared + residual.materialSquared) /
                       residual.scaleSquared),
        residual.maximum);
}

void thermalCase(std::string const& name, octotiger::fld::Parameters const& p,
    double dt, bool variableCapacity = false) {
    auto energy = std::make_shared<std::vector<real>>(H_N3, 0.0);
    std::vector<real> rho(H_N3, 1.0);
    std::vector<real> thermal(H_N3, 1.0);
    std::vector<real> capacities(variableCapacity ? H_N3 : 0, 1.0);
    for (int i = 1; i <= INX; ++i) {
        for (int j = 1; j <= INX; ++j) {
            for (int k = 1; k <= INX; ++k) {
                size_t const h = hydroIndex(i, j, k);
                rho[h] = 0.8 + 0.1 * (i % 4);
                double const capacity = rho[h] *
                    (variableCapacity ? 0.3 + 0.4 * (i % 4) : p.cv);
                if (variableCapacity) {
                    capacities[h] = capacity;
                }
                thermal[h] = capacity * (0.6 + 0.2 * (j % 4));
                (*energy)[h] = 0.02 + 0.15 * (k % 4);
            }
        }
    }
    auto const original = *energy;
    auto const identity = energy.get();
    fld_grid grid(energy, 1.0 / INX);
    grid.initialize(dt, p, rho, thermal, capacities);
    double residual = refreshResidual(grid);
    int iteration = 0;
    for (; iteration < 200 && residual > 2.0e-12; ++iteration) {
        grid.prepareLinearization();
        for (int i = 1; i <= INX; ++i) {
            for (int j = 1; j <= INX; ++j) {
                for (int k = 1; k <= INX; ++k) {
                    require(grid.linear.reaction(i, j, k) >= 1.0,
                        "Schur thermal reaction lost its positive identity term");
                    grid.linear.solution(i, j, k) =
                        grid.linear.rhs(i, j, k) / grid.linear.reaction(i, j, k);
                }
            }
        }
        double damping = grid.prepareUpdate();
        bool accepted = false;
        for (int search = 0; search < 30; ++search) {
            grid.applyUpdate(damping);
            double const trial = refreshResidual(grid);
            if (trial < residual || trial < 2.0e-12) {
                residual = trial;
                accepted = true;
                break;
            }
            damping *= 0.5;
        }
        require(accepted, name + " nonlinear line search failed");
        require(*energy == original && grid.sharedEnergy.get() == identity,
            "an uncommitted nonlinear iteration changed the shared hydro state");
    }
    require(residual <= 2.0e-12, name + " failed to converge");
    auto const preview = grid.materialEnergyIncrement();
    require(*energy == original, "reading the proposed material increment committed radiation");
    auto const increment = grid.commit();
    require(increment == preview, "commit changed the proposed material increment");
    require(grid.sharedEnergy.get() == identity, "commit replaced the shared energy allocation");
    double error = 0.0;
    for (int i = 1; i <= INX; ++i) {
        for (int j = 1; j <= INX; ++j) {
            for (int k = 1; k <= INX; ++k) {
                size_t const h = hydroIndex(i, j, k);
                double const capacity = variableCapacity ? capacities[h] : rho[h] * p.cv;
                double const expected =
                    referenceTemperature(rho[h], thermal[h], original[h], capacity, dt, p);
                double const committedThermal = thermal[h] + increment[h];
                double const actual = committedThermal / capacity;
                error = std::max(error, std::abs(actual - expected));
                near(actual, expected, 2.0e-10, name + " backward-Euler temperature");
                near(actual, grid.temperature(i, j, k), 2.0e-10,
                    name + " conservative commit disagrees with solved gas temperature");
                near(committedThermal + (*energy)[h], thermal[h] + original[h], 5.0e-16,
                    name + " material/radiation energy conservation");
                require((*energy)[h] >= 0.0 && committedThermal > 0.0,
                    name + " committed a nonpositive state");
            }
        }
    }
    std::cout << "FLD_THERMAL case=" << name << " iterations=" << iteration
              << " residual=" << residual << " temperature_error=" << error << '\n';
}

void physicalBoundaryGradientCase() {
    auto energy = std::make_shared<std::vector<real>>(H_N3, 2.0);
    octotiger::fld::Parameters p;
    p.absorption = 0.0;
    p.scattering = 2.0;
    p.c = 3.0;
    constexpr double dx = 0.125;
    constexpr double dt = 0.4;
    std::array<double, NDIM> const gradient{{0.7, -0.4, 0.9}};
    fld_grid grid(energy, dx);
    grid.initialize(dt, p, std::vector<real>(H_N3, 1.0), std::vector<real>(H_N3, 1.0));
    std::array<bool, NFACE> const allPhysical{{true, true, true, true, true, true}};
    std::array<bool, NFACE> const onlyLeftPhysical{{true, false, false, false, false, false}};
    for (auto const& physical : {allPhysical, onlyLeftPhysical}) {
        // Match the application: physical ghosts repeat the nearest cell;
        // other ghosts carry the neighboring block's smooth affine solution.
        for (int i = 0; i <= INX + 1; ++i) {
            for (int j = 0; j <= INX + 1; ++j) {
                for (int k = 0; k <= INX + 1; ++k) {
                    std::array<int, NDIM> sample{{i, j, k}};
                    for (int d = 0; d < NDIM; ++d) {
                        if (sample[d] == 0 && physical[2 * d]) {
                            sample[d] = 1;
                        } else if (sample[d] == INX + 1 && physical[2 * d + 1]) {
                            sample[d] = INX;
                        }
                    }
                    double value = 2.0;
                    for (int d = 0; d < NDIM; ++d) {
                        value += gradient[d] * (sample[d] - 0.5) * dx;
                    }
                    grid.energy(i, j, k) = value;
                }
            }
        }
        grid.refreshCellCoefficients();
        grid.refreshFaceCoefficients();
        for (int axis = 0; axis < NDIM; ++axis) {
            for (int side = 0; side < 2; ++side) {
                if (!physical[2 * axis + side]) {
                    continue;
                }
                grid.refreshPhysicalBoundaryFaceCoefficients(axis, side, true, physical);
                int const a = (axis + 1) % NDIM;
                int const b = (axis + 2) % NDIM;
                for (int j = 1; j <= INX; ++j) {
                    for (int k = 1; k <= INX; ++k) {
                        std::array<int, NDIM> face{};
                        face[axis] = side ? INX + 1 : 1;
                        face[a] = j;
                        face[b] = k;
                        auto cell = face;
                        cell[axis] = side ? INX : 1;
                        double const expected = dt * octotiger::fld::vacuumConductance(
                            grid.energy(cell[0], cell[1], cell[2]), gradient,
                            p.scattering, p.c, 0.5 * dx);
                        near(grid.linear.wallConductance[axis](face[0], face[1], face[2]),
                            expected, 2.0e-15,
                            "physical faces, edges and corners retain the affine interior gradient");
                        require(grid.linear.diffusion[axis](face[0], face[1], face[2]) == 0.0,
                            "vacuum face must not retain a neighbor diffusion stencil");
                    }
                }
                grid.refreshPhysicalBoundaryFaceCoefficients(axis, side, false, physical);
                for (int j = 1; j <= INX; ++j) {
                    for (int k = 1; k <= INX; ++k) {
                        std::array<int, NDIM> face{};
                        face[axis] = side ? INX + 1 : 1;
                        face[a] = j;
                        face[b] = k;
                        require(grid.linear.wallConductance[axis](face[0], face[1], face[2]) == 0.0 &&
                                grid.linear.diffusion[axis](face[0], face[1], face[2]) == 0.0,
                            "reflecting faces clear both escape and neighbor diffusion");
                    }
                }
            }
        }
    }
}

void faceCoefficientCase() {
    for (bool const massScattering : std::array<bool, 2>{{false, true}}) {
        std::string const name = massScattering ? "mass scattering" : "density-dependent absorption";
        auto energy = std::make_shared<std::vector<real>>(H_N3, 2.0);
        std::vector<real> rho(H_N3, 1.0);
        std::vector<real> thermal(H_N3, 1.0);
        octotiger::fld::Parameters p;
        p.absorption = massScattering ? 0.0 : 1.0;
        p.scattering = massScattering ? 1.0 : 0.0;
        p.densityExponent = 1.0;
        p.c = 3.0;
        fld_grid grid(energy, 0.125);
        grid.initialize(0.4, p, rho, thermal);
        for (int i = 0; i <= INX + 1; ++i) {
            for (int j = 0; j <= INX + 1; ++j) {
                for (int k = 0; k <= INX + 1; ++k) {
                    grid.density(i, j, k) = i == 0 ? 8.0 : i < 4 ? 1.0 : 4.0;
                    grid.capacity(i, j, k) = grid.density(i, j, k) * p.cv;
                    grid.thermalStar(i, j, k) =
                        grid.density(i, j, k) * p.cv * grid.temperature(i, j, k);
                }
            }
        }
        grid.refreshCellCoefficients();
        for (int i : std::array<int, 3>{{0, 1, 4}}) {
            double const expected = i == 0 ? 8.0 : i == 1 ? 1.0 : 4.0;
            near(grid.extinction(i, 3, 3), expected, 2.0e-15,
                name + " extinction must scale with density");
            near(grid.opacity(i, 3, 3), massScattering ? 0.0 : expected, 2.0e-15,
                name + " thermal coupling must contain absorption only");
        }
        grid.refreshFaceCoefficients();
        near(grid.linear.diffusion[0](4, 3, 3), 0.16, 2.0e-15,
            name + " face coefficient integrates the two half-cell optical depths");
        grid.refreshBoundaryFaceCoefficients(0, 0, 2.0);
        near(grid.linear.diffusion[0](1, 3, 3), 0.8 / 17.0, 2.0e-15,
            name + " fine/coarse face uses path-weighted opacity and center distance");
        if (massScattering) {
            grid.prepareLinearization();
            for (int i = 1; i <= INX; ++i) {
                for (int j = 1; j <= INX; ++j) {
                    for (int k = 1; k <= INX; ++k) {
                        near(grid.linear.reaction(i, j, k), 1.0, 0.0,
                            "pure scattering must have no thermal reaction");
                        near(grid.linear.rhs(i, j, k), grid.energyStar(i, j, k), 0.0,
                            "pure scattering must have no thermal source");
                        // A radiation update can arise from diffusion, but
                        // elastic scattering must not heat the material.
                        grid.linear.solution(i, j, k) += 0.25;
                    }
                }
            }
            grid.prepareUpdate();
            for (int i = 1; i <= INX; ++i) {
                for (int j = 1; j <= INX; ++j) {
                    for (int k = 1; k <= INX; ++k) {
                        near(grid.proposedTemperature(i, j, k), grid.temperature(i, j, k), 0.0,
                            "pure scattering radiation update must not change material temperature");
                    }
                }
            }
        }
        grid.energy(0, 3, 3) = 0.01;
        grid.energy(1, 3, 3) = 5.0;
        grid.refreshCellCoefficients();
        grid.refreshBoundaryFaceCoefficients(0, 0, 2.0);
        grid.linear.solution.values = grid.energy.values;
        grid.linear.computeFluxes();
        double const faceEnergy = (0.01 + 2.0 * 5.0) / 3.0;
        require(std::abs(grid.linear.flux[0](1, 3, 3)) / grid.dt <= p.c * faceEnergy,
            name + " fine/coarse diffusion flux exceeds the face streaming bound");
    }
}

void independentMeansCase() {
    auto energy = std::make_shared<std::vector<real>>(H_N3, 2.0);
    std::vector<real> rho(H_N3, 1.0);
    std::vector<real> thermal(H_N3, 1.0);
    octotiger::fld::Parameters p;
    p.model = octotiger::fld::OpacityModel::independent;
    p.absorption = 7.0;
    p.rosseland = 2.0;
    p.c = 3.0;
    fld_grid grid(energy, 0.125);
    grid.initialize(0.4, p, rho, thermal);
    grid.refreshCellCoefficients();
    grid.refreshFaceCoefficients();
    near(grid.opacity(3, 3, 3), 7.0, 2e-15, "independent Planck absorption");
    near(grid.extinction(3, 3, 3), 2.0, 2e-15, "independent Rosseland extinction");
    near(grid.linear.diffusion[0](3, 3, 3), 0.2, 2.0e-15,
        "transport must use the Rosseland mean");
    grid.prepareLinearization();
    double const coupling = grid.dt * p.c * p.absorption;
    double const reaction = 1.0 + coupling / (1.0 + 4.0 * coupling);
    near(grid.linear.reaction(3, 3, 3), reaction, 2.0e-15,
        "material coupling must use the Planck mean");
    thermalCase("unequal_means", p, 0.4);
    thermalCase("varying_cell_capacity", p, 0.4, true);
    p.absorption = 0.0;
    thermalCase("independent_pure_transport", p, 0.4, true);
}

void coldPhysicalCase() {
    auto energy = std::make_shared<std::vector<real>>(H_N3, 0.0);
    std::vector<real> rho(H_N3, 1.0e-10);
    std::vector<real> thermal(H_N3, 0.0);
    std::vector<real> capacities(H_N3, 2.0);
    std::vector<real> ye(H_N3, 0.5);
    std::vector<real> z2(H_N3, 1.0);
    octotiger::fld::Parameters p;
    p.model = octotiger::fld::OpacityModel::free_free_thomson;
    p.temperatureFloor = 1.0e4;
    fld_grid grid(energy, 1.0);
    grid.initialize(1.0, p, rho, thermal, capacities, ye, z2);
    grid.refreshCellCoefficients();
    require(grid.regularizedCells == static_cast<size_t>(INX * INX * INX),
        "physical cold opacity must count interior cutoff cells only");
    require(grid.opacity(3, 3, 3) > 0.0 && grid.extinction(3, 3, 3) > 0.0,
        "physical cold opacity must be finite and positive");
    near(grid.temperature(3, 3, 3), 0.0, 0.0,
        "opacity cutoff must not replace actual material temperature");
    near(grid.capacity(3, 3, 3), 2.0, 0.0, "explicit heat capacity");
    near(grid.electronDensity(3, 3, 3), rho[0] * 0.5, 0.0, "explicit electron density moment");
    near(grid.ionChargeSquaredDensity(3, 3, 3), rho[0], 0.0, "explicit ionic charge density moment");
    grid.prepareLinearization();
    near(grid.linear.rhs(3, 3, 3), 0.0, 0.0,
        "opacity cutoff must not inject Planck emission into a cold cell");
    grid.linear.solution.fill(0.0);
    near(grid.prepareUpdate(), 1.0, 0.0, "cold equilibrium positivity damping");
    grid.applyUpdate(1.0);
    near(grid.temperature(3, 3, 3), 0.0, 0.0, "zero-temperature equilibrium update");
}

void stableColdUpdateCase() {
    auto energy = std::make_shared<std::vector<real>>(H_N3, 1.0e-20);
    std::vector<real> rho(H_N3, 1.0e-10);
    std::vector<real> thermal(H_N3, 1.0e-50);
    std::vector<real> capacities(H_N3, 3.895e-20);
    octotiger::fld::Parameters p;
    p.absorption = 1.0177e8;
    fld_grid grid(energy, 1.0 / INX);
    grid.initialize(1.0, p, rho, thermal, capacities);
    refreshResidual(grid);
    grid.prepareLinearization();
    grid.linear.solution.fill(1.0e-110);
    grid.energy(2, 2, 2) = 0.0;
    grid.linear.solution(2, 2, 2) = -1.0e-200;
    require(grid.prepareUpdate() == 1.0,
        "negative linear roundoff at zero radiation must not pin the global step");
    grid.applyUpdate(0.25);
    require(std::abs(grid.energy(1, 1, 1) / 0.75e-20 - 1.0) < 1.0e-15,
        "damped radiation proposal must interpolate the absolute states");
    grid.applyUpdate(1.0);
    require(grid.energy(1, 1, 1) == 1.0e-110,
        "full update must retain a radiation candidate below subtraction precision");
    require(grid.energy(2, 2, 2) == 0.0,
        "negative linear radiation roundoff must project to zero");
    require(grid.temperature(2, 2, 2) == thermal[0] / capacities[0],
        "projecting radiation must not create a thermal floor or heat");
    require((*energy)[hydroIndex(1, 1, 1)] == 1.0e-20,
        "feasible proposal must remain uncommitted");

    // Projection cannot make an inaccurate linear solution acceptable: the
    // caller's unchanged nonlinear residual must still reject lost energy.
    require(refreshResidual(grid) > 0.99,
        "projected update must still expose its physical energy residual");
}

void coldLinearResidualCase() {
    auto energy = std::make_shared<std::vector<real>>(H_N3, 0.0);
    std::vector<real> rho(H_N3, 1.0e-10);
    std::vector<real> thermal(H_N3, 1.0e-50);
    std::vector<real> capacities(H_N3, 3.895e-20);
    size_t const hot = hydroIndex(INX, INX, INX);
    thermal[hot] = 1.0;
    capacities[hot] = 1.0;
    octotiger::fld::Parameters p;
    p.absorption = 1.0177e8;
    fld_grid grid(energy, 1.0 / INX);
    grid.initialize(1.0, p, rho, thermal, capacities);
    refreshResidual(grid);
    grid.prepareLinearization();
    for (int i = 1; i <= INX; ++i) {
        for (int j = 1; j <= INX; ++j) {
            for (int k = 1; k <= INX; ++k) {
                grid.linear.solution(i, j, k) =
                    grid.linear.rhs(i, j, k) / grid.linear.reaction(i, j, k);
            }
        }
    }
    // An absolutely tiny interpolation error deposits far more than the
    // available energy in a cold stiff cell; the hot cell hides it in L2.
    grid.linear.solution(1, 1, 1) = 1.2365e-29;
    grid.linear.computeFluxes();
    grid.linear.updateFluxDefect();
    grid.linear.computeResidual();
    require(std::abs(grid.linear.residual(1, 1, 1)) < 2.0e-21,
        "cold interpolation contamination must be tiny in absolute units");
    require(grid.computeLinearResidualMaximum() > 0.99,
        "local linear criterion must detect cold-cell energy contamination");
    grid.linear.solution(1, 1, 1) =
        grid.linear.rhs(1, 1, 1) / grid.linear.reaction(1, 1, 1);
    grid.linear.computeFluxes();
    grid.linear.updateFluxDefect();
    grid.linear.computeResidual();
    require(grid.computeLinearResidualMaximum() < 1.0e-14,
        "exact isolated cold/hot linear solution must pass the local criterion");
}

void conservativeCompositionCase() {
    // Each fine octet contains equal volumes of rho=1 hydrogen and rho=3
    // carbon. Fractions must be mass-weighted, and T capacity-weighted.
    auto fineEnergy = std::make_shared<std::vector<real>>(H_N3, 1.0);
    std::vector<real> rho(H_N3, 1.0), thermal(H_N3, 6.0e4);
    std::vector<real> capacities(H_N3, 3.0), ye(H_N3, 1.0), z2(H_N3, 1.0);
    for (int i = 1; i <= INX; ++i) {
        for (int j = 1; j <= INX; ++j) {
            for (int k = 1; k <= INX; ++k) {
                size_t const h = hydroIndex(i, j, k);
                if (i % 2 == 0) {
                    rho[h] = 3.0;
                    capacities[h] = 21.0 / 8.0;
                    thermal[h] = capacities[h] * 4.0e4;
                    ye[h] = 0.5;
                    z2[h] = 3.0;
                }
            }
        }
    }
    octotiger::fld::Parameters p;
    p.model = octotiger::fld::OpacityModel::free_free_thomson;
    p.temperatureFloor = 1.0e3;
    p.densityUnit = 1.0e-10;
    fld_grid fine(fineEnergy, 0.5);
    fine.initialize(1.0, p, rho, thermal, capacities, ye, z2);

    double const expectedRho = 2.0;
    double const expectedCapacity = 45.0 / 16.0;
    double const expectedTemperature = 8.25e4 / expectedCapacity;
    double const expectedYe = 5.0 / 8.0;
    double const expectedZ2 = 5.0 / 2.0;
    auto coarseEnergy = std::make_shared<std::vector<real>>(H_N3, 1.0);
    fld_grid coarse(coarseEnergy, 1.0);
    coarse.initialize(1.0, p, std::vector<real>(H_N3, expectedRho),
        std::vector<real>(H_N3, 8.25e4), std::vector<real>(H_N3, expectedCapacity),
        std::vector<real>(H_N3, expectedYe), std::vector<real>(H_N3, expectedZ2));
    auto weightedTemperature = fine.temperature;
    for (size_t cell = 0; cell < weightedTemperature.values.size(); ++cell) {
        weightedTemperature.values[cell] *= fine.capacity.values[cell];
    }
    auto restrictedTemperature = fine.linear.restrictField(weightedTemperature);
    auto const restrictedCapacity = fine.linear.restrictField(fine.capacity);
    for (size_t cell = 0; cell < restrictedTemperature.size(); ++cell) {
        restrictedTemperature[cell] /= restrictedCapacity[cell];
    }
    for (int child = 0; child < 8; ++child) {
        coarse.linear.setChild(coarse.density, child, fine.linear.restrictField(fine.density));
        coarse.linear.setChild(coarse.capacity, child, restrictedCapacity);
        coarse.linear.setChild(coarse.electronDensity, child,
            fine.linear.restrictField(fine.electronDensity));
        coarse.linear.setChild(coarse.ionChargeSquaredDensity, child,
            fine.linear.restrictField(fine.ionChargeSquaredDensity));
        coarse.linear.setChild(coarse.temperature, child, restrictedTemperature);
    }
    near(coarse.capacity(3, 3, 3), expectedCapacity, 1.0e-15,
        "coarse heat capacity conserves the volume integral");
    near(coarse.temperature(3, 3, 3), expectedTemperature, 1.0e-15,
        "coarse temperature conserves thermal energy at variable capacity");
    near(coarse.electronDensity(3, 3, 3) / coarse.density(3, 3, 3), expectedYe,
        1.0e-15, "coarse electron fraction is density weighted");
    near(coarse.ionChargeSquaredDensity(3, 3, 3) / coarse.density(3, 3, 3), expectedZ2,
        1.0e-15, "coarse ionic charge moment is density weighted");
    coarse.refreshCellCoefficients();
    auto const expected = octotiger::fld::evaluate_opacity(
        p, expectedRho, expectedTemperature, expectedYe, expectedZ2);
    near(coarse.opacity(3, 3, 3), expected.planckAbsorption, 1.0e-15,
        "coarse Planck opacity uses conserved composition and thermal state");
    near(coarse.extinction(3, 3, 3), expected.rosselandExtinction, 1.0e-15,
        "coarse Rosseland opacity uses conserved composition and thermal state");
    auto const naive = octotiger::fld::evaluate_opacity(p, expectedRho, 3.0e4, 0.75, 2.0);
    require(std::abs(naive.planckAbsorption / expected.planckAbsorption - 1.0) > 1.0e-3,
        "mixed-cell fixture must distinguish volume-averaged fractions and temperature");
}

}    // namespace

int main() {
    try {
        physicalBoundaryGradientCase();
        faceCoefficientCase();
        independentMeansCase();
        coldPhysicalCase();
        stableColdUpdateCase();
        coldLinearResidualCase();
        conservativeCompositionCase();
        octotiger::fld::Parameters p;
        p.scattering = 1.0;
        thermalCase("weak", p, 1.0e-6);
        p.absorption = 1.0e12;
        thermalCase("stiff", p, 1.0);
        p.absorption = 1.0e5;
        p.densityExponent = 1.0;
        p.temperatureExponent = -3.0;
        p.cv = 0.7;
        p.ar = 1.3;
        thermalCase("inverse_temperature", p, 1.0);
        p.temperatureExponent = 2.0;
        thermalCase("positive_temperature", p, 1.0);
        p.absorption = 0.0;
        thermalCase("scattering_only", p, 1.0);
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
