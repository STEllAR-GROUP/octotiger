// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "octotiger/fld_closure.hpp"
#include "octotiger/multigrid/reaction_diffusion.hpp"

namespace {

using octotiger::multigrid::Field;
using octotiger::multigrid::ReactionDiffusionBlock;
using position = std::array<double, 3>;

void require(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance, std::string const& message) {
    require(std::isfinite(actual) &&
            std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)),
        message + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}

template <class Function>
void rejects(Function function, std::string const& message) {
    bool rejected = false;
    try {
        function();
    } catch (std::exception const&) {
        rejected = true;
    }
    require(rejected, message);
}

void testOpacityAndClosure() {
    using octotiger::fld::absorption;
    using octotiger::fld::closure;
    near(absorption(4.0, 2.0, 3.0, 1.5, -2.0), 6.0, 2.0e-15,
        "opacity uses the stated density and temperature exponents without another rho");
    near(absorption(2.0, 0.25, 0.0, -2.0, -3.5), 0.0, 0.0,
        "zero absorption remains zero for a scattering-only material");
    rejects([&] { absorption(1.0, 0.0, 1.0, 0.0, -3.5); },
        "zero temperature cannot enter a singular power-law opacity");
    rejects(
        [&] { absorption(-1.0, 1.0, 1.0, 1.0, 0.0); }, "negative density cannot enter the opacity");
    rejects(
        [&] { absorption(1.0e200, 1.0, 1.0, 2.0, 0.0); }, "overflowing opacity must be reported");

    double const energy = 3.0;
    auto const isotropic = closure(energy, {{0.0, 0.0, 0.0}}, 2.0);
    near(isotropic.lambda, 1.0 / 3.0, 0.0, "zero-gradient diffusion limit");
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            near(isotropic.pressure[row][column], row == column ? 1.0 : 0.0, 2.0e-15,
                "isotropic pressure tensor");
        }
    }
    position const direction{{0.6, -0.8, 0.0}};
    for (double const r : {1.0e-100, 1.0e-3, 0.1, 1.0, 10.0, 1.0e3, 1.0e100}) {
        position gradient{};
        for (size_t axis = 0; axis < 3; ++axis) {
            gradient[axis] = r * energy * direction[axis];
        }
        auto const result = closure(energy, gradient, 1.0);
        require(result.lambda >= 0.0 && result.lambda <= 1.0 / 3.0 + 1.0e-15,
            "flux limiter leaves its physical interval");
        require(result.lambda * r <= 1.0 + 1.0e-15, "FLD diffusion exceeds the streaming flux cE");
        require(result.eddington >= 1.0 / 3.0 - 1.0e-15 && result.eddington <= 1.0 + 1.0e-15,
            "Eddington factor leaves its physical interval");
        double trace = 0.0;
        double normalPressure = 0.0;
        for (size_t row = 0; row < 3; ++row) {
            trace += result.pressure[row][row];
            for (size_t column = 0; column < 3; ++column) {
                near(result.pressure[row][column], result.pressure[column][row], 2.0e-15,
                    "radiation pressure is symmetric");
                normalPressure += direction[row] * result.pressure[row][column] * direction[column];
            }
        }
        near(trace, energy, 3.0e-15, "radiation pressure trace equals radiation energy");
        near(normalPressure, result.eddington * energy, 3.0e-15,
            "gradient-direction pressure has the Eddington eigenvalue");
    }
    auto const transparent = closure(energy, direction, 0.0);
    near(transparent.lambda, 0.0, 0.0, "zero extinction has a finite streaming limiter");
    near(transparent.eddington, 1.0, 0.0, "zero extinction approaches a directed beam");
    rejects([&] { closure(-1.0, direction, 1.0); }, "negative radiation energy is rejected");
    rejects([&] { closure(1.0, {{0.0, std::numeric_limits<double>::infinity(), 0.0}}, 1.0); },
        "nonfinite radiation gradient is rejected");
}

double quadratic(position const& point) {
    return 1.0 + point[0] * point[0] + 2.0 * point[1] * point[1] + 3.0 * point[2] * point[2];
}

void fillManufactured(ReactionDiffusionBlock& block) {
    for (int i = 0; i <= block.n + 1; ++i) {
        for (int j = 0; j <= block.n + 1; ++j) {
            for (int k = 0; k <= block.n + 1; ++k) {
                position const point{
                    {(i - 0.5) * block.dx, (j - 0.5) * block.dx, (k - 0.5) * block.dx}};
                block.solution(i, j, k) = quadratic(point);
                double const reaction = 0.5 + point[0] + 0.2 * point[1] + 0.3 * point[2];
                block.reaction(i, j, k) = reaction;
                block.rhs(i, j, k) = reaction * quadratic(point) -
                    (14.5 + 4.0 * point[0] + 4.0 * point[1] + 24.0 * point[2]);
                block.diffusion[0](i, j, k) = 1.0 + (i - 1.0) * block.dx;
                block.diffusion[1](i, j, k) = 2.0 + 0.5 * (j - 1.0) * block.dx;
                block.diffusion[2](i, j, k) = 0.75 + 2.0 * (k - 1.0) * block.dx;
            }
        }
    }
}

void testVariableCoefficientManufacturedOperator() {
    ReactionDiffusionBlock block(8, 0.125);
    fillManufactured(block);
    block.computeFluxes();
    block.updateFluxDefect();
    block.computeResidual();
    for (int i = 1; i <= block.n; ++i) {
        for (int j = 1; j <= block.n; ++j) {
            for (int k = 1; k <= block.n; ++k) {
                near(block.applyLocal(i, j, k), block.rhs(i, j, k), 3.0e-13,
                    "variable face diffusion reproduces the analytic quadratic operator");
                near(block.residual(i, j, k), 0.0, 2.0e-12,
                    "exact manufactured solution has zero physical residual");
                require(block.diagonal(i, j, k) > block.reaction(i, j, k),
                    "elliptic diffusion adds a positive diagonal");
            }
        }
    }
}

double jumpDiffusion(double x) {
    return x < 0.5 ? 1.0 : 100.0;
}

double jumpSolution(double x) {
    return x < 0.5 ? 1.0 + x : 1.5 + (x - 0.5) / 100.0;
}

double maximumResidual(ReactionDiffusionBlock& block) {
    block.computeFluxes();
    block.updateFluxDefect();
    block.computeResidual();
    double maximum = 0.0;
    for (int i = 1; i <= block.n; ++i) {
        for (int j = 1; j <= block.n; ++j) {
            for (int k = 1; k <= block.n; ++k) {
                maximum = std::max(maximum, std::abs(block.residual(i, j, k)));
            }
        }
    }
    return maximum;
}

void testCoefficientJumpSolve() {
    ReactionDiffusionBlock block(8, 0.125);
    for (int i = 0; i <= block.n + 1; ++i) {
        for (int j = 0; j <= block.n + 1; ++j) {
            for (int k = 0; k <= block.n + 1; ++k) {
                double const x = (i - 0.5) * block.dx;
                block.solution(i, j, k) = jumpSolution(x);
                block.reaction(i, j, k) = 2.0;
                block.rhs(i, j, k) = 2.0 * jumpSolution(x);
                double const left = jumpDiffusion(x - block.dx);
                double const right = jumpDiffusion(x);
                block.diffusion[0](i, j, k) = 2.0 * left * right / (left + right);
                block.diffusion[1](i, j, k) = right;
                block.diffusion[2](i, j, k) = right;
            }
        }
    }
    near(maximumResidual(block), 0.0, 2.0e-11,
        "harmonic interface coefficient preserves the exact constant flux");
    for (int i = 1; i <= block.n + 1; ++i) {
        near(block.flux[0](i, 3, 4), -1.0, 2.0e-13,
            "diffusion flux is continuous across a hundredfold coefficient jump");
    }
    for (int i = 1; i <= block.n; ++i) {
        for (int j = 1; j <= block.n; ++j) {
            for (int k = 1; k <= block.n; ++k) {
                block.solution(i, j, k) = 0.0;
            }
        }
    }
    int iterations = 0;
    double residual = maximumResidual(block);
    while (residual > 2.0e-9 && iterations < 4000) {
        block.relax();
        ++iterations;
        residual = maximumResidual(block);
    }
    require(residual <= 2.0e-9, "reaction-diffusion relaxation failed at a coefficient jump");
    double error = 0.0;
    for (int i = 1; i <= block.n; ++i) {
        for (int j = 1; j <= block.n; ++j) {
            for (int k = 1; k <= block.n; ++k) {
                error = std::max(
                    error, std::abs(block.solution(i, j, k) - jumpSolution((i - 0.5) * block.dx)));
            }
        }
    }
    require(error < 2.0e-10, "small jump-problem residual did not recover the exact solution");
    std::cout << "FLD_JUMP iterations=" << iterations << " residual=" << residual
              << " error=" << error << '\n';
}

void testClosedBoundaryConservation() {
    ReactionDiffusionBlock block(6, 1.0 / 6.0);
    for (int i = 0; i <= block.n + 1; ++i) {
        for (int j = 0; j <= block.n + 1; ++j) {
            for (int k = 0; k <= block.n + 1; ++k) {
                block.solution(i, j, k) = 1.0 + 0.1 * i * i + 0.2 * j + 0.03 * k * k;
                block.reaction(i, j, k) = 0.0;
                block.diffusion[0](i, j, k) = (i == 1 || i == block.n + 1) ? 0.0 : 1.7;
                block.diffusion[1](i, j, k) = (j == 1 || j == block.n + 1) ? 0.0 : 0.2;
                block.diffusion[2](i, j, k) = (k == 1 || k == block.n + 1) ? 0.0 : 3.1;
            }
        }
    }
    block.computeFluxes();
    double sum = 0.0;
    for (int i = 1; i <= block.n; ++i) {
        for (int j = 1; j <= block.n; ++j) {
            for (int k = 1; k <= block.n; ++k) {
                sum += block.fluxDivergence(i, j, k) * std::pow(block.dx, 3);
            }
        }
    }
    near(sum, 0.0, 2.0e-13, "closed-boundary diffusion conserves the volume integral");
}

void testCoarseFluxReconciliation() {
    ReactionDiffusionBlock coarse(4, 0.25);
    coarse.solution.fill(0.0);
    coarse.rhs.fill(0.0);
    coarse.reaction.fill(1.0);
    for (auto& diffusion : coarse.diffusion) {
        diffusion.fill(1.0);
    }
    coarse.computeFluxes();
    coarse.setRestrictedFlux(0, 5, 0, 0, std::vector<double>(4, 2.0));
    coarse.updateFluxDefect();
    near(coarse.fluxDivergence(4, 1, 1), 8.0, 0.0,
        "coarse residual uses the area-averaged fine flux");
    coarse.rhs(4, 1, 1) = 8.0;
    coarse.computeResidual();
    near(coarse.residual(4, 1, 1), 0.0, 0.0,
        "fine flux replacement is included once in the physical residual");
    near(coarse.fluxDefect(4, 1, 1), 8.0, 0.0,
        "relaxation defect has the correct coarse-fine flux sign");
}

void testOneCellWallLeakage() {
    ReactionDiffusionBlock block(1, 0.25);
    block.solution.fill(17.0);
    block.solution(1, 1, 1) = 0.0;
    block.reaction.fill(1.0);
    block.rhs.fill(2.0);
    double sumConductance = 0.0;
    for (int axis = 0; axis < NDIM; ++axis) {
        for (int side = 0; side < 2; ++side) {
            std::array<int, NDIM> face{{1, 1, 1}};
            face[axis] += side;
            double const conductance = 0.1 * (1 + 2 * axis + side);
            block.wallConductance[axis](face[0], face[1], face[2]) = conductance;
            sumConductance += conductance;
        }
    }
    double const expected = 2.0 / (1.0 + sumConductance / block.dx);
    near(block.diagonal(1, 1, 1), 1.0 + sumConductance / block.dx, 2.0e-15,
        "all six outgoing faces contribute to the one-cell diagonal");
    block.computeFluxes();
    block.updateFluxDefect();
    block.relax();
    near(block.solution(1, 1, 1), expected, 2.0e-15,
        "one-cell backward-Euler wall leakage has the analytic solution");
    near(maximumResidual(block), 0.0, 2.0e-15,
        "wall leakage appears once in the physical residual");
    double escaped = 0.0;
    for (int axis = 0; axis < NDIM; ++axis) {
        for (int side = 0; side < 2; ++side) {
            std::array<int, NDIM> face{{1, 1, 1}};
            face[axis] += side;
            double const outward = (side ? 1.0 : -1.0) *
                block.flux[axis](face[0], face[1], face[2]);
            double const conductance = 0.1 * (1 + 2 * axis + side);
            near(outward, conductance * expected, 2.0e-15,
                "wall flux has the correct outward orientation and ignores ghost values");
            escaped += outward * block.dx * block.dx;
        }
    }
    double const volume = block.dx * block.dx * block.dx;
    near(expected * volume + escaped, 2.0 * volume, 2.0e-15,
        "one-cell radiation plus escaped energy equals initial energy");
    near(2.0 - expected - block.fluxDivergence(1, 1, 1), 0.0, 2.0e-15,
        "pure transport wall loss leaves the conservative material increment zero");
    block.wallConductance[0](1, 1, 1) = -1.0;
    rejects([&] { block.computeFluxes(); }, "negative wall conductance is rejected");
}

void testWallConservationAndFas() {
    ReactionDiffusionBlock block(6, 1.0 / 6.0);
    for (int i = 0; i <= block.n + 1; ++i) {
        for (int j = 0; j <= block.n + 1; ++j) {
            for (int k = 0; k <= block.n + 1; ++k) {
                block.solution(i, j, k) = 1.0 + 0.1 * i * i + 0.2 * j + 0.03 * k * k;
                block.reaction(i, j, k) = 0.7;
                std::array<int, NDIM> const cell{{i, j, k}};
                for (int axis = 0; axis < NDIM; ++axis) {
                    bool const wall = cell[axis] == 1 || cell[axis] == block.n + 1;
                    block.diffusion[axis](i, j, k) = wall ? 0.0 : 0.2 * (axis + 1);
                    block.wallConductance[axis](i, j, k) = wall ?
                        0.03 * (axis + 1) + 0.001 * (i + j + k) : 0.0;
                }
            }
        }
    }
    block.computeFluxes();
    block.updateFluxDefect();
    double integratedDivergence = 0.0;
    double escaped = 0.0;
    Field restrictedResidual(block.n);
    for (int i = 1; i <= block.n; ++i) {
        for (int j = 1; j <= block.n; ++j) {
            for (int k = 1; k <= block.n; ++k) {
                integratedDivergence += block.fluxDivergence(i, j, k) * std::pow(block.dx, NDIM);
                near(block.applyLocal(i, j, k),
                    block.reaction(i, j, k) * block.solution(i, j, k) +
                        block.fluxDivergence(i, j, k), 2.0e-13,
                    "local operator and physical flux agree at outgoing faces");
                restrictedResidual(i, j, k) = 0.01 * (i - 2 * j + 3 * k);
            }
        }
    }
    for (int axis = 0; axis < NDIM; ++axis) {
        int const a = (axis + 1) % NDIM;
        int const b = (axis + 2) % NDIM;
        for (int side = 0; side < 2; ++side) {
            for (int j = 1; j <= block.n; ++j) {
                for (int k = 1; k <= block.n; ++k) {
                    std::array<int, NDIM> face{};
                    face[axis] = side ? block.n + 1 : 1;
                    face[a] = j;
                    face[b] = k;
                    auto cell = face;
                    cell[axis] = side ? block.n : 1;
                    escaped += block.wallConductance[axis](face[0], face[1], face[2]) *
                        block.solution(cell[0], cell[1], cell[2]) * block.dx * block.dx;
                }
            }
        }
    }
    near(integratedDivergence, escaped, 2.0e-13,
        "volume divergence telescopes to outgoing physical-wall energy");
    block.formCoarseRhs(restrictedResidual);
    block.computeResidual();
    for (int i = 1; i <= block.n; ++i) {
        for (int j = 1; j <= block.n; ++j) {
            for (int k = 1; k <= block.n; ++k) {
                near(block.residual(i, j, k), restrictedResidual(i, j, k), 2.0e-13,
                    "FAS preserves its restricted residual with outgoing walls");
            }
        }
    }
    block.wallConductance[0](2, 1, 1) = 1.0;
    rejects([&] { block.computeFluxes(); }, "an interior face cannot be an outgoing wall");
}

void testWallAreaRestriction() {
    ReactionDiffusionBlock coarse(4, 0.5);
    coarse.solution.fill(3.0);
    double fineEscaped = 0.0;
    for (int child = 0; child < NCHILD; ++child) {
        ReactionDiffusionBlock fine(4, 0.25);
        fine.solution.fill(3.0);
        for (int axis = 0; axis < NDIM; ++axis) {
            int const a = (axis + 1) % NDIM;
            int const b = (axis + 2) % NDIM;
            int const side = (child >> axis) & 1;
            for (int j = 1; j <= fine.n; ++j) {
                for (int k = 1; k <= fine.n; ++k) {
                    std::array<int, NDIM> face{};
                    face[axis] = side ? fine.n + 1 : 1;
                    face[a] = j;
                    face[b] = k;
                    int const globalJ = ((child >> a) & 1) * fine.n + j;
                    int const globalK = ((child >> b) & 1) * fine.n + k;
                    double const conductance = 0.1 * (axis + 1) + 0.5 * side +
                        0.02 * globalJ + 0.003 * globalK;
                    fine.wallConductance[axis](face[0], face[1], face[2]) = conductance;
                    fineEscaped += conductance * 3.0 * fine.dx * fine.dx;
                }
            }
            coarse.setChildWallConductance(axis, child, fine.restrictWallConductance(axis));
        }
    }
    coarse.computeFluxes();
    double coarseEscaped = 0.0;
    for (int i = 1; i <= coarse.n; ++i) {
        for (int j = 1; j <= coarse.n; ++j) {
            for (int k = 1; k <= coarse.n; ++k) {
                coarseEscaped += coarse.fluxDivergence(i, j, k) * std::pow(coarse.dx, NDIM);
            }
        }
    }
    near(coarse.wallConductance[0](1, 1, 1), 0.1 + 0.02 * 1.5 + 0.003 * 1.5,
        2.0e-15, "wall conductance restriction averages area without dx rescaling");
    near(coarseEscaped, fineEscaped, 3.0e-14,
        "coarse wall leakage equals the eight children's boundary-area sum");
}

}    // namespace

int main() {
    try {
        testOpacityAndClosure();
        testVariableCoefficientManufacturedOperator();
        testCoefficientJumpSolve();
        testClosedBoundaryConservation();
        testCoarseFluxReconciliation();
        testOneCellWallLeakage();
        testWallConservationAndFas();
        testWallAreaRestriction();
        std::cout << "FLD closure and reaction-diffusion tests passed\n";
    } catch (std::exception const& error) {
        std::cerr << "FLD numerical test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
