// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "octotiger/multigrid/block.hpp"
#include "octotiger/multigrid/poisson_cycle.hpp"

using octotiger::multigrid::Block;
using octotiger::multigrid::Field;
using octotiger::multigrid::PoissonStage;
using Position = std::array<double, NDIM>;

namespace {

void require(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double polynomial(Position const& x) {
    return 0.5 + x[0] + 2.0 * x[1] - 3.0 * x[2] + x[0] * x[0] + 2.0 * x[1] * x[1] +
        3.0 * x[2] * x[2] + x[0] * x[1];
}

double smoothSolution(Position const& x) {
    double const pi = std::acos(-1.0);
    return std::sin(pi * x[0]) * std::sin(pi * x[1]) * std::sin(pi * x[2]) +
        0.1 * (x[0] + 2.0 * x[1] - x[2]);
}

void testTransfersAndStencil() {
    Block block(8, 0.125);
    for (int i = 0; i <= block.n + 1; ++i) {
        for (int j = 0; j <= block.n + 1; ++j) {
            for (int k = 0; k <= block.n + 1; ++k) {
                block.phi(i, j, k) = polynomial({{double(i), double(j), double(k)}});
                block.correction(i, j, k) = 0.2 * i - 0.7 * j + 1.1 * k;
            }
        }
    }
    require(std::abs(block.phi.laplacianNumerator(4, 4, 4) - 12.0) < 1.e-13,
        "Seven-point stencil does not reproduce a quadratic Laplacian");
    Position const point{{3.75, 4.25, 2.75}};
    require(std::abs(Block::quadraticInterpolate(block.phi, point) - polynomial(point)) < 1.e-12,
        "Scorpio coarse physical interpolation does not reproduce a quadratic");
    for (int octant = 0; octant < NCHILD; ++octant) {
        auto const interpolated = block.prolongatedCorrection(octant);
        Block child(block.n, block.dx / 2.0);
        child.correction.fill(2.0);
        child.addCorrection(interpolated);
        for (int i = 1; i <= block.n; ++i) {
            for (int j = 1; j <= block.n; ++j) {
                for (int k = 1; k <= block.n; ++k) {
                    double const x = 0.5 * i + 0.25 + (octant & 1) * (block.n / 2);
                    double const y = 0.5 * j + 0.25 + ((octant >> 1) & 1) * (block.n / 2);
                    double const z = 0.5 * k + 0.25 + ((octant >> 2) & 1) * (block.n / 2);
                    require(std::abs(child.correction(i, j, k) -
                                (2.0 + 0.2 * x - 0.7 * y + 1.1 * z)) < 1.e-13,
                        "Additive trilinear prolongation does not reproduce a linear function");
                }
            }
        }
    }
    block.computeForces();
    block.initializeCorrection();
    for (int i = 1; i <= block.n; ++i) {
        for (int j = 1; j <= block.n; ++j) {
            for (int k = 1; k <= block.n; ++k) {
                require(block.correction(i, j, k) == 0.0 &&
                        std::abs(block.correctionRhs(i, j, k) - 2.0) < 1.e-13,
                    "Correction initialization has the wrong Poisson residual scaling");
                block.correction(i, j, k) = 0.2 * i - 0.7 * j + 1.1 * k;
            }
        }
    }
    auto const before = block.correction;
    std::vector<double> const target(64, 3.0);
    block.replaceCorrectionMean(target);
    auto const restricted = block.restrictField(block.correction);
    for (double value : restricted) {
        require(std::abs(value - 3.0) < 2.e-14, "Upward transfer did not replace octet means");
    }
    require(std::abs((block.correction(1, 1, 1) - block.correction(2, 2, 2)) -
                (before(1, 1, 1) - before(2, 2, 2))) < 2.e-14,
        "Upward transfer discarded fine correction variation");
    block.setChild(block.source, 5, restricted);
    require(block.childValues(block.source, 5) == restricted, "Child payload ordering differs");
    block.phi.fill(7.0);
    require(block.phi.laplacianNumerator(3, 3, 3) == 0.0, "Constant null mode not annihilated");
}

// Serial validation fixture only. Production ownership and communication live
// in the node-server driver. Both use the shared Poisson cycle schedule;
// this fixture supplies serial data movement with the same stage barriers.
struct Node
{
    Block block;
    int level;
    std::array<int, NDIM> location;
    int parent = -1;
    int octant = 0;
    Node(int n, int lev, std::array<int, NDIM> loc, int par, int oct)
      : block(n, std::ldexp(1.0 / n, -lev))
      , level(lev)
      , location(loc)
      , parent(par)
      , octant(oct) {}
};

class Fixture
{
public:
    int n;
    int maximumLevel = 0;
    std::vector<std::unique_ptr<Node>> nodes;
    std::vector<std::vector<int>> spatial{{0}};
    std::function<double(Position const&)> boundary = smoothSolution;

    explicit Fixture(int interior)
      : n(interior) {
        nodes.push_back(std::make_unique<Node>(n, 0, std::array<int, NDIM>{{0, 0, 0}}, -1, 0));
    }

    void refine(int index) {
        Node& parent = *nodes.at(index);
        require(!parent.block.refined, "Fixture node was already refined");
        parent.block.refined = true;
        int const nextLevel = parent.level + 1;
        maximumLevel = std::max(maximumLevel, nextLevel);
        if (spatial.size() <= static_cast<size_t>(nextLevel)) {
            size_t const width = size_t(1) << nextLevel;
            spatial.emplace_back(width * width * width, -1);
        }
        auto const location = parent.location;
        for (int octant = 0; octant < NCHILD; ++octant) {
            std::array<int, NDIM> child{};
            for (int d = 0; d < NDIM; ++d) {
                child[d] = 2 * location[d] + ((octant >> d) & 1);
            }
            size_t const width = size_t(1) << nextLevel;
            spatial[nextLevel][(child[0] * width + child[1]) * width + child[2]] =
                static_cast<int>(nodes.size());
            nodes.push_back(std::make_unique<Node>(n, nextLevel, child, index, octant));
        }
    }

    Position position(Node const& node, int i, int j, int k) const {
        std::array<int, NDIM> const index{{i, j, k}};
        Position result{};
        for (int d = 0; d < NDIM; ++d) {
            result[d] = node.block.dx * (n * node.location[d] + index[d] - 0.5);
        }
        return result;
    }

    Position logical(Node const& node, Position const& p) const {
        Position result{};
        for (int d = 0; d < NDIM; ++d) {
            result[d] = p[d] / node.block.dx - n * node.location[d] + 0.5;
        }
        return result;
    }

    void refineUniformly(int targetLevel) {
        for (int level = 0; level < targetLevel; ++level) {
            size_t const count = nodes.size();
            for (size_t index = 0; index < count; ++index) {
                if (nodes[index]->level == level) {
                    refine(static_cast<int>(index));
                }
            }
        }
    }

    int find(int level, Position const& p) const {
        if (level < 0 || static_cast<size_t>(level) >= spatial.size() || outside(p)) {
            return -1;
        }
        size_t const width = size_t(1) << level;
        auto const i = static_cast<size_t>(p[0] * width);
        auto const j = static_cast<size_t>(p[1] * width);
        auto const k = static_cast<size_t>(p[2] * width);
        return spatial[level][(i * width + j) * width + k];
    }

    static bool outside(Position const& p) {
        return p[0] < 0.0 || p[0] >= 1.0 || p[1] < 0.0 || p[1] >= 1.0 || p[2] < 0.0 || p[2] >= 1.0;
    }

    void syncPhi() {
        for (int level = maximumLevel; level > 0; --level) {
            for (auto const& ptr : nodes) {
                Node& node = *ptr;
                if (node.level == level) {
                    Block& parent = nodes[node.parent]->block;
                    parent.setChild(
                        parent.phi, node.octant, node.block.restrictField(node.block.phi));
                }
            }
        }
        for (int level = 0; level <= maximumLevel; ++level) {
            for (auto const& ptr : nodes) {
                Node& node = *ptr;
                if (node.level != level) {
                    continue;
                }
                for (int i = 0; i <= n + 1; ++i) {
                    for (int j = 0; j <= n + 1; ++j) {
                        for (int k = 0; k <= n + 1; ++k) {
                            if (i != 0 && i != n + 1 && j != 0 && j != n + 1 && k != 0 &&
                                k != n + 1) {
                                continue;
                            }
                            auto const p = position(node, i, j, k);
                            if (outside(p)) {
                                node.block.phi(i, j, k) = boundary(p);
                                continue;
                            }
                            int owner = find(level, p);
                            if (owner >= 0) {
                                Node const& from = *nodes[owner];
                                node.block.phi(i, j, k) =
                                    Block::sampleNearestCell(from.block.phi, logical(from, p));
                            } else {
                                owner = find(level - 1, p);
                                require(
                                    owner >= 0, "Fixture requires a 2:1 balanced coarse neighbor");
                                Node const& from = *nodes[owner];
                                node.block.phi(i, j, k) =
                                    Block::quadraticInterpolate(from.block.phi, logical(from, p));
                            }
                        }
                    }
                }
            }
        }
    }

    void syncCorrection(int level, bool injectCoarse) {
        for (auto const& ptr : nodes) {
            Node& node = *ptr;
            if (node.level != level) {
                continue;
            }
            for (int i = 0; i <= n + 1; ++i) {
                for (int j = 0; j <= n + 1; ++j) {
                    for (int k = 0; k <= n + 1; ++k) {
                        if (i != 0 && i != n + 1 && j != 0 && j != n + 1 && k != 0 &&
                            k != n + 1) {
                            continue;
                        }
                        auto const p = position(node, i, j, k);
                        double value = 0.0;
                        if (!outside(p)) {
                            int owner = find(level, p);
                            if (owner >= 0) {
                                Node const& from = *nodes[owner];
                                value = Block::sampleNearestCell(
                                    from.block.correction, logical(from, p));
                            } else if (injectCoarse) {
                                owner = find(level - 1, p);
                                require(owner >= 0, "Missing coarse correction halo owner");
                                Node const& from = *nodes[owner];
                                value = Block::quadraticInterpolate(
                                    from.block.correction, logical(from, p));
                            }
                        }
                        node.block.correction(i, j, k) = value;
                    }
                }
            }
        }
    }

    template <class Callback>
    void interfaces(int coarseLevel, Callback callback) {
        for (auto const& ptr : nodes) {
            Node const& fine = *ptr;
            if (fine.level != coarseLevel + 1) {
                continue;
            }
            for (int axis = 0; axis < NDIM; ++axis) {
                int const a = (axis + 1) % NDIM;
                int const b = (axis + 2) % NDIM;
                for (int side = 0; side < 2; ++side) {
                    std::array<int, NDIM> index{{n / 2, n / 2, n / 2}};
                    index[axis] = side == 0 ? 0 : n + 1;
                    auto const p = position(fine, index[0], index[1], index[2]);
                    if (outside(p) || find(fine.level, p) >= 0) {
                        continue;
                    }
                    int const owner = find(coarseLevel, p);
                    require(owner >= 0, "Missing coarse interface owner");
                    Node& coarse = *nodes[owner];
                    double const plane = (fine.location[axis] + side) * n * fine.block.dx;
                    int const face = static_cast<int>(std::llround(plane / coarse.block.dx)) -
                        coarse.location[axis] * n + 1;
                    int const t0 = fine.location[a] * n / 2 - coarse.location[a] * n;
                    int const t1 = fine.location[b] * n / 2 - coarse.location[b] * n;
                    callback(coarse.block, fine.block, axis, side, face, t0, t1);
                }
            }
        }
    }

    void computeForces() {
        for (auto const& node : nodes) {
            node->block.computeForces();
        }
        for (int level = maximumLevel - 1; level >= 0; --level) {
            interfaces(level,
                [](Block& coarse, Block const& fine, int axis, int side, int face, int t0, int t1) {
                    coarse.setRestrictedForce(
                        axis, face, t0, t1, fine.coarseFaceForces(axis, side));
                });
        }
    }

    double residual() const {
        double result = 0.0;
        for (auto const& node : nodes) {
            result += node->block.residualNorm();
        }
        return result;
    }

    void stage(PoissonStage operation, int level) {
        if (operation == PoissonStage::initialize) {
            for (auto const& node : nodes) {
                node->block.initializeCorrection();
            }
        } else if (operation == PoissonStage::restrict_residual) {
            for (auto const& node : nodes) {
                if (node->level == level + 1) {
                    Block& parent = nodes[node->parent]->block;
                    parent.setChild(parent.correctionRhs, node->octant,
                        node->block.restrictCorrectionResidual());
                }
            }
        } else if (operation == PoissonStage::adjust_rhs) {
            interfaces(level,
                [](Block& coarse, Block const& fine, int axis, int side, int face, int t0, int t1) {
                    coarse.addCorrectionFluxRhs(
                        axis, face, t0, t1, fine.coarseCorrectionFaceJumps(axis, side));
                });
        } else if (operation == PoissonStage::halo_down || operation == PoissonStage::halo_up) {
            syncCorrection(level, operation == PoissonStage::halo_up);
        } else if (operation == PoissonStage::smooth) {
            for (auto const& node : nodes) {
                if (node->level == level) {
                    node->block.relax();
                }
            }
        } else if (operation == PoissonStage::prolong) {
            for (auto const& node : nodes) {
                if (node->level != level) {
                    continue;
                }
                Block const& parent = nodes[node->parent]->block;
                node->block.addCorrection(parent.prolongatedCorrection(node->octant));
            }
        } else if (operation == PoissonStage::retire) {
            for (auto const& node : nodes) {
                node->block.retireCorrection();
            }
        } else {
            throw std::runtime_error("Unknown shared Poisson cycle stage");
        }
    }

    void cycle() {
        octotiger::multigrid::poissonCycle(maximumLevel,
            [this](PoissonStage operation, int level) { stage(operation, level); });
        syncPhi();
        computeForces();
    }

    void fillExact() {
        for (auto const& node : nodes) {
            for (int i = 1; i <= n; ++i) {
                for (int j = 1; j <= n; ++j) {
                    for (int k = 1; k <= n; ++k) {
                        node->block.phi(i, j, k) = boundary(position(*node, i, j, k));
                    }
                }
            }
        }
        syncPhi();
        computeForces();
    }

    double conservationError() const {
        double interiorDivergence = 0.0;
        double boundaryFlux = 0.0;
        for (auto const& ptr : nodes) {
            Node const& node = *ptr;
            if (node.block.refined) {
                continue;
            }
            double const h = node.block.dx;
            for (int i = 1; i <= n; ++i) {
                for (int j = 1; j <= n; ++j) {
                    for (int k = 1; k <= n; ++k) {
                        interiorDivergence +=
                            (node.block.physicalResidual(i, j, k) + node.block.source(i, j, k)) *
                            h * h * h;
                    }
                }
            }
            int const count = 1 << node.level;
            for (int axis = 0; axis < NDIM; ++axis) {
                int const a = (axis + 1) % NDIM;
                int const b = (axis + 2) % NDIM;
                for (int side = 0; side < 2; ++side) {
                    if (node.location[axis] != (side == 0 ? 0 : count - 1)) {
                        continue;
                    }
                    for (int j = 1; j <= n; ++j) {
                        for (int k = 1; k <= n; ++k) {
                            std::array<int, NDIM> cell{};
                            cell[axis] = side == 0 ? 1 : n + 1;
                            cell[a] = j;
                            cell[b] = k;
                            boundaryFlux += (side == 0 ? 1.0 : -1.0) * h * h *
                                node.block.force[axis](cell[0], cell[1], cell[2]);
                        }
                    }
                }
            }
        }
        return std::abs(interiorDivergence - boundaryFlux);
    }
};

void testDiscreteSolve(std::string const& topology) {
    bool const deep = topology == "deep_uniform";
    Fixture mesh(deep ? 8 : 6);
    if (deep) {
        mesh.refineUniformly(4);
    } else if (topology == "amr_interior") {
        mesh.refineUniformly(2);
        mesh.refine(mesh.find(2, {{0.375, 0.375, 0.375}}));
    } else {
        mesh.refine(0);
        if (topology == "amr_corner" || topology == "amr_nested") {
            mesh.refine(1);
        }
        if (topology == "amr_nested") {
            mesh.refine(mesh.find(2, {{0.125, 0.125, 0.125}}));
        }
    }
    mesh.fillExact();
    double const conservationTolerance = deep ? 5.e-11 : 2.e-11;
    require(mesh.conservationError() < conservationTolerance,
        "Composite face fluxes are not conservative");
    std::vector<Field> expected;
    for (auto const& node : mesh.nodes) {
        expected.push_back(node->block.phi);
        for (int i = 1; i <= mesh.n; ++i) {
            for (int j = 1; j <= mesh.n; ++j) {
                for (int k = 1; k <= mesh.n; ++k) {
                    node->block.source(i, j, k) = node->block.physicalResidual(i, j, k);
                }
            }
        }
        node->block.phi.fill(0.0);
    }
    mesh.syncPhi();
    mesh.computeForces();
    double const initial = mesh.residual();
    int cycles = 0;
    double const relativeTolerance = deep ? 1.e-9 : 1.e-10;
    int const maximumCycles = deep ? 120 : 500;
    while (cycles < maximumCycles && mesh.residual() > relativeTolerance * initial) {
        mesh.cycle();
        ++cycles;
    }
    double error = 0.0;
    for (size_t p = 0; p < mesh.nodes.size(); ++p) {
        Block const& block = mesh.nodes[p]->block;
        if (block.refined) {
            continue;
        }
        for (int i = 1; i <= mesh.n; ++i) {
            for (int j = 1; j <= mesh.n; ++j) {
                for (int k = 1; k <= mesh.n; ++k) {
                    error = std::max(error, std::abs(block.phi(i, j, k) - expected[p](i, j, k)));
                }
            }
        }
    }
    std::cout << topology << " cycles=" << cycles
              << " relative_residual=" << mesh.residual() / initial << " max_error=" << error
              << '\n';
    require(mesh.residual() <= relativeTolerance * initial,
        "Shared residual-correction V-cycle did not converge");
    require(error < (deep ? 2.e-8 : 2.e-9),
        "Converged discrete manufactured potential is incorrect");
    require(mesh.conservationError() < conservationTolerance,
        "Converged composite flux balance failed");
}

double continuumError(int n) {
    Fixture mesh(n);
    mesh.refine(0);
    mesh.refine(1);
    double const pi = std::acos(-1.0);
    for (auto const& node : mesh.nodes) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                for (int k = 1; k <= n; ++k) {
                    auto const p = mesh.position(*node, i, j, k);
                    node->block.source(i, j, k) = -3.0 * pi * pi * std::sin(pi * p[0]) *
                        std::sin(pi * p[1]) * std::sin(pi * p[2]);
                }
            }
        }
    }
    mesh.syncPhi();
    mesh.computeForces();
    double const initial = mesh.residual();
    for (int iteration = 0; iteration < 500 && mesh.residual() > 1.e-10 * initial; ++iteration) {
        mesh.cycle();
    }
    require(mesh.residual() < 1.e-10 * initial, "Continuum manufactured solve failed");
    double error = 0.0;
    for (auto const& node : mesh.nodes) {
        if (node->block.refined) {
            continue;
        }
        double const volume = std::pow(node->block.dx, 3);
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                for (int k = 1; k <= n; ++k) {
                    double const difference =
                        node->block.phi(i, j, k) - smoothSolution(mesh.position(*node, i, j, k));
                    error += volume * difference * difference;
                }
            }
        }
    }
    return std::sqrt(error);
}

}    // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--deep") {
            testDiscreteSolve("deep_uniform");
            std::cout << "Deep uniform multigrid regression passed\n";
            return 0;
        }
        require(argc == 1, "Expected no arguments or --deep");
        testTransfersAndStencil();
        testDiscreteSolve("uniform");
        testDiscreteSolve("amr_corner");
        testDiscreteSolve("amr_nested");
        testDiscreteSolve("amr_interior");
        double const coarse = continuumError(4);
        double const fine = continuumError(8);
        std::cout << "AMR continuum L2 errors=" << coarse << ',' << fine
                  << " ratio=" << coarse / fine << '\n';
        require(coarse / fine > 3.2, "AMR manufactured solution did not approach second order");
        std::cout << "Multigrid numerical checks passed\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
