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

using octotiger::multigrid::Block;
using octotiger::multigrid::Field;
using Position = std::array<double, 3>;

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
// in the node-server driver. Every fixture stage mirrors the historical
// communication barrier, including fixed halos during a red/black pair.
struct Node
{
    Block block;
    int level;
    std::array<int, 3> location;
    int parent = -1;
    int octant = 0;
    Node(int n, int lev, std::array<int, 3> loc, int par, int oct)
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
    std::function<double(Position const&)> boundary = smoothSolution;

    explicit Fixture(int interior)
      : n(interior) {
        nodes.push_back(std::make_unique<Node>(n, 0, std::array<int, 3>{{0, 0, 0}}, -1, 0));
    }

    void refine(int index) {
        Node& parent = *nodes.at(index);
        require(!parent.block.refined, "Fixture node was already refined");
        parent.block.refined = true;
        int const nextLevel = parent.level + 1;
        maximumLevel = std::max(maximumLevel, nextLevel);
        auto const location = parent.location;
        for (int octant = 0; octant < 8; ++octant) {
            std::array<int, 3> child{};
            for (int d = 0; d < 3; ++d) {
                child[d] = 2 * location[d] + ((octant >> d) & 1);
            }
            nodes.push_back(std::make_unique<Node>(n, nextLevel, child, index, octant));
        }
    }

    Position position(Node const& node, int i, int j, int k) const {
        std::array<int, 3> const index{{i, j, k}};
        Position result{};
        for (int d = 0; d < 3; ++d) {
            result[d] = node.block.dx * (n * node.location[d] + index[d] - 0.5);
        }
        return result;
    }

    Position logical(Node const& node, Position const& p) const {
        Position result{};
        for (int d = 0; d < 3; ++d) {
            result[d] = p[d] / node.block.dx - n * node.location[d] + 0.5;
        }
        return result;
    }

    int find(int level, Position const& p) const {
        for (size_t i = 0; i < nodes.size(); ++i) {
            Node const& node = *nodes[i];
            if (node.level != level) {
                continue;
            }
            bool inside = true;
            for (int d = 0; d < 3; ++d) {
                double const lower = node.location[d] * n * node.block.dx;
                inside = inside && p[d] >= lower && p[d] < lower + n * node.block.dx;
            }
            if (inside) {
                return static_cast<int>(i);
            }
        }
        return -1;
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
            for (int axis = 0; axis < 3; ++axis) {
                int const a = (axis + 1) % 3;
                int const b = (axis + 2) % 3;
                for (int side = 0; side < 2; ++side) {
                    for (int j = 1; j <= n; ++j) {
                        for (int k = 1; k <= n; ++k) {
                            std::array<int, 3> cell{};
                            cell[axis] = side == 0 ? 0 : n + 1;
                            cell[a] = j;
                            cell[b] = k;
                            auto const p = position(node, cell[0], cell[1], cell[2]);
                            if (outside(p)) {
                                node.block.correction(cell[0], cell[1], cell[2]) = 0.0;
                                continue;
                            }
                            int owner = find(level, p);
                            if (owner < 0 && injectCoarse) {
                                owner = find(level - 1, p);
                            }
                            if (owner >= 0) {
                                Node const& from = *nodes[owner];
                                node.block.correction(cell[0], cell[1], cell[2]) =
                                    Block::sampleNearestCell(from.block.correction, logical(from, p));
                            }
                        }
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
            for (int axis = 0; axis < 3; ++axis) {
                int const a = (axis + 1) % 3;
                int const b = (axis + 2) % 3;
                for (int side = 0; side < 2; ++side) {
                    std::array<int, 3> index{{n / 2, n / 2, n / 2}};
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

    void cycle() {
        for (int level = maximumLevel; level >= 0; --level) {
            for (auto const& node : nodes) {
                if (node->level == level) {
                    node->block.beginDown();
                }
            }
            interfaces(level,
                [](Block& coarse, Block const& fine, int axis, int side, int face, int t0, int t1) {
                    coarse.adjustFluxDefect(axis, face, t0, t1, fine.coarseFaceJumps(axis, side));
                });
            for (auto const& node : nodes) {
                if (node->level == level) {
                    node->block.saveCorrectionRhs();
                }
            }
            for (int sweep = 0; sweep < 2; ++sweep) {
                syncCorrection(level, false);
                for (auto const& node : nodes) {
                    if (node->level == level) {
                        node->block.relax();
                    }
                }
            }
            if (level > 0) {
                for (auto const& node : nodes) {
                    if (node->level == level) {
                        Block& parent = nodes[node->parent]->block;
                        parent.setChild(parent.correction, node->octant,
                            node->block.restrictField(node->block.correction));
                    }
                }
            }
        }
        for (int level = 0; level <= maximumLevel; ++level) {
            if (level > 0) {
                for (auto const& node : nodes) {
                    if (node->level == level) {
                        Block const& parent = nodes[node->parent]->block;
                        node->block.replaceCorrectionMean(
                            parent.childValues(parent.correction, node->octant));
                    }
                }
            }
            syncCorrection(level, true);
            for (int sweep = 0; sweep < 4; ++sweep) {
                syncCorrection(level, false);
                for (auto const& node : nodes) {
                    if (node->level == level) {
                        node->block.relax();
                    }
                }
            }
        }
        for (auto const& node : nodes) {
            node->block.retireCorrection();
        }
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
            for (int axis = 0; axis < 3; ++axis) {
                int const a = (axis + 1) % 3;
                int const b = (axis + 2) % 3;
                for (int side = 0; side < 2; ++side) {
                    if (node.location[axis] != (side == 0 ? 0 : count - 1)) {
                        continue;
                    }
                    for (int j = 1; j <= n; ++j) {
                        for (int k = 1; k <= n; ++k) {
                            std::array<int, 3> cell{};
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

void testDiscreteSolve(bool amr) {
    Fixture mesh(6);
    mesh.refine(0);
    if (amr) {
        mesh.refine(1);
    }
    mesh.fillExact();
    require(mesh.conservationError() < 2.e-12, "Composite face fluxes are not conservative");
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
    while (cycles < 500 && mesh.residual() > 1.e-10 * initial) {
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
    std::cout << (amr ? "AMR" : "uniform") << " cycles=" << cycles
              << " relative_residual=" << mesh.residual() / initial << " max_error=" << error
              << '\n';
    require(mesh.residual() <= 1.e-10 * initial, "Historical V-cycle did not converge");
    require(error < 2.e-9, "Converged discrete manufactured potential is incorrect");
    require(mesh.conservationError() < 2.e-12, "Converged composite flux balance failed");
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

int main() {
    try {
        testTransfersAndStencil();
        testDiscreteSolve(false);
        testDiscreteSolve(true);
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
