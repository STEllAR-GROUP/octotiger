// Copyright (c) 2026 AUTHORS. Boost Software License, Version 1.0.
// Offline initial-profile indicator sampling. No application evolution or AMR.
#include "octotiger/amr/gradient.hpp"
#include "test_problems/radiation/profiles.hpp"
#include "test_problems/radiation/hydro_benchmarks.hpp"
#include <stddef.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::invalid_argument("sample_initial CASE FINEST_LEVEL FIELD FLOOR BUFFER");
        }
        const std::string name = argv[1], field = argv[3];
        const int level = std::stoi(argv[2]), buffer = std::stoi(argv[5]);
        const double floor = std::stod(argv[4]);
        gradientRefinement::Parameters settings;
        settings.floor = floor;
        settings.buffer = buffer;
        settings.validate();
        if (level < 2 || level > 5 || (field != "density" && field != "radiation_energy")) {
            throw std::invalid_argument("Invalid sampling level or field");
        }
        const bool shock = name == "subcritical_shock" || name == "supercritical_shock";
        const bool blast = name == "radiating_blast";
        const bool sphere = name == "equilibrium_sphere";
        const bool gaussian = name == "gaussian_pulse";
        const bool wave = name == "streaming_wave", front = name == "streaming_front";
        if (!(shock || blast || sphere || gaussian || wave || front)) {
            throw std::invalid_argument("Unknown radiation case");
        }
        const bool periodic = !(shock || blast || sphere);
        const int n = 8 * (1 << (level - 1)), blocks = n / 8, halo = buffer + 1, width = n + 2 * halo;
        const auto problem = blast ? radiationHydroTests::Case::blastWave :
            name == "subcritical_shock" ? radiationHydroTests::Case::subcriticalShock :
                radiationHydroTests::Case::supercriticalShock;
        const double halfLength = shock || blast ? radiationHydroTests::setup(problem).halfLength : 1.;
        const double dx = 2 * halfLength / n;
        radiationTests::Parameters parameters;
        // Fixed-medium samples use c=1 and the unchanged template coordinates.
        // The application harness uses code_to_cm=physical_c to realize this.
        std::map<std::array<int, 3>, double> radialCache;
        std::map<int, double> shockCache;
        const auto coordinate = [&](int index) { return -halfLength + (index + .5) * dx; };
        const auto profile = [&](int i, int j, int k) {
            if (field == "density" && !(shock || blast)) {
                return 1.;
            }
            std::array<int, 3> indices{i, j, k};
            if (periodic) {
                for (int& index : indices) {
                    index = (index % n + n) % n;
                }
            } else if (blast) {
                for (int& index : indices) {
                    index = std::clamp(index, 0, n - 1);
                }
            } else if (shock) {
                if (indices[0] >= n) {
                    const auto state = radiationHydroTests::initialState(problem, {halfLength, 0, 0}, 0);
                    return field == "density" ? state.rho : state.radiation[0];
                }
                if (indices[0] < 0) {
                    indices[0] = -indices[0] - 1;
                }
                const auto found = shockCache.find(indices[0]);
                if (found != shockCache.end()) {
                    return found->second;
                }
                const auto state = radiationHydroTests::initialState(problem, {coordinate(indices[0]), 0, 0}, dx);
                const double value = field == "density" ? state.rho : state.radiation[0];
                shockCache.emplace(indices[0], value);
                return value;
            }
            radiationTests::Point x{coordinate(indices[0]), coordinate(indices[1]), coordinate(indices[2])};
            if (wave) {
                return radiationTests::streamingWave(x, 0, dx, parameters)[0];
            }
            if (front) {
                return radiationTests::streamingFront(x, 0, dx, parameters)[0];
            }
            // Radial cell averages are invariant under coordinate reflections
            // and permutations. This cache reduces quadrature work, not resolution.
            std::array<int, 3> key;
            for (int d = 0; d < 3; ++d) {
                key[d] = std::abs(2 * indices[d] + 1 - n);
            }
            std::sort(key.begin(), key.end());
            const auto found = radialCache.find(key);
            if (found != radialCache.end()) {
                return found->second;
            }
            for (int d = 0; d < 3; ++d) {
                x[d] = key[d] * dx / 2;
            }
            double value;
            if (blast) {
                const auto state = radiationHydroTests::initialState(problem, x, dx);
                value = field == "density" ? state.rho : state.radiation[0];
            } else if (sphere) {
                value = radiationTests::sphereAverage(x, dx, parameters)[0];
            } else {
                // Same exact periodized Gaussian integrals as the reference
                // generator at t=0. Nyquist removal is analytically zero here
                // by reflection symmetry; FFT roundoff is not reproduced.
                value = parameters.amplitude;
                for (double position : x) {
                    double one = 0;
                    for (int image = -4; image <= 4; ++image) {
                        one += .5 * std::sqrt(radiationTests::pi) * parameters.width / dx *
                            (std::erf((position + dx / 2 + image * parameters.length) / parameters.width) -
                             std::erf((position - dx / 2 + image * parameters.length) / parameters.width));
                    }
                    value *= one;
                }
                value += parameters.background;
            }
            radialCache.emplace(key, value);
            return value;
        };
        const auto index = [&](int i, int j, int k) {
            return size_t(k + halo) + size_t(width) * (j + halo + size_t(width) * (i + halo));
        };
        std::vector<double> values(size_t(width) * width * width);
        for (int i = -halo; i < n + halo; ++i) {
            for (int j = -halo; j < n + halo; ++j) {
                for (int k = -halo; k < n + halo; ++k) {
                    values[index(i, j, k)] = profile(i, j, k);
                }
            }
        }
        std::cout << std::setprecision(17) << "{\"case\":\"" << name << "\",\"finest_level\":" << level
            << ",\"base_level\":" << level - 1 << ",\"base_blocks\":" << blocks * blocks * blocks
            << ",\"field\":\"" << field << "\",\"floor\":" << floor << ",\"buffer\":" << buffer
            << ",\"dx\":" << dx << ",\"block_scores\":[";
        bool first = true;
        for (int bi = 0; bi < blocks; ++bi) {
            for (int bj = 0; bj < blocks; ++bj) {
                for (int bk = 0; bk < blocks; ++bk) {
                    long double absolute = 0, relative = 0;
                    for (int di = -buffer; di < 8 + buffer; ++di) {
                        for (int dj = -buffer; dj < 8 + buffer; ++dj) {
                            for (int dk = -buffer; dk < 8 + buffer; ++dk) {
                                if ((di < 0 || di >= 8) + (dj < 0 || dj >= 8) + (dk < 0 || dk >= 8) > 1) {
                                    continue;
                                }
                                const int i = 8 * bi + di, j = 8 * bj + dj, k = 8 * bk + dk;
                                const std::array<double, 6> neighbors{values[index(i-1,j,k)], values[index(i+1,j,k)],
                                    values[index(i,j-1,k)], values[index(i,j+1,k)], values[index(i,j,k-1)], values[index(i,j,k+1)]};
                                const double center = values[index(i,j,k)];
                                absolute = std::max(absolute, gradientRefinement::edgeGradient(center, neighbors, dx,
                                    gradientRefinement::Mode::absolute, floor));
                                relative = std::max(relative, gradientRefinement::edgeGradient(center, neighbors, dx,
                                    gradientRefinement::Mode::relative, floor));
                            }
                        }
                    }
                    std::cout << (first ? "" : ",") << "[" << static_cast<double>(absolute) << ","
                        << static_cast<double>(relative) << "]";
                    first = false;
                }
            }
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
