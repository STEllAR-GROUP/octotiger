// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include "octotiger/radiation/accelerator.hpp"
#include "octotiger/radiation/accelerator_math.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace radiation_accelerator {
namespace {
size_t cells(Grid grid)
{
    if (grid.n <= 0 || grid.bw < 2 || grid.bw > (grid.n - 1) / 2 ||
        !std::isfinite(grid.dx) || grid.dx <= 0 || !std::isfinite(grid.c) || grid.c <= 0 ||
        !std::isfinite(grid.dt) || grid.dt < 0 || !std::isfinite(grid.omega) ||
        !std::isfinite(grid.reduced_c_factor) || grid.reduced_c_factor <= 0 || grid.reduced_c_factor > 1) {
        throw std::invalid_argument("Invalid radiation accelerator grid or parameters");
    }
    const size_t n = static_cast<size_t>(grid.n);
    if (n > std::numeric_limits<size_t>::max() / n / n / (12 * sizeof(double))) {
        throw std::invalid_argument("Radiation accelerator grid size overflows");
    }
    if (n * n * n > size_t(std::numeric_limits<int>::max()) / 12) {
        throw std::invalid_argument("Radiation accelerator grid exceeds signed kernel index range");
    }
    return n * n * n;
}

void check(bool valid)
{
    if (!valid) {
        throw std::runtime_error("Radiation accelerator cell failed its numerical validity checks; outputs retained");
    }
}
} // namespace

Backend backend(const std::string& name)
{
    if (name == "cpu") {
        return Backend::cpu;
    }
#ifdef OCTOTIGER_HAVE_RADIATION_CUDA
    if (name == "cuda") {
        return Backend::cuda;
    }
#endif
#ifdef OCTOTIGER_HAVE_RADIATION_KOKKOS
    if (name == "kokkos") {
        return Backend::kokkos;
    }
#endif
    throw std::invalid_argument("Radiation backend '" + name + "' is unknown or was not enabled in this build");
}

void initialize(Backend selected)
{
    switch (selected) {
    case Backend::cpu:
        return;
#ifdef OCTOTIGER_HAVE_RADIATION_CUDA
    case Backend::cuda:
        return;
#endif
#ifdef OCTOTIGER_HAVE_RADIATION_KOKKOS
    case Backend::kokkos:
        detail::kokkos_initialize();
        return;
#endif
    default:
        throw std::invalid_argument("Radiation backend unavailable in this build");
    }
}

void finalize()
{
#ifdef OCTOTIGER_HAVE_RADIATION_CUDA
    detail::cuda_finalize();
#endif
#ifdef OCTOTIGER_HAVE_RADIATION_KOKKOS
    detail::kokkos_finalize();
#endif
}

void compute_flux(Backend selected, Grid grid, const std::vector<double>& state,
    const std::vector<double>& coordinates, std::vector<double>& flux)
{
    const size_t count = cells(grid);
    if (state.size() != 4 * count || coordinates.size() != 3 * count) {
        throw std::invalid_argument("Radiation accelerator flux array shape differs from grid");
    }
    switch (selected) {
    case Backend::cpu: {
        std::vector<double> primitives(4 * count), result(12 * count, 0.);
        for (size_t r = 0; r < count; ++r) {
            check(math::primitives(grid, state.data(), primitives.data(), r));
        }
        for (int normal = 0; normal < 3; ++normal) {
            for (size_t r = 0; r < count; ++r) {
                check(math::face_flux(grid, primitives.data(), coordinates.data(), result.data(), normal, r));
            }
        }
        flux.swap(result);
        return;
    }
#ifdef OCTOTIGER_HAVE_RADIATION_CUDA
    case Backend::cuda:
        detail::cuda_flux(grid, state, coordinates, flux);
        return;
#endif
#ifdef OCTOTIGER_HAVE_RADIATION_KOKKOS
    case Backend::kokkos:
        detail::kokkos_flux(grid, state, coordinates, flux);
        return;
#endif
    default:
        throw std::invalid_argument("Radiation backend unavailable in this build");
    }
}

void advance(Backend selected, Grid grid, std::vector<double>& state,
    const std::vector<double>& flux, std::vector<double>& before_rotation)
{
    const size_t count = cells(grid);
    if (state.size() != 4 * count || flux.size() != 12 * count || &state == &before_rotation) {
        throw std::invalid_argument("Radiation accelerator advance array shape or output alias is invalid");
    }
    switch (selected) {
    case Backend::cpu: {
        auto result = state;
        std::vector<double> before(2 * count, 0.);
        for (size_t r = 0; r < count; ++r) {
            check(math::advance_cell(grid, result.data(), flux.data(), before.data(), r));
        }
        state.swap(result);
        before_rotation.swap(before);
        return;
    }
#ifdef OCTOTIGER_HAVE_RADIATION_CUDA
    case Backend::cuda:
        detail::cuda_advance(grid, state, flux, before_rotation);
        return;
#endif
#ifdef OCTOTIGER_HAVE_RADIATION_KOKKOS
    case Backend::kokkos:
        detail::kokkos_advance(grid, state, flux, before_rotation);
        return;
#endif
    default:
        throw std::invalid_argument("Radiation backend unavailable in this build");
    }
}

void couple(Backend selected, const std::vector<CouplingInput>& input, double dt,
    double c, double reduced_c_factor, bool marshak, std::vector<CouplingOutput>& output)
{
    if (!std::isfinite(dt) || dt < 0 || !std::isfinite(c) || c <= 0 ||
        !std::isfinite(reduced_c_factor) || reduced_c_factor <= 0 || reduced_c_factor > 1) {
        throw std::invalid_argument("Invalid radiation accelerator coupling parameters");
    }
    switch (selected) {
    case Backend::cpu: {
        std::vector<CouplingOutput> result(input.size());
        for (size_t r = 0; r < input.size(); ++r) {
            check(math::couple_cell(input[r], dt, c, reduced_c_factor, marshak, result[r]));
        }
        output.swap(result);
        return;
    }
#ifdef OCTOTIGER_HAVE_RADIATION_CUDA
    case Backend::cuda:
        detail::cuda_couple(input, dt, c, reduced_c_factor, marshak, output);
        return;
#endif
#ifdef OCTOTIGER_HAVE_RADIATION_KOKKOS
    case Backend::kokkos:
        detail::kokkos_couple(input, dt, c, reduced_c_factor, marshak, output);
        return;
#endif
    default:
        throw std::invalid_argument("Radiation backend unavailable in this build");
    }
}

bool couple_implicit(Backend selected, const std::vector<CouplingInput>& input, double dt,
    double c, double reduced_c_factor, bool marshak, std::vector<CouplingOutput>& output)
{
    if (!std::isfinite(dt) || dt < 0 || !std::isfinite(c) || c <= 0 ||
        !std::isfinite(reduced_c_factor) || reduced_c_factor <= 0 || reduced_c_factor > 1) {
        return false;
    }
    switch (selected) {
    case Backend::cpu: {
        std::vector<CouplingOutput> result(input.size());
        for (size_t r = 0; r < input.size(); ++r) {
            if (!detail::implicit_couple_cell(input[r], dt, c, reduced_c_factor, marshak, result[r])) {
                return false;
            }
        }
        output.swap(result);
        return true;
    }
#ifdef OCTOTIGER_HAVE_RADIATION_CUDA
    case Backend::cuda:
        return detail::cuda_couple_implicit(input, dt, c, reduced_c_factor, marshak, output);
#endif
#ifdef OCTOTIGER_HAVE_RADIATION_KOKKOS
    case Backend::kokkos:
        return detail::kokkos_couple_implicit(input, dt, c, reduced_c_factor, marshak, output);
#endif
    default:
        throw std::invalid_argument("Radiation backend unavailable in this build");
    }
}
} // namespace radiation_accelerator
