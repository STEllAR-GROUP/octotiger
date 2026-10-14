#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "octotiger/radiation/implicit_source.hpp"

namespace radiation_accelerator {

enum class Backend { cpu, cuda, kokkos };
Backend backend(const std::string& name);

// Host-side AMR continues to own these arrays. All flattened grid arrays use
// field-major storage with n^3 entries per field; U stores physical F, not F/c.
struct Grid {
    int n, bw;
    double dx, c, reduced_c_factor, dt, omega;
};

struct CouplingInput {
    double radiation[4]; // E, F/c
    double momentum[3];
    double gas_energy, internal_energy, density;
    double absorption, extinction, log_alpha;
};

struct CouplingOutput {
    double radiation[4]; // E, F/c
    double momentum[3];
    double gas_energy, internal_energy;
};

// Device failures throw on the host before output arrays are committed.
void compute_flux(Backend, Grid, const std::vector<double>& state,
    const std::vector<double>& coordinates, std::vector<double>& flux);
// before_rotation contains the transported Fx,Fy (two fields), for the host's
// long-double conservation reduction. Ghost state is preserved.
void advance(Backend, Grid, std::vector<double>& state,
    const std::vector<double>& flux, std::vector<double>& before_rotation);
void couple(Backend, const std::vector<CouplingInput>& input,
    double dt, double c, double reduced_c_factor, bool marshak,
    std::vector<CouplingOutput>& output);
// Solve the fully implicit source stage Y=rhs+dt*S(Y). Invalid parameters or
// nonphysical/nonconvergent solutions return false and retain the ENTIRE batch.
// Device/runtime failures still throw. A successful empty batch clears output.
bool couple_implicit(Backend, const std::vector<CouplingInput>& input,
    double dt, double c, double reduced_c_factor, bool marshak,
    std::vector<CouplingOutput>& output);

// Called inside the HPX runtime. Finalize after callers finish and before any
// externally owned Kokkos finalization; cached device storage is released here.
// These functions only own Kokkos itself if they initialize it.
void initialize(Backend);
void finalize();

namespace detail {
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
inline bool implicit_couple_cell(const CouplingInput& input, double dt, double c,
    double reduced_c_factor, bool marshak, CouplingOutput& output)
{
    namespace Source = radiation::implicit_source;
    Source::State<double> rhs{}, result{};
    for (int f = 0; f < 4; ++f) {
        rhs.radiation[f] = input.radiation[f];
    }
    for (int d = 0; d < 3; ++d) {
        rhs.momentum[d] = input.momentum[d];
    }
    rhs.gas_energy = input.gas_energy;
    rhs.internal_energy = input.internal_energy;
    const Source::Parameters<double> parameters{input.density, input.absorption,
        input.extinction, input.log_alpha, c, reduced_c_factor, marshak};
    if (!Source::solve(rhs, parameters, dt, result)) {
        return false;
    }
    for (int f = 0; f < 4; ++f) {
        output.radiation[f] = result.radiation[f];
    }
    for (int d = 0; d < 3; ++d) {
        output.momentum[d] = result.momentum[d];
    }
    output.gas_energy = result.gas_energy;
    output.internal_energy = result.internal_energy;
    return true;
}

void cuda_flux(Grid, const std::vector<double>&, const std::vector<double>&, std::vector<double>&);
void cuda_advance(Grid, std::vector<double>&, const std::vector<double>&, std::vector<double>&);
void cuda_couple(const std::vector<CouplingInput>&, double, double, double, bool,
    std::vector<CouplingOutput>&);
bool cuda_couple_implicit(const std::vector<CouplingInput>&, double, double, double, bool,
    std::vector<CouplingOutput>&);
void cuda_finalize();
void kokkos_flux(Grid, const std::vector<double>&, const std::vector<double>&, std::vector<double>&);
void kokkos_advance(Grid, std::vector<double>&, const std::vector<double>&, std::vector<double>&);
void kokkos_couple(const std::vector<CouplingInput>&, double, double, double, bool,
    std::vector<CouplingOutput>&);
bool kokkos_couple_implicit(const std::vector<CouplingInput>&, double, double, double, bool,
    std::vector<CouplingOutput>&);
void kokkos_initialize();
void kokkos_finalize();
} // namespace detail
} // namespace radiation_accelerator
