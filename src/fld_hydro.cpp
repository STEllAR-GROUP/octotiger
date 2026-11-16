#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "octotiger/fld_closure.hpp"
#include "octotiger/fld_configuration.hpp"
#include "octotiger/fld_hydro.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/options.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/hydro_impl/flux.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct.hpp"

namespace {

using vector3 = std::array<double, NDIM>;
using tensor3 = std::array<vector3, NDIM>;

vector3 radiation_gradient(std::vector<real> const& energy, integer const index, double const dx) {
    vector3 gradient{};
    for (int dim = 0; dim < NDIM; ++dim) {
        gradient[dim] = (energy[index + H_DN[dim]] - energy[index - H_DN[dim]]) / (2.0 * dx);
    }
    return gradient;
}

double extinction(std::vector<std::vector<safe_real>> const& state, integer const index) {
    double const rho = state[rho_i][index];
    auto const material = octotiger::fld::fld_material_state(state, index);
    return octotiger::fld::evaluate_opacity(octotiger::fld::fld_opacity_parameters(), rho,
        material.temperature, material.electronFraction, material.ionChargeSquared).rosselandExtinction;
}

}    // namespace

timestep_t grid::compute_fld_fluxes() {
    // The ordinary hydro reconstruction, face quadrature, and HLL solver also
    // transport the separate radiation scalar. Only auxiliary views are added.
    using geometry = cell_geometry<NDIM, INX>;
    auto const& reconstructed = hydro.reconstruct(U, X, omega);
    std::vector<std::vector<safe_real>> radiation_recon(
        geometry::NDIR, std::vector<safe_real>(H_N3));
    std::vector<std::vector<double>> const no_discontinuities;
    hydro.reconstruct_ppm(radiation_recon, *fld_energy_, false, false, no_discontinuities);

    double pressure_work_speed = 0.0;
    std::array<std::vector<real>, NDIM> acoustic;
    std::array<std::vector<real>, NDIM> radiation_flux;
    for (int dim = 0; dim < NDIM; ++dim) {
        acoustic[dim].resize(H_N3);
        radiation_flux[dim].resize(H_N3);
    }
    for (integer i = 1; i < H_NX - 1; ++i) {
        for (integer j = 1; j < H_NX - 1; ++j) {
            for (integer k = 1; k < H_NX - 1; ++k) {
                integer const index = hindex(i, j, k);
                double const energy = (*fld_energy_)[index];
                auto const closure = octotiger::fld::closure(
                    energy, radiation_gradient(*fld_energy_, index, dx), extinction(U, index));
                if (i >= H_BW && i < H_NX - H_BW && j >= H_BW && j < H_NX - H_BW && k >= H_BW &&
                    k < H_NX - H_BW && energy > 0.0) {
                    tensor3 minus{};
                    tensor3 plus{};
                    for (int axis = 0; axis < NDIM; ++axis) {
                        integer const lower = index - H_DN[axis];
                        integer const upper = index + H_DN[axis];
                        for (int component = 0; component < NDIM; ++component) {
                            minus[axis][component] = U[sx_i + component][lower] / U[rho_i][lower];
                            plus[axis][component] = U[sx_i + component][upper] / U[rho_i][upper];
                        }
                    }
                    double const work = octotiger::fld::pressure_work(
                        closure.pressure, octotiger::fld::velocity_gradient(minus, plus, dx));
                    // Reserve half the radiation energy for advection. The node's
                    // dt = cfl * dx / a then bounds explicit expansion work.
                    pressure_work_speed = std::max(
                        pressure_work_speed, 2.0 * opts().cfl * dx * std::max(work, 0.0) / energy);
                }
                for (int dim = 0; dim < NDIM; ++dim) {
                    double const f = energy > 0.0 ? closure.pressure[dim][dim] / energy : 1.0 / 3.0;
                    acoustic[dim][index] =
                        octotiger::fld::radiation_sound_speed_squared(1.0, 1.0, closure.lambda, f);
                }
            }
        }
    }
    hydro::flux_type flux(
        NDIM, std::vector<std::vector<safe_real>>(opts().n_fields, std::vector<safe_real>(H_N3)));
    timestep_t result =
        hydro.flux(U, reconstructed, flux, X, omega, &radiation_recon, &acoustic, &radiation_flux);
    for (int dim = 0; dim < NDIM; ++dim) {
        for (integer i = 0; i <= INX; ++i) {
            for (integer j = 0; j <= INX; ++j) {
                for (integer k = 0; k <= INX; ++k) {
                    integer const face = findex(i, j, k);
                    integer const cell = hindex(i + H_BW, j + H_BW, k + H_BW);
                    for (integer field = 0; field < opts().n_fields; ++field) {
                        F[dim][field][face] = flux[dim][field][cell];
                    }
                    double mass_flux = 0.0;
                    for (integer species = 0; species < opts().n_species; ++species) {
                        mass_flux += F[dim][spc_i + species][face];
                    }
                    F[dim][rho_i][face] = mass_flux;
                    fld_flux_[dim][face] = radiation_flux[dim][cell];
                }
            }
        }
    }
    result.a = std::max(result.a, pressure_work_speed);
    return result;
}

void grid::compute_fld_sources() {
    for (integer i = H_BW; i < H_NX - H_BW; ++i) {
        for (integer j = H_BW; j < H_NX - H_BW; ++j) {
            for (integer k = H_BW; k < H_NX - H_BW; ++k) {
                integer const index = hindex(i, j, k);
                integer const interior = h0index(i - H_BW, j - H_BW, k - H_BW);
                auto const gradient = radiation_gradient(*fld_energy_, index, dx);
                auto const closure =
                    octotiger::fld::closure((*fld_energy_)[index], gradient, extinction(U, index));
                tensor3 minus{};
                tensor3 plus{};
                vector3 force{};
                for (int dim = 0; dim < NDIM; ++dim) {
                    force[dim] = -closure.lambda * gradient[dim];
                    dUdt[sx_i + dim][interior] += force[dim];
                    dUdt[egas_i][interior] += U[sx_i + dim][index] / U[rho_i][index] * force[dim];
                    integer const lower = index - H_DN[dim];
                    integer const upper = index + H_DN[dim];
                    for (int component = 0; component < NDIM; ++component) {
                        minus[dim][component] = U[sx_i + component][lower] / U[rho_i][lower];
                        plus[dim][component] = U[sx_i + component][upper] / U[rho_i][upper];
                    }
                }
                dUdt[lx_i][interior] += X[YDIM][index] * force[ZDIM] - X[ZDIM][index] * force[YDIM];
                dUdt[ly_i][interior] += X[ZDIM][index] * force[XDIM] - X[XDIM][index] * force[ZDIM];
                dUdt[lz_i][interior] += X[XDIM][index] * force[YDIM] - X[YDIM][index] * force[XDIM];
                fld_rhs_[interior] = -octotiger::fld::pressure_work(
                    closure.pressure, octotiger::fld::velocity_gradient(minus, plus, dx));
            }
        }
    }
}

void grid::compute_fld_dudt() {
    for (integer i = 0; i < INX; ++i) {
        for (integer j = 0; j < INX; ++j) {
            for (integer k = 0; k < INX; ++k) {
                integer const cell = h0index(i, j, k);
                integer const face = findex(i, j, k);
                for (int dim = 0; dim < NDIM; ++dim) {
                    fld_rhs_[cell] -=
                        (fld_flux_[dim][face + F_DN[dim]] - fld_flux_[dim][face]) / dx;
                }
            }
        }
    }
}

void grid::next_fld_energy(integer const rk, real const dt) {
    double outflow_rate = 0.0;
    for (integer i = 0; i < INX; ++i) {
        for (integer j = 0; j < INX; ++j) {
            for (integer k = 0; k < INX; ++k) {
                integer const cell = hindex(i + H_BW, j + H_BW, k + H_BW);
                integer const interior = h0index(i, j, k);
                double const trial = (*fld_energy_)[cell] + dt * fld_rhs_[interior];
                double const updated =
                    (1.0 - rk_beta[rk]) * fld_energy0_[cell] + rk_beta[rk] * trial;
                if (updated < 0.0 || !std::isfinite(updated)) {
                    throw std::runtime_error("Explicit FLD hydro update lost radiation positivity; "
                                             "reduce hydro timestep");
                }
                (*fld_energy_)[cell] = updated;
                std::array<integer, NDIM> const coordinate{i, j, k};
                for (int dim = 0; dim < NDIM; ++dim) {
                    integer const face = findex(i, j, k);
                    if (coordinate[dim] == 0 && X[dim][cell] < -scaling_factor + dx) {
                        outflow_rate -= dx * dx * fld_flux_[dim][face];
                    }
                    if (coordinate[dim] == INX - 1 && X[dim][cell] > scaling_factor - dx) {
                        outflow_rate += dx * dx * fld_flux_[dim][face + F_DN[dim]];
                    }
                }
            }
        }
    }
    fld_outflow_ =
        (1.0 - rk_beta[rk]) * fld_outflow0_ + rk_beta[rk] * (fld_outflow_ + dt * outflow_rate);
}

void grid::complete_fld_amr_boundary() {
    for (integer i = 0; i < H_NX; ++i) {
        for (integer j = 0; j < H_NX; ++j) {
            for (integer k = 0; k < H_NX; ++k) {
                integer const coarse = hSindex((i + H_BW) / 2, (j + H_BW) / 2, (k + H_BW) / 2);
                if (is_coarse[coarse]) {
                    // Conservative nonnegative injection at coarse/fine ghost interfaces.
                    (*fld_energy_)[hindex(i, j, k)] = fld_shadow_[coarse];
                }
            }
        }
    }
}

void grid::set_fld_physical_boundary(geo::face const& face) {
    auto const dim = face.get_dimension();
    bool const lower = face.get_side() == geo::MINUS;
    for (integer i = 0; i < H_NX; ++i) {
        for (integer j = 0; j < H_NX; ++j) {
            for (integer k = 0; k < H_NX; ++k) {
                std::array<integer, NDIM> source{i, j, k};
                integer const normal = source[dim];
                if ((lower && normal >= H_BW) || (!lower && normal < H_NX - H_BW)) {
                    continue;
                }
                if (opts().reflect_bc) {
                    source[dim] = lower ? 2 * H_BW - normal - 1 : 2 * (H_NX - H_BW) - normal - 1;
                } else {
                    source[dim] = lower ? H_BW : H_NX - H_BW - 1;
                }
                (*fld_energy_)[hindex(i, j, k)] =
                    (*fld_energy_)[hindex(source[0], source[1], source[2])];
            }
        }
    }
}
