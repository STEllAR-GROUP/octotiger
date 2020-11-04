#include <iostream>
#include <vector>

#include "octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp"



void complete_hydro_amr_boundary_cpu(const double dx, const bool energy_only,
    const std::vector<std::vector<real>>& Ushad, const std::vector<std::atomic<int>>& is_coarse,
    const std::array<double, NDIM>& xmin, std::vector<std::vector<double>>& U) {
    //std::cout << "Calling hydro cpu version!" << std::endl;

    std::vector<double, recycler::aggressive_recycle_aligned<double, 32>> unified_uf(
        opts().n_fields * HS_N3 * 8);
    std::vector<double, recycler::aggressive_recycle_aligned<double, 32>> unified_u(
        opts().n_fields * H_N3);
    std::vector<double, recycler::aggressive_recycle_aligned<double, 32>> unified_ushad(
        opts().n_fields * HS_N3);
    // Create non-atomic copy
    std::vector<int, recycler::aggressive_recycle_aligned<int, 32>> coarse(HS_N3);

    for (int f = 0; f < opts().n_fields; f++) {
        if (!energy_only || f == egas_i) {
            std::copy(
                Ushad[f].begin(), Ushad[f].begin() + HS_N3, unified_ushad.begin() + f * HS_N3);
        }
    }

    for (int i = 0; i < HS_N3; i++) {
        coarse[i] = is_coarse[i];
    }
    constexpr int field_offset = HS_N3 * 8;


    // Phase 1: From UShad to Uf
    for (int i0 = 1; i0 < HS_NX - 1; i0++) {
        for (int j0 = 1; j0 < HS_NX - 1; j0++) {
            for (int k0 = 1; k0 < HS_NX - 1; k0++) {
              complete_hydro_amr_boundary_inner_loop(dx, energy_only, unified_ushad.data(), coarse.data(),
                  xmin.data(), unified_uf.data(), i0, j0, k0, opts().n_fields);
            }
        }
    }

    // Phase 2: Process U    // Phase 3: From Uf to U
    for (int f = 0; f < opts().n_fields; f++) {
        if (!energy_only || f == egas_i) {
            // std::copy(U[f].begin(), U[f].end(), unified_u.begin() + f * H_N3);
            for (int i = 0; i < H_NX; i++) {
                for (int j = 0; j < H_NX; j++) {
                    for (int k = 0; k < H_NX; k++) {
                        const int i0 = (i + H_BW) / 2;
                        const int j0 = (j + H_BW) / 2;
                        const int k0 = (k + H_BW) / 2;
                        const int iii0 = hSindex(i0, j0, k0);
                        const int iiir = hindex(i, j, k);
                        if (coarse[iii0]) {
                            int ir, jr, kr;
                            if HOST_CONSTEXPR (H_BW % 2 == 0) {
                                ir = i % 2;
                                jr = j % 2;
                                kr = k % 2;
                            } else {
                                ir = 1 - (i % 2);
                                jr = 1 - (j % 2);
                                kr = 1 - (k % 2);
                            }
                            const int oct_index = ir * 4 + jr * 2 + kr;
                            // unified_u[f * H_N3 + iiir] =
                            //    unified_uf[f * field_offset + 8 * iii0 + oct_index];
                            U[f][iiir] = unified_uf[f * field_offset + 8 * iii0 + oct_index];
                        }
                    }
                }
            }
            // std::copy(unified_u.begin() + f * H_N3, unified_u.begin() + f * H_N3 + H_N3,
            // U[f].begin());
        }
    }
    // Copy field back for compatibility with the rest of the code (for now)..
    /*for (int f = 0; f < opts().n_fields; f++) {
        if (!energy_only || f == egas_i) {
            for (int i = 0; i < H_NX; i++) {
                for (int j = 0; j < H_NX; j++) {
                    for (int k = 0; k < H_NX; k++) {
                        const int i0 = (i + H_BW) / 2;
                        const int j0 = (j + H_BW) / 2;
                        const int k0 = (k + H_BW) / 2;
                        const int iii0 = hSindex(i0, j0, k0);
                        if (coarse[iii0]) {
                            const int iiir = hindex(i, j, k);
                            U[f][iiir] = unified_u[f * H_N3 + iiir];
                        }
                    }
                }
            }
        }
    }*/
}
