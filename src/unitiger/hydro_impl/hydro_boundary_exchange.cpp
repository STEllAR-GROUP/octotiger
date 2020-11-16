//#pragma GCC push_options
//#pragma GCC optimize("unroll-loops")

#include <iostream>
#include <vector>

#include "octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp"
#include "octotiger/util/vec_scalar_host_wrapper.hpp"

void complete_hydro_amr_boundary_cpu(const double dx, const bool energy_only,
    const std::vector<std::vector<real>>& Ushad, const std::vector<std::atomic<int>>& is_coarse,
    const std::array<double, NDIM>& xmin, std::vector<std::vector<double>>& U) {
    // std::cout << "Calling hydro cpu version!" << std::endl;

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
    constexpr int uf_max = 15;
    double uf_local[uf_max * 8];
    for (int i0 = 1; i0 < HS_NX - 1; i0++) {
        for (int j0 = 1; j0 < HS_NX - 1; j0++) {
            for (int k0 = 1; k0 < HS_NX - 1; k0++) {
                const int iii0 = i0 * HS_DNX + j0 * HS_DNY + k0 * HS_DNZ;
                if (coarse[iii0]) {
                    complete_hydro_amr_boundary_inner_loop<double>(dx, energy_only,
                        unified_ushad.data(), coarse.data(), xmin.data(), i0, j0,
                        k0, opts().n_fields, true, 0, iii0, uf_local);
                    int i = 2 * i0 - H_BW;
                    int j = 2 * j0 - H_BW;
                    int k = 2 * k0 - H_BW;
                    int ir = 0;
                    if (i < 0)
                        ir = 1;
                    for (; ir < 2 && i + ir < H_NX; ir++) {
                        int jr = 0;
                        if (j < 0)
                            jr = 1;
                        for (; jr < 2 && j + jr < H_NX; jr++) {
                            int kr = 0;
                            if (k < 0)
                                kr = 1;
                            for (; kr < 2 && k + kr < H_NX; kr++) {
                                const int iiir = hindex(i + ir, j + jr, k + kr);
                                const int oct_index = ir * 4 + jr * 2 + kr;
                                for (int f = 0; f < opts().n_fields; f++) {
                                    if (!energy_only || f == egas_i) {
                                        U[f][iiir] =
                                            uf_local[f * 8 + oct_index];
                                    }
                                }
                            }
                        }
                    }
                } 
            }
        }
    }
}
//#pragma GCC pop_options
