#pragma GCC push_options
#pragma GCC optimize("unroll-loops")

#include <iostream>
#include <vector>

#include "octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp"
#include "octotiger/util/vec_vc_wrapper.hpp"

void complete_hydro_amr_boundary_vc(const double dx, const bool energy_only,
    const std::vector<std::vector<real>>& Ushad, const std::vector<std::atomic<int>>& is_coarse,
    const std::array<double, NDIM>& xmin, std::vector<std::vector<double>>& U) {
    // std::cout << "Calling hydro cpu version!" << std::endl;

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

    constexpr int uf_max = 15;
    // TODO Use once mapping is fixed
    //vc_type uf_local[uf_max * 8];
    const vc_type zindices = vc_type::IndexesFromZero();
    for (int i0 = 1; i0 < HS_NX - 1; i0++) {
        for (int j0 = 1; j0 < HS_NX - 1; j0++) {
            bool mask_coarse[vc_type::size()];
            for (int k0 = 1; k0 < HS_NX - 1; k0 += vc_type::size()) {
                const int iii0 = i0 * HS_DNX + j0 * HS_DNY + k0 * HS_DNZ;
                for (int mi = 0; mi < vc_type::size(); mi++)
                    mask_coarse[mi] = coarse[mi + iii0];
                const int border = HS_NX - 1 - k0;
                const mask_type mask1 = (zindices < border);
                const mask_type mask2(mask_coarse);
                const mask_type mask = mask1 && mask2;
                if (Vc::none_of(mask))
                    continue;
                complete_hydro_amr_boundary_inner_loop<vc_type>(dx, energy_only,
                    unified_ushad.data(), coarse.data(), xmin.data(), unified_uf.data(), i0, j0, k0,
                    opts().n_fields, mask, zindices, iii0);

                // TODO fix mapping
                /*for (int mi = 0; mi < vc_type::size(); mi++) {
                if (coarse[iii0 + mi] && k0 + mi < HS_NX -1) {
                    int i = 2 * i0 - H_BW ;
                    int j = 2 * j0 - H_BW ;
                    int k = 2 * (k0 + mi) - H_BW ;
                    int ir = 0, jr = 0, kr = 0;
                    if (i < 0) i = 0;
                    if (j < 0) j = 0;
                    if (k < 0) k = 0;
                    for (int ir = 0; ir < 2; ir++) {
                        for (int jr = 0; jr < 2; jr++) {
                            for (int kr = 0; kr < 2; kr++) {
                                const int iiir = hindex(i + ir, j + jr, k + kr);
                                const int oct_index = ir * 4 + jr * 2 + kr;
                                for (int f = 0; f < opts().n_fields; f++) {
                                    if (!energy_only || f == egas_i)
                                        U[f][iiir] =
                                            unified_uf[f * field_offset + iii0 + mi + oct_index * HS_N3];
                                       // U[f][iiir] =
                                        //    uf_local[f * 8 + oct_index][mi];
                                }
                            }
                        }
                    }
                }
                }*/
            }
        }
    }

    // TODO Remove this once mapping is fixed
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
                            U[f][iiir] = unified_uf[f * field_offset + iii0 + oct_index * HS_N3];
                        }
                    }
                }
            }
            // std::copy(unified_u.begin() + f * H_N3, unified_u.begin() + f * H_N3 + H_N3,
            // U[f].begin());
        }
    }
}

#pragma GCC pop_options
