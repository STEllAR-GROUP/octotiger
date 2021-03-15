//#pragma GCC push_options
//#pragma GCC optimize("unroll-loops")

#include <iostream>
#include <vector>

#include "octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp"
#include "octotiger/util/vec_scalar_host_wrapper.hpp"

#ifdef OCTOTIGER_HAVE_CUDA
__host__ void launch_complete_hydro_amr_boundary_cuda(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor, double dx, bool energy_only, const std::vector<std::vector<real>> &Ushad, const std::vector<std::atomic<int>> &is_coarse, const std::array<double, NDIM> &xmin, std::vector<std::vector<real>> &U) {

    // Create host buffers
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> unified_uf(
        opts().n_fields * HS_N3 * 8);
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> unified_ushad(
        opts().n_fields * HS_N3);
    std::vector<int, recycler::recycle_allocator_cuda_host<int>> coarse(HS_N3);
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> x_min(HS_N3);

    // Create device buffers
    recycler::cuda_device_buffer<double> device_uf(opts().n_fields * HS_N3 * 8);
    recycler::cuda_device_buffer<double> device_ushad(opts().n_fields * HS_N3);
    recycler::cuda_device_buffer<double> device_coarse(HS_N3);
    recycler::cuda_device_buffer<double> device_xmin(NDIM);

    for (int d = 0; d < NDIM; d++) {
      x_min[d] = xmin[d];
    }
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
    cudaMemcpyAsync, device_xmin.device_side_buffer,
    x_min.data(), (NDIM) * sizeof(double), cudaMemcpyHostToDevice);

    // Fill host buffers
    for (int f = 0; f < opts().n_fields; f++) {
        if (!energy_only || f == egas_i) {
            std::copy(
                Ushad[f].begin(), Ushad[f].begin() + HS_N3, unified_ushad.begin() + f * HS_N3);
        }
    }
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
    cudaMemcpyAsync, device_ushad.device_side_buffer,
    unified_ushad.data(), (opts().n_fields * HS_N3) * sizeof(double), cudaMemcpyHostToDevice);

    //
    for (int i = 0; i < HS_N3; i++) {
        coarse[i] = is_coarse[i];
   }
   hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
   cudaMemcpyAsync, device_coarse.device_side_buffer,
   coarse.data(), (HS_N3) * sizeof(int), cudaMemcpyHostToDevice);


    dim3 const grid_spec(1, 1, HS_NX - 2);
    dim3 const threads_per_block(1,  HS_NX - 2, HS_NX - 2);
    int nfields = opts().n_fields;
    void* args[] = {&dx, &energy_only, &(device_ushad.device_side_buffer), &(device_coarse.device_side_buffer),
      &(device_xmin.device_side_buffer), &(device_uf.device_side_buffer), &nfields};

   /*executor.post(
   cudaLaunchKernel<decltype(complete_hydro_amr_cuda_kernel)>,
   complete_hydro_amr_cuda_kernel, grid_spec, threads_per_block, args, 0);*/
    launch_complete_hydro_amr_boundary_cuda_post(executor, grid_spec, threads_per_block, args);

    auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               cudaMemcpyAsync, unified_uf.data(), device_uf.device_side_buffer,
               (opts().n_fields * HS_N3 * 8) * sizeof(double), cudaMemcpyDeviceToHost);
    fut.get();

    constexpr int field_offset = HS_N3 * 8;
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
#endif

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
