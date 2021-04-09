#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/defs.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp"

// __device__ inline void reconstruct_minmod_cuda(double* __restrict__ combined_q,
//     const double* __restrict__ combined_u_face, int d, int f, int i, int q_i) {
//     const auto di = dir[d];
//     const int start_index = f * q_face_offset + d * q_dir_offset;
//     combined_q[q_i + start_index] = combined_u_face[i] +
//         0.5 *
//             minmod_cuda(combined_u_face[i + di] - combined_u_face[i],
//                 combined_u_face[i] - combined_u_face[i - di]);
// }

__global__ void __launch_bounds__(64, 4) reconstruct_cuda_kernel_no_amc(const double omega,
    const int nf_, const int angmom_index_, int* __restrict__ smooth_field_,
    int* __restrict__ disc_detect_, double* __restrict__ combined_q,
    double* __restrict__ combined_x, double* __restrict__ combined_u, double* __restrict__ AM,
    const double dx, const double* __restrict__ cdiscs, const int n_species_, const int ndir,
    const int nangmom) {
    const int q_i = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    const int i =
        ((q_i / 100) + 2) * 14 * 14 + (((q_i % 100) / 10) + 2) * 14 + (((q_i % 100) % 10) + 2);
    if (q_i < 1000) {
        for (int d = 0; d < ndir; d++) {
            reconstruct_inner_loop_p1(nf_, angmom_index_, smooth_field_, disc_detect_, combined_q,
                combined_u, AM, dx, cdiscs, d, i, q_i, ndir, nangmom);
        }
        // Phase 2
        for (int d = 0; d < ndir; d++) {
            reconstruct_inner_loop_p2(omega, angmom_index_, combined_q, combined_x, combined_u, AM,
                dx, d, i, q_i, ndir, nangmom, n_species_);
        }
    }
}

__global__ void __launch_bounds__(64, 4) reconstruct_cuda_kernel(const double omega, const int nf_,
    const int angmom_index_, int* __restrict__ smooth_field_, int* __restrict__ disc_detect_,
    double* __restrict__ combined_q, double* __restrict__ combined_x,
    double* __restrict__ combined_u, double* __restrict__ AM, const double dx,
    const double* __restrict__ cdiscs, const int n_species_, const int ndir, const int nangmom) {
    const int sx_i = angmom_index_;
    const int zx_i = sx_i + NDIM;

    const int q_i = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    const int i =
        ((q_i / 100) + 2) * 14 * 14 + (((q_i % 100) / 10) + 2) * 14 + (((q_i % 100) % 10) + 2);
    if (q_i < 1000) {
        for (int n = 0; n < nangmom; n++) {
            AM[n * am_offset + q_i] = combined_u[(zx_i + n) * u_face_offset + i] * combined_u[i];
        }
        for (int d = 0; d < ndir; d++) {
            reconstruct_inner_loop_p1(nf_, angmom_index_, smooth_field_, disc_detect_, combined_q,
                combined_u, AM, dx, cdiscs, d, i, q_i, ndir, nangmom);
        }
        // Phase 2
        for (int d = 0; d < ndir; d++) {
            reconstruct_inner_loop_p2(omega, angmom_index_, combined_q, combined_x, combined_u, AM,
                dx, d, i, q_i, ndir, nangmom, n_species_);
        }
    }
}

void launch_reconstruct_cuda(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor, double omega,
    int nf_, int angmom_index_, int* smooth_field_, int* disc_detect_, double* combined_q,
    double* combined_x, double* combined_u, double* AM, double dx, double* cdiscs, int n_species_) {
    static const cell_geometry<NDIM, INX> geo;

    // Current implementation limitations of this kernel - can be resolved but that takes more work
    //    assert(angmom_index_ > -1);
    assert(NDIM > 2);
    //    assert(nf_ == 15); // is not required anymore
    assert(geo.NDIR == 27);
    assert(INX == 8);

    dim3 const grid_spec(1, 1, 16);
    dim3 const threads_per_block(1, 8, 8);
    int ndir = geo.NDIR;
    int nangmom = geo.NANGMOM;
    void* args[] = {&omega, &nf_, &angmom_index_, &(smooth_field_), &(disc_detect_), &(combined_q),
        &(combined_x), &(combined_u), &(AM), &dx, &(cdiscs), &n_species_, &ndir, &nangmom};
    if (angmom_index_ > -1) {
        executor.post(cudaLaunchKernel<decltype(reconstruct_cuda_kernel)>, reconstruct_cuda_kernel,
            grid_spec, threads_per_block, args, 0);
    } else {
        executor.post(cudaLaunchKernel<decltype(reconstruct_cuda_kernel_no_amc)>,
            reconstruct_cuda_kernel_no_amc, grid_spec, threads_per_block, args, 0);
    }
}

__global__ void __launch_bounds__(12 * 12, 1)
    discs_phase1(double* __restrict__ P, const double* __restrict__ combined_u, const double A_,
        const double B_, const double fgamma_, const double de_switch_1) {
    cell_find_contact_discs_phase1(P, combined_u, A_, B_, fgamma_, de_switch_1, blockIdx.z, threadIdx.y, threadIdx.z);
}

__global__ void __launch_bounds__(10 * 10, 1) discs_phase2(
    double* __restrict__ disc, const double* __restrict__ P, const double fgamma_, const int ndir) {
    cell_find_contact_discs_phase2(disc, P, fgamma_, ndir, blockIdx.z, threadIdx.y, threadIdx.z);
}

void launch_find_contact_discs_cuda(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    double* device_u, double* device_P, double* device_disc, double A_, double B_, double fgamma_,
    double de_switch_1) {
    static const cell_geometry<NDIM, INX> geo;
    dim3 const grid_spec_phase1(1, 1, 12);
    dim3 const threads_per_block_phase1(1, 12, 12);
    void* args_phase1[] = {&(device_P), &(device_u), &A_, &B_, &fgamma_, &de_switch_1};
    executor.post(cudaLaunchKernel<decltype(discs_phase1)>, discs_phase1, grid_spec_phase1,
        threads_per_block_phase1, args_phase1, 0);

    int ndir = geo.NDIR;
    dim3 const grid_spec_phase2(1, 1, 10);
    dim3 const threads_per_block_phase2(1, 10, 10);
    void* args_phase2[] = {&device_disc, &device_P, &fgamma_, &ndir};
    executor.post(cudaLaunchKernel<decltype(discs_phase2)>, discs_phase2, grid_spec_phase2,
        threads_per_block_phase2, args_phase2, 0);
}

__global__ void __launch_bounds__(14 * 14, 1)
    hydro_pre_recon_cuda(double* __restrict__ device_X, safe_real omega, bool angmom,
        double* __restrict__ device_u, const int nf, const int n_species_) {
    cell_hydro_pre_recon(device_X, omega, angmom, device_u, nf, n_species_,
     blockIdx.z, threadIdx.y, threadIdx.z);
}

void launch_hydro_pre_recon_cuda(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    double* device_X, double omega, bool angmom, double* device_u, int nf, int n_species_) {
    dim3 const grid_spec(1, 1, 14);
    dim3 const threads_per_block(1, 14, 14);
    void* args[] = {&(device_X), &omega, &angmom, &(device_u), &nf, &n_species_};
    executor.post(cudaLaunchKernel<decltype(hydro_pre_recon_cuda)>, hydro_pre_recon_cuda, grid_spec,
        threads_per_block, args, 0);
}

#endif
