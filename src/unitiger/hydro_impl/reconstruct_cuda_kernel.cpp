#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
#include <hpx/modules/async_cuda.hpp>

#include "octotiger/defs.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp"
#include "octotiger/cuda_util/cuda_helper.hpp"

#if defined(OCTOTIGER_HAVE_HIP)
#define cudaLaunchKernel hipLaunchKernel
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyAsync hipMemcpyAsync
#endif

#if defined(OCTOTIGER_HAVE_CUDA)
__global__ void __launch_bounds__(64, 4) reconstruct_cuda_kernel_no_amc(const double omega,
#elif defined(OCTOTIGER_HAVE_HIP)
__global__ void reconstruct_cuda_kernel_no_amc(const double omega,
#endif
    const int nf_, const int angmom_index_, int* __restrict__ smooth_field_,
    int* __restrict__ disc_detect_, double* __restrict__ combined_q,
    double* __restrict__ combined_x, double* __restrict__ combined_u, double* __restrict__ AM,
    const double dx, const double* __restrict__ cdiscs, const int n_species_, const int ndir,
    const int nangmom) {
    const int q_i = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
        (((q_i % q_inx2) / q_inx) + 2) * inx_large + (((q_i % q_inx2) % q_inx) + 2);
    if (q_i < q_inx3) {
        for (int d = 0; d < ndir; d++) {
            cell_reconstruct_inner_loop_p1(nf_, angmom_index_, smooth_field_, disc_detect_,
                combined_q, combined_u, AM, dx, cdiscs, d, i, q_i, ndir, nangmom);
        }
        // Phase 2
        for (int d = 0; d < ndir; d++) {
            cell_reconstruct_inner_loop_p2(omega, angmom_index_, combined_q, combined_x, combined_u,
                AM, dx, d, i, q_i, ndir, nangmom, n_species_);
        }
    }
}

#if defined(OCTOTIGER_HAVE_CUDA)
__global__ void __launch_bounds__(64, 4) reconstruct_cuda_kernel(const double omega, const int nf_,
#elif defined(OCTOTIGER_HAVE_HIP)
__global__ void reconstruct_cuda_kernel(const double omega, const int nf_,
#endif
    const int angmom_index_, int* __restrict__ smooth_field_, int* __restrict__ disc_detect_,
    double* __restrict__ combined_q, double* __restrict__ combined_x,
    double* __restrict__ combined_u, double* __restrict__ AM, const double dx,
    const double* __restrict__ cdiscs, const int n_species_, const int ndir, const int nangmom) {
    const int sx_i = angmom_index_;
    const int zx_i = sx_i + NDIM;

    const int q_i = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
        (((q_i % q_inx2) / q_inx) + 2) * inx_large + (((q_i % q_inx2) % q_inx) + 2);
    if (q_i < q_inx3) {
        for (int n = 0; n < nangmom; n++) {
            AM[n * am_offset + q_i] = combined_u[(zx_i + n) * u_face_offset + i] * combined_u[i];
        }
        for (int d = 0; d < ndir; d++) {
            cell_reconstruct_inner_loop_p1(nf_, angmom_index_, smooth_field_, disc_detect_,
                combined_q, combined_u, AM, dx, cdiscs, d, i, q_i, ndir, nangmom);
        }
        // Phase 2
        for (int d = 0; d < ndir; d++) {
            cell_reconstruct_inner_loop_p2(omega, angmom_index_, combined_q, combined_x, combined_u,
                AM, dx, d, i, q_i, ndir, nangmom, n_species_);
        }
    }
}

void launch_reconstruct_cuda(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor, double omega,
    int nf_, int angmom_index_, int* smooth_field_, int* disc_detect_, double* combined_q,
    double* combined_x, double* combined_u, double* AM, double dx, double* cdiscs, int n_species_) {
    static const cell_geometry<NDIM, INX> geo;

    assert(geo.NDIR == 27);

    constexpr int blocks = q_inx3 / 64 + 1;
    dim3 const grid_spec(1, 1, blocks);
    dim3 const threads_per_block(1, 8, 8);
    int ndir = geo.NDIR;
    int nangmom = geo.NANGMOM;
    void* args[] = {&omega, &nf_, &angmom_index_, &(smooth_field_), &(disc_detect_), &(combined_q),
        &(combined_x), &(combined_u), &(AM), &dx, &(cdiscs), &n_species_, &ndir, &nangmom};
    if (angmom_index_ > -1) {
        const auto reconstruct_cuda_kernel_func_p = reconstruct_cuda_kernel;
        executor.post(cudaLaunchKernel, &reconstruct_cuda_kernel_func_p,
            grid_spec, threads_per_block, args, 0);
    } else {
        const auto reconstruct_cuda_kernel_no_amc_func_p = reconstruct_cuda_kernel_no_amc;
        executor.post(cudaLaunchKernel,
            &reconstruct_cuda_kernel_no_amc_func_p, grid_spec, threads_per_block, args, 0);
    }
}

// TODO Launch bounds do not work with larger subgrid size (>8)
__global__ void    //__launch_bounds__(12 * 12, 1)
discs_phase1(double* __restrict__ P, const double* __restrict__ combined_u, const double A_,
    const double B_, const double fgamma_, const double de_switch_1) {
    const int index = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    printf("hi from disc1");
    if (index < inx_normal * inx_normal * inx_normal) {
        const int grid_x = index / (inx_normal * inx_normal);
        const int grid_y = (index % (inx_normal * inx_normal)) / inx_normal;
        const int grid_z = (index % (inx_normal * inx_normal)) % inx_normal;
        cell_find_contact_discs_phase1(
            P, combined_u, A_, B_, fgamma_, de_switch_1, grid_x, grid_y, grid_z);
    }
}

#if defined(OCTOTIGER_HAVE_CUDA)
__global__ void __launch_bounds__(64, 4) discs_phase2(
#elif defined(OCTOTIGER_HAVE_HIP)
__global__ void discs_phase2(
#endif
    double* __restrict__ disc, const double* __restrict__ P, const double fgamma_, const int ndir) {
    const int index = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    printf("hi from disc2");
    if (index < q_inx3) {
        const int grid_x = index / (q_inx * q_inx);
        const int grid_y = (index % (q_inx * q_inx)) / q_inx;
        const int grid_z = (index % (q_inx * q_inx)) % q_inx;
        cell_find_contact_discs_phase2(disc, P, fgamma_, ndir, grid_x, grid_y, grid_z);
    }
}
__global__ void discs_phase3(void) {
    printf("hi from disc3");
    }


void launch_find_contact_discs_cuda(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    double* device_u, double* device_P, double* device_disc, double A_, double B_, double fgamma_,
    double de_switch_1) {
    static const cell_geometry<NDIM, INX> geo;
    const int blocks = (inx_normal * inx_normal * inx_normal) / 64 + 1;
    dim3 const grid_spec_phase1(1, 1, blocks);
    dim3 const threads_per_block_phase1(1, 8, 8);
    void* args_phase1[] = {&(device_P), &(device_u), &A_, &B_, &fgamma_, &de_switch_1};
    const auto discs_phase1_func_p = discs_phase3;
    hipLaunchKernel(&discs_phase1_func_p, grid_spec_phase1,
        threads_per_block_phase1, args_phase1, 0);
    //executor.post(cudaLaunchKernel, &discs_phase1_func_p, grid_spec_phase1,
    //    threads_per_block_phase1, args_phase1, 0);

    int ndir = geo.NDIR;
    const int blocks2 = (q_inx * q_inx * q_inx) / 64 + 1;
    dim3 const grid_spec_phase2(1, 1, blocks2);
    dim3 const threads_per_block_phase2(1, 8, 8);
    void* args_phase2[] = {&device_disc, &device_P, &fgamma_, &ndir};
    const auto discs_phase2_func_p = discs_phase2;
    executor.post(cudaLaunchKernel, &discs_phase2_func_p, grid_spec_phase2,
        threads_per_block_phase2, args_phase2, 0);
}

__global__ void __launch_bounds__(64, 4)
    hydro_pre_recon_cuda(double* __restrict__ device_X, safe_real omega, bool angmom,
        double* __restrict__ device_u, const int nf, const int n_species_) {
    // Index mapping to actual grid
    const int index = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    if (index < inx_large * inx_large * inx_large) {
        const int grid_x = index / (inx_large * inx_large);
        const int grid_y = (index % (inx_large * inx_large)) / inx_large;
        const int grid_z = (index % (inx_large * inx_large)) % inx_large;
        cell_hydro_pre_recon(
            device_X, omega, angmom, device_u, nf, n_species_, grid_x, grid_y, grid_z);
    }
}

void launch_hydro_pre_recon_cuda(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    double* device_X, double omega, bool angmom, double* device_u, int nf, int n_species_) {
    const int blocks = (inx_large * inx_large * inx_large) / 64 + 1;
    dim3 const grid_spec(1, 1, blocks);
    dim3 const threads_per_block(1, 8, 8);
    void* args[] = {&(device_X), &omega, &angmom, &(device_u), &nf, &n_species_};
    const auto hydro_pre_func_p = hydro_pre_recon_cuda;
    executor.post(cudaLaunchKernel, &hydro_pre_func_p, grid_spec,
        threads_per_block, args, 0);
}

#endif
