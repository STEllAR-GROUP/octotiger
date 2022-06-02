//#include <__clang_cuda_builtin_vars.h>
#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
#include <hpx/modules/async_cuda.hpp>

#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp"

// TODO include kokkos simd?
#include "octotiger/common_kernel/kokkos_simd.hpp"

#if defined(OCTOTIGER_HAVE_HIP)
#define cudaLaunchKernel hipLaunchKernel
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyAsync hipMemcpyAsync
#endif

#if defined(OCTOTIGER_HAVE_CUDA)
__global__ void __launch_bounds__(64, 4) reconstruct_cuda_kernel_no_amc(double omega,
#elif defined(OCTOTIGER_HAVE_HIP)
__global__ void reconstruct_cuda_kernel_no_amc(double omega,
#endif
     int nf_,  int angmom_index_,  int* __restrict__ smooth_field_,
     int* __restrict__ disc_detect_, double* __restrict__ combined_q,
     double* __restrict__ combined_x,  double* __restrict__ combined_u, double* __restrict__ AM,
     double *dx,  double* __restrict__ cdiscs,  int n_species_,  int ndir,
     int nangmom) {
    const int q_i = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
        (((q_i % q_inx2) / q_inx) + 2) * inx_large + (((q_i % q_inx2) % q_inx) + 2);
    const int slice_id = blockIdx.x;
    device_simd_mask_t mask(true); // placeholder to make it work with the simd methods
    if (q_i < q_inx3) {
        for (int d = 0; d < ndir; d++) {
            cell_reconstruct_inner_loop_p1_simd<device_simd_t, device_simd_mask_t>(nf_, angmom_index_,
                smooth_field_, disc_detect_, combined_q, combined_u, AM, dx[slice_id], cdiscs, d, i,
                q_i, ndir, nangmom, slice_id, mask);
        }
        // Phase 2
        for (int d = 0; d < ndir; d++) {
            cell_reconstruct_inner_loop_p2_simd<device_simd_t, device_simd_mask_t>(omega, angmom_index_,
                combined_q, combined_x, combined_u, AM, dx[slice_id], d, i, q_i, ndir, nangmom,
                n_species_, nf_, slice_id, mask);
        }
    }
}

#if defined(OCTOTIGER_HAVE_CUDA)
__global__ void __launch_bounds__(64, 4) reconstruct_cuda_kernel(double omega, int nf_,
#elif defined(OCTOTIGER_HAVE_HIP)
__global__ void reconstruct_cuda_kernel(const double omega, const int nf_,
#endif
     int angmom_index_,  int* __restrict__ smooth_field_,  int* __restrict__ disc_detect_,
    double* __restrict__ combined_q,  double* __restrict__ combined_x,
     double* __restrict__ combined_u, double* __restrict__ AM,  double *dx,
     double* __restrict__ cdiscs,  int n_species_,  int ndir,  int nangmom) {
    const int sx_i = angmom_index_;
    const int zx_i = sx_i + NDIM;


    const int q_i = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    const int i = ((q_i / q_inx2) + 2) * inx_large * inx_large +
        (((q_i % q_inx2) / q_inx) + 2) * inx_large + (((q_i % q_inx2) % q_inx) + 2);
    const int slice_id = blockIdx.x;
    const int u_slice_offset = (nf_ * H_N3 + 128) * slice_id;
    const int am_slice_offset = (NDIM * q_inx3 + 128) * slice_id;
    using simd_t = device_simd_t;
    using simd_mask_t = device_simd_mask_t;
    device_simd_mask_t mask(true); // placeholder to make it work with the simd methods
    if (q_i < q_inx3) {
        for (int n = 0; n < nangmom; n++) {
            AM[n * am_offset + q_i + am_slice_offset] =
                combined_u[(zx_i + n) * u_face_offset + i + u_slice_offset] *
                combined_u[i + u_slice_offset];
        }
        for (int d = 0; d < ndir; d++) {
            /* cell_reconstruct_inner_loop_p1_simd<device_simd_t, device_simd_mask_t>(nf_, angmom_index_, */
            /*     smooth_field_, disc_detect_, combined_q, combined_u, AM, dx[slice_id], cdiscs, d, i, */
            /*     q_i, ndir, nangmom, slice_id, mask); */
    const int q_slice_offset = (nf_ * 27 * q_inx3 + 128) * slice_id;
    const int u_slice_offset = (nf_ * H_N3 + 128) * slice_id;
    const int am_slice_offset = (NDIM * q_inx3 + 128) * slice_id;
    const int disc_slice_offset = (ndir / 2 * H_N3 + 128) * slice_id; 

    int l_start;
    int s_start;
    if (angmom_index_ > -1) {
        s_start = angmom_index_;
        l_start = angmom_index_ + NDIM;
    } else {
        s_start = lx_i;
        l_start = lx_i;
    }
    if (angmom_index_ > -1) {
        if (d < ndir / 2) {
            for (int f = 0; f < s_start; f++) {
                cell_reconstruct_ppm_simd<simd_t, simd_mask_t>(combined_q, combined_u,
                    smooth_field_[f + nf_ * slice_id], disc_detect_[f + nf_ * slice_id], cdiscs, d,
                    f, i + u_slice_offset, q_i + q_slice_offset, i + disc_slice_offset, mask);
            }
            for (int f = s_start; f < l_start; f++) {
                cell_reconstruct_ppm_simd<simd_t, simd_mask_t>(combined_q, combined_u, true, false,
                    cdiscs, d, f, i + u_slice_offset, q_i + q_slice_offset, i + disc_slice_offset, mask);
            }
        }
        for (int f = l_start; f < l_start + nangmom; f++) {
            cell_reconstruct_minmod_simd<simd_t, simd_mask_t>(
                combined_q, combined_u, d, f, i + u_slice_offset, q_i + q_slice_offset, mask);
        }
        if (d < ndir / 2) {
            for (int f = l_start + nangmom; f < nf_; f++) {
                cell_reconstruct_ppm_simd<simd_t, simd_mask_t>(combined_q, combined_u,
                    smooth_field_[f + nf_ * slice_id], disc_detect_[f + nf_ * slice_id], cdiscs, d,
                    f, i + u_slice_offset, q_i + q_slice_offset, i + disc_slice_offset, mask);
            }
        }
    } else {
        for (int f = 0; f < nf_; f++) {
            if (f < lx_i || f > lx_i + nangmom) {
                if (d < ndir / 2) {
                  cell_reconstruct_ppm_simd<simd_t, simd_mask_t>(combined_q, combined_u,
                      smooth_field_[f + nf_ * slice_id], disc_detect_[f + nf_ * slice_id], cdiscs,
                      d, f, i + u_slice_offset, q_i + q_slice_offset, i + disc_slice_offset, mask);
                }
            } else {
                cell_reconstruct_minmod_simd<simd_t, simd_mask_t>(
                    combined_q, combined_u, d, f, i + u_slice_offset, q_i + q_slice_offset, mask);
            }
        }
    }

    if (d != ndir / 2 && angmom_index_ > -1) {
        const int start_index_rho = d * q_dir_offset;

        // n m q Levi Civita
        // 0 1 2 -> 1
        simd_t results0 =
            simd_t(AM + q_i + am_slice_offset, SIMD_NAMESPACE::element_aligned_tag{}) -
            vw[d] * 1.0 * 0.5 * xloc[d][1] *
                simd_t(combined_q + (sx_i + 2) * q_face_offset + d *
                    q_dir_offset + q_i +
                        q_slice_offset,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                simd_t(combined_q + start_index_rho + q_i + q_slice_offset,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                dx[slice_id];

        // n m q Levi Civita
        // 0 2 1 -> -1
        results0 = results0 -
            vw[d] * (-1.0) * 0.5 * xloc[d][2] *
                simd_t(combined_q + (sx_i + 1) * q_face_offset + d * q_dir_offset + q_i +
                        q_slice_offset,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                simd_t(combined_q + start_index_rho + q_i + q_slice_offset,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                dx[slice_id];
        // copy results 0 back
        results0 = SIMD_NAMESPACE::choose(mask, results0,
            simd_t(AM + q_i + am_slice_offset, SIMD_NAMESPACE::element_aligned_tag{}));
        results0.copy_to(AM + q_i + am_slice_offset,
            SIMD_NAMESPACE::element_aligned_tag{});

        // n m q Levi Civita
        // 1 0 2 -> -1
        simd_t results1 = simd_t(AM + am_offset + q_i + am_slice_offset,
                              SIMD_NAMESPACE::element_aligned_tag{}) -
            vw[d] * (-1.0) * 0.5 * xloc[d][0] *
                simd_t(combined_q + (sx_i + 2) * q_face_offset + d * q_dir_offset + q_i +
                        q_slice_offset,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                simd_t(combined_q + start_index_rho + q_i + q_slice_offset,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                dx[slice_id];

        // n m q Levi Civita 1 2 0 -> 1
        results1 -= vw[d] * (1.0) * 0.5 * xloc[d][2] *
            simd_t(combined_q + (sx_i + 0) * q_face_offset + d * q_dir_offset + q_i +
                    q_slice_offset,
                SIMD_NAMESPACE::element_aligned_tag{}) *
            simd_t(combined_q + start_index_rho + q_i + q_slice_offset,
                SIMD_NAMESPACE::element_aligned_tag{}) *
            dx[slice_id];
        // copy results 1 back
        results1 = SIMD_NAMESPACE::choose(mask, results1,
            simd_t(AM + am_offset + q_i + am_slice_offset,
                SIMD_NAMESPACE::element_aligned_tag{}));
        results1.copy_to(
            AM + am_offset + q_i + am_slice_offset, SIMD_NAMESPACE::element_aligned_tag{});

        // n m q Levi Civita
        // 2 0 1 -> 1
        simd_t results2 = simd_t(AM + 2 * am_offset + q_i + am_slice_offset,
                              SIMD_NAMESPACE::element_aligned_tag{}) -
            vw[d] * (1.0) * 0.5 * xloc[d][0] *
                simd_t(combined_q + (sx_i + 1) * q_face_offset + d * q_dir_offset + q_i +
                        q_slice_offset,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                simd_t(combined_q + start_index_rho + q_i + q_slice_offset,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                dx[slice_id];

        // n m q Levi Civita 2 1 0 -> -1
        results2 -= vw[d] * (-1.0) * 0.5 * xloc[d][1] *
            simd_t(combined_q + (sx_i + 0) * q_face_offset + d * q_dir_offset + q_i +
                    q_slice_offset,
                SIMD_NAMESPACE::element_aligned_tag{}) *
            simd_t(combined_q + start_index_rho + q_i + q_slice_offset,
                SIMD_NAMESPACE::element_aligned_tag{}) *
            dx[slice_id];
        // copy results 2 back
        results2 = SIMD_NAMESPACE::choose(mask, results2,
            simd_t(AM + 2 * am_offset + q_i + am_slice_offset,
                SIMD_NAMESPACE::element_aligned_tag{}));
        results2.copy_to(AM + 2 * am_offset + q_i + am_slice_offset,
            SIMD_NAMESPACE::element_aligned_tag{});
    }
        }
        // Phase 2
        for (int d = 0; d < ndir; d++) {
            cell_reconstruct_inner_loop_p2_simd<device_simd_t, device_simd_mask_t>(omega, angmom_index_,
                combined_q, combined_x, combined_u, AM, dx[slice_id], d, i, q_i, ndir, nangmom,
                n_species_, nf_, slice_id, mask);
        }
    }
}

#if defined(OCTOTIGER_HAVE_HIP)
void reconstruct_hip_kernel_ggl_wrapper(dim3 const grid_spec, dim3 const threads_per_block,
    double omega, int nf_, int angmom_index_, int *smooth_field_, int* disc_detect, double *combined_q,
    double *combined_x, double *combined_u, double *AM, double *dx, double *cdiscs, int n_species_, int ndir, int nangmom,
    cudaStream_t& stream) {
    if (angmom_index_ > -1) {
				hipLaunchKernelGGL(reconstruct_cuda_kernel, grid_spec, threads_per_block, 0, stream, omega, nf_,
						angmom_index_, smooth_field_, disc_detect, combined_q, combined_x, combined_u, AM, dx, cdiscs,
             n_species_, ndir, nangmom);
    } else {
				hipLaunchKernelGGL(reconstruct_cuda_kernel_no_amc, grid_spec, threads_per_block, 0, stream, omega, nf_,
						angmom_index_, smooth_field_, disc_detect, combined_q, combined_x, combined_u, AM, dx, cdiscs,
             n_species_, ndir, nangmom);
    }
}
#endif

void launch_reconstruct_cuda(
    aggregated_executor_t& executor, double omega,
    int nf_, int angmom_index_, int* smooth_field_, int* disc_detect_, double* combined_q,
    double* combined_x, double* combined_u, double* AM, double *dx, double* cdiscs, int n_species_) {
    static const cell_geometry<NDIM, INX> geo;

    static_assert(device_simd_t::size() == 1, "CUDA/HIP kernels expect scalar SIMD types");
    static_assert(NDIM == 3);
    static_assert(geo.NANGMOM == 3);
    static_assert(geo.NDIR == 27);

    constexpr int blocks = q_inx3 / 64 + 1;
    dim3 const grid_spec(executor.number_slices, 1, blocks);
    dim3 const threads_per_block(1, 8, 8);
    int ndir = geo.NDIR;
    int nangmom = geo.NANGMOM;
    hpx::lcos::future<void> fut;
#if defined(OCTOTIGER_HAVE_CUDA)
    void* args[] = {&omega, &nf_, &angmom_index_, &(smooth_field_), &(disc_detect_), &(combined_q),
        &(combined_x), &(combined_u), &(AM), &(dx), &(cdiscs), &n_species_, &ndir, &nangmom};
    if (angmom_index_ > -1) {
        executor.post(cudaLaunchKernel<decltype(reconstruct_cuda_kernel)>, reconstruct_cuda_kernel,
            grid_spec, threads_per_block, args, 0);
    } else {
        executor.post(cudaLaunchKernel<decltype(reconstruct_cuda_kernel_no_amc)>,
            reconstruct_cuda_kernel_no_amc, grid_spec, threads_per_block, args, 0);
    }
#elif defined(OCTOTIGER_HAVE_HIP)
		executor.post(reconstruct_hip_kernel_ggl_wrapper, grid_spec, threads_per_block, omega, nf_, angmom_index_, smooth_field_,
         disc_detect_, combined_q, combined_x, combined_u, AM, dx, cdiscs, n_species_, ndir, nangmom);
#endif
}

// TODO Launch bounds do not work with larger subgrid size (>8)
__global__ void    //__launch_bounds__(12 * 12, 1)
discs_phase1(double* __restrict__ P, double* __restrict__ combined_u, const double A_,
    const double B_, const double fgamma_, const double de_switch_1, const int nf) {

    const int index = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    const int slice_id = blockIdx.x;

    if (index < inx_normal * inx_normal * inx_normal) {
        const int grid_x = index / (inx_normal * inx_normal);
        const int grid_y = (index % (inx_normal * inx_normal)) / inx_normal;
        const int grid_z = (index % (inx_normal * inx_normal)) % inx_normal;
        cell_find_contact_discs_phase1(
            P, combined_u, A_, B_, fgamma_, de_switch_1, nf, grid_x, grid_y, grid_z, slice_id);
    }
}

#if defined(OCTOTIGER_HAVE_CUDA)
__global__ void __launch_bounds__(64, 4) discs_phase2(
#elif defined(OCTOTIGER_HAVE_HIP)
__global__ void discs_phase2(
#endif
    double* __restrict__ disc, double* __restrict__ P, const double fgamma_, const int ndir) {
    const int index = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    const int slice_id = blockIdx.x;
    if (index < q_inx3) {
        const int grid_x = index / (q_inx * q_inx);
        const int grid_y = (index % (q_inx * q_inx)) / q_inx;
        const int grid_z = (index % (q_inx * q_inx)) % q_inx;
        cell_find_contact_discs_phase2(disc, P, fgamma_, ndir, grid_x, grid_y, grid_z, slice_id);
    }
}

#if defined(OCTOTIGER_HAVE_HIP)
void disc1_hip_kernel_ggl_wrapper(dim3 const grid_spec, dim3 const threads_per_block,
    double* device_P, double* device_u, double A_, double B_, double fgamma_, double de_switch_1, int nf,
    cudaStream_t& stream) {
    hipLaunchKernelGGL(discs_phase1, grid_spec, threads_per_block, 0, stream, device_P, device_u,
        A_, B_, fgamma_, de_switch_1, nf);
}
void disc2_hip_kernel_ggl_wrapper(dim3 const grid_spec, dim3 const threads_per_block,
    double* device_disc, double* device_P, double fgamma_, int ndir, cudaStream_t& stream) {
    hipLaunchKernelGGL(discs_phase2, grid_spec, threads_per_block, 0, stream, device_disc, device_P,
        fgamma_, ndir);
}
#endif

void launch_find_contact_discs_cuda(
    aggregated_executor_t& executor,
    double* device_u, double* device_P, double* device_disc, double A_, double B_, double fgamma_,
    double de_switch_1, int nf) {
    static const cell_geometry<NDIM, INX> geo;
    const int blocks = (inx_normal * inx_normal * inx_normal) / 64 + 1;
    dim3 const grid_spec_phase1(executor.number_slices, 1, blocks);
    dim3 const threads_per_block_phase1(1, 8, 8);
#if defined(OCTOTIGER_HAVE_CUDA)
    void* args_phase1[] = {&(device_P), &(device_u), &A_, &B_, &fgamma_, &de_switch_1, &nf};
    executor.post(cudaLaunchKernel<decltype(discs_phase1)>, discs_phase1, grid_spec_phase1,
        threads_per_block_phase1, args_phase1, 0);
#elif defined(OCTOTIGER_HAVE_HIP)
    executor.post(disc1_hip_kernel_ggl_wrapper, grid_spec_phase1, threads_per_block_phase1,
        device_P, device_u, A_, B_, fgamma_, de_switch_1, nf);
#endif
    int ndir = geo.NDIR;
    const int blocks2 = (q_inx * q_inx * q_inx) / 64 + 1;
    dim3 const grid_spec_phase2(executor.number_slices, 1, blocks2);
    dim3 const threads_per_block_phase2(1, 8, 8);
#if defined(OCTOTIGER_HAVE_CUDA)
    void* args_phase2[] = {&device_disc, &device_P, &fgamma_, &ndir};
    executor.post(cudaLaunchKernel<decltype(discs_phase2)>, discs_phase2, grid_spec_phase2,
        threads_per_block_phase2, args_phase2, 0);
#elif defined(OCTOTIGER_HAVE_HIP)
    executor.post(disc2_hip_kernel_ggl_wrapper, grid_spec_phase2, threads_per_block_phase2,
        device_disc, device_P, fgamma_, ndir);
#endif
}

__global__ void __launch_bounds__(64, 4)
    hydro_pre_recon_cuda(double* __restrict__ device_X, safe_real omega, bool angmom,
        double* __restrict__ device_u, const int nf, const int n_species_) {
    // Index mapping to actual grid
    const int index = (blockIdx.z * 1 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
    const int slice_id = blockIdx.x;
    if (index < inx_large * inx_large * inx_large) {
        const int grid_x = index / (inx_large * inx_large);
        const int grid_y = (index % (inx_large * inx_large)) / inx_large;
        const int grid_z = (index % (inx_large * inx_large)) % inx_large;
        cell_hydro_pre_recon(
            device_X, omega, angmom, device_u, nf, n_species_, grid_x, grid_y, grid_z, slice_id);
    }
}
#if defined(OCTOTIGER_HAVE_HIP)
void pre_recon_hip_kernel_ggl_wrapper(dim3 const grid_spec, dim3 const threads_per_block,
    double* device_X, double omega, bool angmom, double* device_u, int nf, int n_species_,
    cudaStream_t& stream) {
    hipLaunchKernelGGL(hydro_pre_recon_cuda, grid_spec, threads_per_block, 0, stream, device_X,
        omega, angmom, device_u, nf, n_species_);
}
#endif

void launch_hydro_pre_recon_cuda(
    aggregated_executor_t& executor,
    double* device_X, double omega, bool angmom, double* device_u, int nf, int n_species_) {
    const int blocks = (inx_large * inx_large * inx_large) / 64 + 1;
    dim3 const grid_spec(executor.number_slices, 1, blocks);
    dim3 const threads_per_block(1, 8, 8);
#if defined(OCTOTIGER_HAVE_CUDA)
    void* args[] = {&(device_X), &omega, &angmom, &(device_u), &nf, &n_species_};
    executor.post(cudaLaunchKernel<decltype(hydro_pre_recon_cuda)>, hydro_pre_recon_cuda, grid_spec,
        threads_per_block, args, 0);
#elif defined(OCTOTIGER_HAVE_HIP)
    executor.post(pre_recon_hip_kernel_ggl_wrapper, grid_spec, threads_per_block, device_X, omega,
        angmom, device_u, nf, n_species_);
#endif
}

#endif
