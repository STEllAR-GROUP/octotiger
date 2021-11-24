#include <hpx/config/compiler_specific.hpp> 
#ifndef HPX_COMPUTE_DEVICE_CODE

#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"

#include <aligned_buffer_util.hpp>
#include <boost/container/vector.hpp>    // to get non-specialized vector<bool>
#include <buffer_manager.hpp>


boost::container::vector<bool> create_masks() {
    constexpr int length = INX + 2;
    constexpr int length_short = INX + 1;
    boost::container::vector<bool> masks(NDIM * length * length * length);
    constexpr size_t dim_offset = length * length * length;
    const cell_geometry<3, 8> geo;
    for (int dim = 0; dim < NDIM; dim++) {
        std::array<int, NDIM> ubs = {length_short, length_short, length_short};
        for (int dimension = 0; dimension < NDIM; dimension++) {
            ubs[dimension] = geo.xloc()[geo.face_pts()[dim][0]][dimension] == -1 ? (length) : (length_short);
        }
        for (size_t ix = 0; ix < length; ix++) {
            for (size_t iy = 0; iy < length; iy++) {
                for (size_t iz = 0; iz < length; iz++) {
                    const size_t index = ix * length * length + iy * length + iz + dim_offset * dim;
                    if (ix > 0 && iy > 0 && iz > 0 && ix < ubs[0] && iy < ubs[1] && iz < ubs[2])
                        masks[index] = true;
                    else
                        masks[index] = false;
                }
            }
        }
    }
    return masks;
}

#ifdef __x86_64__ // currently only works on x86
#ifdef OCTOTIGER_HAVE_VC
#pragma GCC push_options
#pragma GCC optimize("unroll-loops")

#include <Vc/Vc>
#include <Vc/common/mask.h>
#include <Vc/vector.h>

using vc_type = Vc::Vector<double, Vc::VectorAbi::Avx>;
using mask_type = vc_type::mask_type;
using index_type = Vc::Vector<int, Vc::VectorAbi::Avx>;

// helpers for using vectortype specialization functions
template <>
CUDA_GLOBAL_METHOD inline void select_wrapper<vc_type, mask_type>(
    vc_type& target, const mask_type cond, const vc_type& tmp1, const vc_type& tmp2) {
    target = tmp2;
    Vc::where(cond, target) = tmp1;
}

template <>
CUDA_GLOBAL_METHOD inline vc_type max_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
    return Vc::max(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline vc_type min_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
    return Vc::min(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline vc_type sqrt_wrapper<vc_type>(const vc_type& tmp1) {
    return Vc::sqrt(tmp1);
}
/// Awful workaround for missing Vc::pow
template <>
CUDA_GLOBAL_METHOD inline vc_type pow_wrapper<vc_type>(const vc_type& tmp1, const double& tmp2) {
    // TODO(daissgr) is this accurate enough?
    return Vc::exp(static_cast<vc_type>(tmp2) * Vc::log(tmp1));

    // vc_type ret = 0.0;
    // for (auto vec_i = 0; vec_i < vc_type::size(); vec_i++) {
    // ret[vec_i] = std::pow(tmp1[vec_i], tmp2);
    //}
    // return ret;
}
template <>
CUDA_GLOBAL_METHOD inline vc_type asinh_wrapper<vc_type>(const vc_type& tmp1) {
    // not implemented 
    //return Vc::asinh(tmp1);

    vc_type ret = 0.0;
    for (auto vec_i = 0; vec_i < vc_type::size(); vec_i++) {
      ret[vec_i] = std::asinh(tmp1[vec_i]);
    }
    return ret;
}
template <>
CUDA_GLOBAL_METHOD inline bool skippable<mask_type>(const mask_type& tmp1) {
    return Vc::none_of(tmp1);
}
template <>
CUDA_GLOBAL_METHOD inline vc_type load_value<vc_type>(const double* __restrict__ data, const size_t index) {
    return vc_type(data + index);
}

timestep_t flux_cpu_kernel(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F, hydro::x_type& X,
    safe_real omega, const size_t nf_) {
    // input Q, X
    // output F

    timestep_t ts;
    ts.a = 0.0;

    // bunch of small helpers
    static const cell_geometry<3, 8> geo;
    static constexpr auto faces = geo.face_pts();
    static constexpr auto weights = geo.face_weight();
    static constexpr auto xloc = geo.xloc();
    double p, v, v0, c;
    const auto A_ = physics<NDIM>::A_;
    const auto B_ = physics<NDIM>::B_;
    double current_amax = 0.0;
    size_t current_max_index = 0;
    size_t current_d = 0;
    size_t current_dim = 0;

    const double dx = X[0][geo.H_DNX] - X[0][0];

    std::vector<vc_type> UR(nf_), UL(nf_), this_flux(nf_);
    std::array<vc_type, NDIM> x;
    std::array<vc_type, NDIM> vg;

    for (int dim = 0; dim < NDIM; dim++) {
        const auto indices = geo.get_indexes(3, geo.face_pts()[dim][0]);

        std::array<int, NDIM> lbs = {3, 3, 3};
        std::array<int, NDIM> ubs = {geo.H_NX - 3, geo.H_NX - 3, geo.H_NX - 3};
        for (int dimension = 0; dimension < NDIM; dimension++) {
            ubs[dimension] = geo.xloc()[geo.face_pts()[dim][0]][dimension] == -1 ?
                (geo.H_NX - 3 + 1) :
                (geo.H_NX - 3);
            lbs[dimension] = geo.xloc()[geo.face_pts()[dim][0]][dimension] == +1 ? (3 - 1) : 3;
        }

        // zero-initialize F
        for (int f = 0; f < nf_; f++) {
#pragma ivdep
            for (const auto& i : indices) {
                F[dim][f][i] = 0.0;
            }
        }

        for (int fi = 0; fi < geo.NFACEDIR; fi++) {    // 9
            vc_type ap = 0.0, am = 0.0;                // final am ap for this i
            safe_real this_amax = 0.0;
            const auto d = faces[dim][fi];
            const auto flipped_dim = geo.flip_dim(d, dim);
            // std::cout << dim << " flipped: " << flipped_dim << " d: " << d << std::endl;
            const vc_type zindices = vc_type::IndexesFromZero();
            for (size_t ix = lbs[0]; ix < ubs[0]; ix++) {
                for (size_t iy = lbs[1]; iy < ubs[1]; iy++) {
                    for (size_t iz = lbs[2]; iz < geo.H_NX; iz += vc_type::size()) {
                        const int border = ubs[2] - iz;
                        const mask_type mask = (zindices < border);
                        if (Vc::none_of(mask))
                            continue;
                        const size_t i = ix * geo.H_NX * geo.H_NX + iy * geo.H_NX + iz;
                        vc_type this_ap = 0.0, this_am = 0.0;    // tmps

                        for (int f = 0; f < nf_; f++) {
                            UR[f] = vc_type(((Q[f][d]).data()) + i);
                            UL[f] = vc_type(((Q[f][flipped_dim]).data()) + i - geo.H_DN[dim]);
                            Vc::where(!mask, UR[f]) = 1.0;
                            Vc::where(!mask, UL[f]) = 1.0;
                        }
                        for (int dim = 0; dim < NDIM; dim++) {
                            x[dim] = vc_type(X[dim].data() + i) + vc_type(0.5 * xloc[d][dim] * dx);
                        }
                        vg[0] =
                            -omega * (vc_type(X[1].data() + i) + vc_type(0.5 * xloc[d][1] * dx));
                        vg[1] =
                            +omega * (vc_type(X[0].data() + i) + vc_type(0.5 * xloc[d][0] * dx));
                        vg[2] = 0.0;
                        inner_flux_loop<vc_type>(omega, nf_, A_, B_, UR.data(), UL.data(),
                            this_flux.data(), x.data(), vg.data(), this_ap, this_am, dim, d, dx,
                            physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1);
                        Vc::where(!mask, this_ap) = 0.0;
                        Vc::where(!mask, this_am) = 0.0;
                        am = min_wrapper(am, this_am);
                        ap = max_wrapper(ap, this_ap);
                        vc_type tmp_amax = max_wrapper(ap, (-am));

                        for (auto vec_i = 0; vec_i < vc_type::size(); vec_i++) {
                            if (tmp_amax[vec_i] > current_amax) {
                                current_amax = tmp_amax[vec_i];
                                current_max_index = i + vec_i;
                                current_d = d;
                                current_dim = dim;
                            }
                        }
                        for (int f = 0; f < nf_; f++) {
                            // mask not required
                            // Vc::where(!mask, this_flux[f]) = 0.0;
                            // field update from flux
                            const vc_type final_f =
                                vc_type(F[dim][f].data() + i) + weights[fi] * this_flux[f];
                            final_f.store(F[dim][f].data() + i);
                        }
                    }    // end z
                }        // end y
            }            // end x
        }                // end dirs
    }                    // end dim
    static thread_local std::vector<double> URs(nf_), ULs(nf_);
    // std::cout << "current amax: " << current_amax << std::endl;
    // std::cin.get();
    ts.a = current_amax;
    ts.x = X[0][current_max_index];
    ts.y = X[1][current_max_index];
    ts.z = X[2][current_max_index];
    const auto flipped_dim = geo.flip_dim(current_d, current_dim);
    for (int f = 0; f < nf_; f++) {
        URs[f] = Q[f][current_d][current_max_index];
        ULs[f] = Q[f][flipped_dim][current_max_index - geo.H_DN[current_dim]];
    }
    ts.ul = URs;
    ts.ur = ULs;
    ts.dim = current_dim;
    return ts;
}

/*timestep_t flux_unified_cpu_kernel(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F,
    hydro::x_type& X, safe_real omega, const size_t nf_) {
    // input Q, X
    // output F
    // SAGIV: This is the last function that contains explicitly the number 15 as the number of fields (10 physical fields in 3D + 5 specie fields).
    // I did not changed this function because it is not being called anyway. Maybe we can remove this function.

    timestep_t ts;
    ts.a = 0.0;
    // Convert
    std::vector<double, recycler::aggressive_recycle_std<double>> combined_q(
        15 * 27 * 10 * 10 * 10 + 32);
    auto it = combined_q.begin();
    for (auto face = 0; face < 15; face++) {
        for (auto d = 0; d < 27; d++) {
            auto start_offset = 2 * 14 * 14 + 2 * 14 + 2;
            for (auto ix = 2; ix < 2 + INX + 2; ix++) {
                for (auto iy = 2; iy < 2 + INX + 2; iy++) {
                    it = std::copy(Q[face][d].begin() + start_offset,
                        Q[face][d].begin() + start_offset + 10, it);
                    start_offset += 14;
                }
                start_offset += (2 + 2) * 14;
            }
        }
    }
    std::vector<double, recycler::aggressive_recycle_std<double>> combined_x(
        NDIM * 1000 + 32);
    auto it_x = combined_x.begin();
    for (size_t dim = 0; dim < NDIM; dim++) {
        auto start_offset = 2 * 14 * 14 + 2 * 14 + 2;
        for (auto ix = 2; ix < 2 + INX + 2; ix++) {
            for (auto iy = 2; iy < 2 + INX + 2; iy++) {
                it_x = std::copy(
                    X[dim].begin() + start_offset, X[dim].begin() + start_offset + 10, it_x);
                start_offset += 14;
            }
            start_offset += (2 + 2) * 14;
        }
    }

    std::vector<double, recycler::aggressive_recycle_std<double>> combined_f(
        NDIM * 15 * 1000 + 32);
    // bunch of tmp containers

    // bunch of small helpers
    static const cell_geometry<3, 8> geo;
    static constexpr auto faces = geo.face_pts();
    static constexpr auto weights = geo.face_weight();
    static constexpr auto xloc = geo.xloc();
    double p, v, v0, c;
    const auto A_ = physics<NDIM>::A_;
    const auto B_ = physics<NDIM>::B_;
    double current_amax = 0.0;
    size_t current_max_index = 0;
    size_t current_d = 0;
    size_t current_dim = 0;

    const double dx = X[0][geo.H_DNX] - X[0][0];

    std::vector<vc_type> this_flux(nf_);
    std::array<vc_type, NDIM> x;
    std::array<vc_type, NDIM> vg;

    // TODO(daissgr) why is this only working with static?
    static const auto masks_container = create_masks();
    static const bool* masks = masks_container.data();

    constexpr size_t dim_offset = 1000;
    constexpr size_t face_offset = 27 * 1000;
    constexpr int compressedH_DN[3] = {100, 10, 1};
    for (int dim = 0; dim < NDIM; dim++) {
        // zero-initialize F
        for (int f = 0; f < nf_; f++) {
            auto it = combined_f.begin() + dim * 15 * 1000 + f * 1000 + 111;
            std::fill(it, it + 889, 0.0);
        }
        //auto it = combined_f.begin() + dim * 15 * 1000;
        //std::fill(it, it + 15000, 0.0);

        for (int fi = 0; fi < geo.NFACEDIR; fi++) {    // 9
            vc_type ap = 0.0, am = 0.0;
            safe_real this_amax = 0.0;
            const auto d = faces[dim][fi];
            const auto flipped_dim = geo.flip_dim(d, dim);

            for (size_t index = 111; index < 10 * 100; index += vc_type::size()) {
                const vc_type::mask_type mask(masks + index + dim * dim_offset);

                if (Vc::none_of(mask))
                    continue;
                vc_type this_ap = 0.0, this_am = 0.0;    // tmps

                for (int dim = 0; dim < NDIM; dim++) {
                    x[dim] = vc_type(combined_x.data() + dim * 1000 + index) +
                        vc_type(0.5 * xloc[d][dim] * dx);
                }
                vg[0] = -omega *
                    (vc_type(combined_x.data() + 1000 + index) + vc_type(0.5 * xloc[d][1] * dx));
                vg[1] =
                    +omega * (vc_type(combined_x.data() + index) + vc_type(0.5 * xloc[d][0] * dx));
                vg[2] = 0.0;
                inner_flux_loop2<vc_type>(omega, nf_, A_, B_, combined_q.data(), this_flux.data(),
                    x.data(), vg.data(), this_ap, this_am, dim, d, dx, physics<NDIM>::fgamma_,
                    physics<NDIM>::de_switch_1, dim_offset * d + index,
                    dim_offset * flipped_dim + index - compressedH_DN[dim], face_offset);

                Vc::where(!mask, this_ap) = 0.0;
                Vc::where(!mask, this_am) = 0.0;
                am = min_wrapper(am, this_am);
                ap = max_wrapper(ap, this_ap);
                vc_type tmp_amax = max_wrapper(ap, (-am));

                for (auto vec_i = 0; vec_i < vc_type::size(); vec_i++) {
                    if (tmp_amax[vec_i] > current_amax) {
                        current_amax = tmp_amax[vec_i];
                        current_max_index = index + vec_i;
                        current_d = d;
                        current_dim = dim;
                    }
                }
                for (int f = 0; f < nf_; f++) {
                    //Vc::where(!mask, this_flux[f]) = 0.0;
                    const vc_type final_f =
                        vc_type(combined_f.data() + dim * 15 * 1000 + f * 1000 + index) +
                        weights[fi] * this_flux[f];
                    final_f.store(combined_f.data() + dim * 15 * 1000 + f * 1000 + index);
                }
            }
        }    // end dirs
    }        // end dim

    // convert f
    for (size_t dim = 0; dim < NDIM; dim++) {
        for (auto face = 0; face < 15; face++) {
            auto face_offse_f = dim * 15 * 1000 + face * 1000;
            auto start_offset = 2 * 14 * 14 + 2 * 14 + 2;
            auto compressed_offset = 0;
            for (auto ix = 2; ix < 2 + INX + 2; ix++) {
                for (auto iy = 2; iy < 2 + INX + 2; iy++) {
                    std::copy(combined_f.begin() + face_offse_f + compressed_offset,
                        combined_f.begin() + face_offse_f + compressed_offset + 10,
                        F[dim][face].data() + start_offset);
                    compressed_offset += 10;
                    start_offset += 14;
                }
                start_offset += (2 + 2) * 14;
            }
        }
    }
    std::vector<double> URs(nf_), ULs(nf_);
    ts.a = current_amax;
    ts.x = combined_x[current_max_index];
    ts.y = combined_x[current_max_index + 1000];
    ts.z = combined_x[current_max_index + 2000];
    const auto flipped_dim = geo.flip_dim(current_d, current_dim);
    for (int f = 0; f < nf_; f++) {
        URs[f] = combined_q[current_max_index + f * face_offset + dim_offset * current_d];
        ULs[f] = combined_q[current_max_index - compressedH_DN[current_dim] + f * face_offset +
            dim_offset * flipped_dim];
    }
    ts.ul = std::move(ULs);
    ts.ur = std::move(URs);
    ts.dim = current_dim;
    return ts;
}*/
#pragma GCC pop_options
#endif
#endif
#endif
