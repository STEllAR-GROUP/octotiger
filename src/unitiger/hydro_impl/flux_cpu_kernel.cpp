
#include <Vc/Vc>
#include <Vc/common/mask.h>
#include <Vc/vector.h>

#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"

using vc_type = Vc::Vector<double, Vc::VectorAbi::Avx>;
using mask_type = vc_type::mask_type;
using index_type = Vc::Vector<int, Vc::VectorAbi::Avx>;

// helpers for using vectortype specialization functions
template <>
inline void select_wrapper<vc_type, mask_type>(
    vc_type& target, const mask_type cond, const vc_type& tmp1, const vc_type& tmp2) {
    target = tmp2;
    Vc::where(cond, target) = tmp1;
    //std::cout << cond << " -- " << tmp1 << " vs " << tmp2 << " = " << target << std::endl;
   // if (!Vc::all_of(cond) && !Vc::none_of(cond))
   // std::cin.get();
}

template <>
inline vc_type max_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
    return Vc::max(tmp1, tmp2);
}
template <>
inline vc_type min_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
    return Vc::min(tmp1, tmp2);
}
template <>
inline vc_type sqrt_wrapper<vc_type>(const vc_type& tmp1) {
    return Vc::sqrt(tmp1);
}
/// Awful workaround for missing Vc::pow
template <>
inline vc_type pow_wrapper<vc_type>(const vc_type& tmp1, const double& tmp2) {
    // TODO(daissgr) is this accurate enough?
    return Vc::exp(static_cast<vc_type>(tmp2) * Vc::log(tmp1));

    //std::cout << "Pow in: " << tmp1 << " ^ " << tmp2 << std::endl;
    //vc_type ret = 0.0;
    //for (auto vec_i = 0; vec_i < vc_type::size(); vec_i++) {
    //ret[vec_i] = std::pow(tmp1[vec_i], tmp2);
    //}
    // //std::cout << "Pow out: " << ret << std::endl;
    // //std::cout << "Pow exp: " << Vc::exp(static_cast<vc_type>(tmp2)*Vc::log(tmp1)) << std::endl;
    //return ret;
}
template <>
inline vc_type asin_wrapper<vc_type>(const vc_type& tmp1) {
    return Vc::asin(tmp1);
}
template <>
inline bool skippable<mask_type>(const mask_type& tmp1) {
    return Vc::none_of(tmp1);
}

timestep_t flux_cpu_kernel(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F, hydro::x_type& X,
    safe_real omega, const size_t nf_) {
    // input Q, X
    // output F

    timestep_t ts;
    ts.a = 0.0;
    // bunch of tmp containers

    // bunch of small helpers
    static const cell_geometry<3, 8> geo;
    static constexpr auto faces = geo.face_pts();
    static constexpr auto weights = geo.face_weight();
    static constexpr auto xloc = geo.xloc();
    static constexpr auto levi_civita = geo.levi_civita();
    double p, v, v0, c;
    const auto A_ = physics<NDIM>::A_;
    const auto B_ = physics<NDIM>::B_;
    double current_amax = 0.0;
    size_t current_max_index = 0;
    size_t current_d = 0;
    size_t current_dim = 0;

    const double dx = X[0][geo.H_DNX] - X[0][0];

    static thread_local std::vector<vc_type> UR(nf_), UL(nf_), this_flux(nf_);
    static thread_local std::vector<vc_type> FR(nf_), FL(nf_);
    static thread_local std::array<vc_type, NDIM> x;
    static thread_local std::array<vc_type, NDIM> vg;
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
            const vc_type zindices = vc_type::IndexesFromZero();
            //std::cout << "--Face:" << fi << "----------------------------------" << std::endl;
            for (size_t ix = lbs[0]; ix < ubs[0]; ix++) {
                for (size_t iy = lbs[1]; iy < ubs[1]; iy++) {
                    for (size_t iz = lbs[2]; iz < geo.H_NX; iz += vc_type::size()) {
                        const int border = ubs[2] - iz;
                        const mask_type mask = (zindices < border);
                        if (Vc::none_of(mask))
                            continue;
                        const size_t i = ix * geo.H_NX * geo.H_NX + iy * geo.H_NX + iz;
                        vc_type this_ap = 0.0, this_am = 0.0;    // tmps

#pragma unroll
                        for (int f = 0; f < nf_; f++) {
                            UR[f] = vc_type(((Q[f][d]).data()) + i);
                            UL[f] = vc_type(((Q[f][flipped_dim]).data()) + i - geo.H_DN[dim]);
                        }
                        for (int dim = 0; dim < NDIM; dim++) {
                           x[dim] =
                            vc_type(X[dim].data() + i) +  vc_type(0.5 * xloc[d][dim] * dx);
                        }
                        vg[0] = -omega * (vc_type(X[1].data() + i) +  vc_type(0.5 * xloc[d][1] * dx));
                        vg[1] = +omega * (vc_type(X[0].data() + i) +  vc_type(0.5 * xloc[d][0] * dx));
                        vg[2] = 0.0;
                        inner_flux_loop<vc_type>(omega, nf_, A_, B_, UR, UL, FR, FL, this_flux, x,
                            vg, this_ap, this_am, dim, d, i, geo, dx);
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
#pragma unroll
                        for (int f = 0; f < nf_; f++) {
                            // field update from flux
                            Vc::where(!mask, this_flux[f]) = 0.0;
                            const vc_type final_f =
                                vc_type(F[dim][f].data() + i) + weights[fi] * this_flux[f];
                            final_f.store(F[dim][f].data() + i);
                        }
                        //std::cin.get();
                    }    // end z
                }        // end y
            }            // end x
        }                // end dirs
    }                    // end dim
    static thread_local std::vector<double> URs(nf_), ULs(nf_);
    //std::cout << "current amax: " << current_amax << std::endl;
    //std::cin.get();
    ts.a = current_amax;
    ts.x = X[0][current_max_index];
    ts.y = X[1][current_max_index];
    ts.z = X[2][current_max_index];
    const auto flipped_dim = geo.flip_dim(current_d, current_dim);
    for (int f = 0; f < nf_; f++) {
        URs[f] = Q[f][current_d][current_max_index];
        ULs[f] = Q[f][flipped_dim][current_max_index - geo.H_DN[current_dim]];
    }
    ts.ul = ULs;
    ts.ur = URs;
    ts.dim = current_dim;
    return ts;
}
