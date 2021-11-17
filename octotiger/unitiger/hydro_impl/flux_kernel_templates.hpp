#pragma once
#if defined(__clang__)
constexpr int faces[3][9] = {{12, 0, 3, 6, 9, 15, 18, 21, 24}, {10, 0, 1, 2, 9, 11, 18, 19, 20},
    {4, 0, 1, 2, 3, 5, 6, 7, 8}};

constexpr double quad_weights[9] = {
    16. / 36., 1. / 36., 4. / 36., 1. / 36., 4. / 36., 4. / 36., 1. / 36., 4. / 36., 1. / 36.};

constexpr int offset = 0;
// TODO Change hard coded offsets
constexpr int dimension_length = INX + 2;
constexpr int compressedH_DN[3] = {dimension_length * dimension_length, dimension_length, 1};
constexpr int dim_offset = dimension_length * dimension_length * dimension_length;
constexpr int face_offset = 27 * dim_offset;
#else
CUDA_CALLABLE_METHOD const int faces[3][9] = {{12, 0, 3, 6, 9, 15, 18, 21, 24},
    {10, 0, 1, 2, 9, 11, 18, 19, 20}, {4, 0, 1, 2, 3, 5, 6, 7, 8}};

CUDA_CALLABLE_METHOD const double quad_weights[9] = {
    16. / 36., 1. / 36., 4. / 36., 1. / 36., 4. / 36., 4. / 36., 1. / 36., 4. / 36., 1. / 36.};

CUDA_CALLABLE_METHOD const int offset = 0;
// TODO Change hard coded offsets
constexpr int dimension_length = INX + 2;
CUDA_CALLABLE_METHOD const int compressedH_DN[3] = {dimension_length * dimension_length, dimension_length, 1};
CUDA_CALLABLE_METHOD const int dim_offset = dimension_length * dimension_length * dimension_length;
CUDA_CALLABLE_METHOD const int face_offset = 27 * dim_offset;
#endif

CUDA_GLOBAL_METHOD inline int flip_dim(const int d, const int flip_dim) {
    int dims[3];
    int k = d;
    for (int dim = 0; dim < 3; dim++) {
        dims[dim] = k % 3;
        k /= 3;
    }
    k = 0;
    dims[flip_dim] = 2 - dims[flip_dim];
    for (int dim = 0; dim < 3; dim++) {
        k *= 3;
        k += dims[2 - dim];
    }
    return k;
}

template <>
CUDA_GLOBAL_METHOD inline void select_wrapper<double, bool>(
    double& target, const bool cond, const double& tmp1, const double& tmp2) {
    target = cond ? tmp1 : tmp2;
}
template <>
CUDA_GLOBAL_METHOD inline double max_wrapper<double>(const double& tmp1, const double& tmp2) {
    return std::max(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline double min_wrapper<double>(const double& tmp1, const double& tmp2) {
    return std::min(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline double sqrt_wrapper<double>(const double& tmp1) {
    return std::sqrt(tmp1);
}
template <>
CUDA_GLOBAL_METHOD inline double pow_wrapper<double>(const double& tmp1, const double& tmp2) {
    return std::pow(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline double asinh_wrapper<double>(const double& tmp1) {
    return std::asinh(tmp1);
}
template <>
CUDA_GLOBAL_METHOD inline bool skippable<bool>(const bool& tmp1) {
    return !tmp1;
}
template <>
CUDA_GLOBAL_METHOD inline double load_value<double>(
    const double* data, const size_t index) {
    return data[index];
}
template <>
CUDA_GLOBAL_METHOD inline void store_value<double>(
    double* data, const size_t index, const double& value) {
    data[index] = value;
}

template <typename double_t, typename container_t, typename const_container_t>
CUDA_GLOBAL_METHOD inline double_t cell_inner_flux_loop(const double omega, const size_t nf_,
    const double A_, const double B_, const_container_t& U,
    container_t& this_flux, const double_t* __restrict__ x,
    const double_t* __restrict__ vg, double_t& ap, double_t& am, const size_t dim, const size_t d,
    const double dx, const double fgamma, const double de_switch_1, const size_t index,
    const size_t flipped_index, const size_t face_offset) {
    double_t amr, apr, aml, apl;
    double_t this_ap, this_am;    // tmps

    //auto rho = load_value<double_t>(U, rho_i * face_offset + index);
    auto rho = U[rho_i * face_offset + index];
    // if(index == 430) {
    //     printf("Rho %f ", rho);
    // }
    auto rhoinv = (1.) / rho;
    double_t pdeg = static_cast<double_t>(0.0), edeg = static_cast<double_t>(0.0),
     dpdeg_drho = static_cast<double_t>(0.0);

    // all workitems choose the same path
    if (A_ != 0.0) {
        const auto Binv = 1.0 / B_;
        const auto x = pow_wrapper(rho * Binv, 1.0 / 3.0);
        const auto x_sqr = x * x;
        const auto x_sqr_sqrt = sqrt_wrapper(x_sqr + 1.0);
        const auto x_pow_5 = x_sqr * x_sqr * x;
        const double_t hdeg = 8.0 * A_ * Binv * (x_sqr_sqrt - 1.0);

        const double_t edeg_tmp1 = rho * hdeg - pdeg;
        const double_t edeg_tmp2 = 2.4 * A_ * x_pow_5;
        const double_t pdeg_tmp1 = A_ * (x * (2 * x_sqr - 3) * x_sqr_sqrt + 3 * asinh_wrapper(x));
        const double_t pdeg_tmp2 = 1.6 * A_ * x_pow_5;
        select_wrapper(edeg, (x > 0.001), edeg_tmp1, edeg_tmp2);
        select_wrapper(pdeg, (x > 0.001), pdeg_tmp1, pdeg_tmp2);

        dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
    }
    double_t ek = 0.0;
    double_t ein;
    for (int dim = 0; dim < NDIM; dim++) {
        ek += U[(sx_i + dim) * face_offset + index] *
            U[(sx_i + dim) * face_offset + index] * rhoinv * 0.5;
    // if(index == 430) {
    //     printf("sx %f ", U[(sx_i + dim) * face_offset + index]);
    // }
    }
    const auto ein1_tmp2 = U[egas_i * face_offset + index] - ek - edeg;
    // if(index == 430) {
    //     printf("egas %f ", U[egas_i * face_offset + index]);
    // }
    const auto ein1_mask =
        (ein1_tmp2 < (de_switch_1 * U[egas_i * face_offset + index]));
    if (!skippable(ein1_mask)) {
        const auto ein1_tmp1 =
            pow_wrapper(U[tau_i * face_offset + index], fgamma);
        select_wrapper(ein, ein1_mask, ein1_tmp1, ein1_tmp2);
    } else {
        ein = ein1_tmp2;
    }
    const auto dp_drho = dpdeg_drho + (fgamma - 1.0) * ein * rhoinv;
    const auto dp_deps = (fgamma - 1.0) * rho;
    const auto v0 = U[(sx_i + dim) * face_offset + index] * rhoinv;
    const auto p = (fgamma - 1.0) * ein + pdeg;
    const auto c = sqrt_wrapper(p * rhoinv * rhoinv * dp_deps + dp_drho);
    const auto v = v0 - vg[dim];
    amr = v - c;
    apr = v + c;

    rho = U[rho_i * face_offset + flipped_index];
    rhoinv = (1.) / rho;
    pdeg = static_cast<double_t>(0.0);
    edeg = static_cast<double_t>(0.0);
    dpdeg_drho = static_cast<double_t>(0.0);

    // all workitems choose the same path
    if (A_ != 0.0) {
        const auto Binv = 1.0 / B_;
        const auto x = pow_wrapper(rho * Binv, 1.0 / 3.0);
        const auto x_sqr = x * x;
        const auto x_sqr_sqrt = sqrt_wrapper(x_sqr + 1.0);
        const auto x_pow_5 = x_sqr * x_sqr * x;
        const double_t hdeg = 8.0 * A_ * Binv * (x_sqr_sqrt - 1.0);
        const double_t edeg_tmp1 = rho * hdeg - pdeg;
        const double_t edeg_tmp2 = 2.4 * A_ * x_pow_5;
        const double_t pdeg_tmp1 = A_ * (x * (2 * x_sqr - 3) * x_sqr_sqrt + 3 * asinh_wrapper(x));
        const double_t pdeg_tmp2 = 1.6 * A_ * x_pow_5;
        select_wrapper(edeg, (x > 0.001), edeg_tmp1, edeg_tmp2);
        select_wrapper(pdeg, (x > 0.001), pdeg_tmp1, pdeg_tmp2);
        dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
    }
    ek = 0.0;
    for (int dim = 0; dim < NDIM; dim++) {
        ek += U[(sx_i + dim) * face_offset + flipped_index] *
            U[(sx_i + dim) * face_offset + flipped_index] * rhoinv * 0.5;
    }
    const auto ein2_tmp2 =
        U[egas_i * face_offset + flipped_index] - ek - edeg;
    const auto ein2_mask =
        (ein2_tmp2 < (de_switch_1 * U[egas_i * face_offset + flipped_index]));
    if (!skippable(ein2_mask)) {
        const auto ein2_tmp1 =
            pow_wrapper(U[tau_i * face_offset + flipped_index], fgamma);
        select_wrapper(ein, ein2_mask, ein2_tmp1, ein2_tmp2);
    } else {
        ein = ein2_tmp2;
    }
    const auto dp_drho2 = dpdeg_drho + (fgamma - 1.0) * ein * rhoinv;
    const auto dp_deps2 = (fgamma - 1.0) * rho;
    const auto v02 = U[(sx_i + dim) * face_offset + flipped_index] * rhoinv;
    const auto p2 = (fgamma - 1.0) * ein + pdeg;
    const auto c2 = sqrt_wrapper(p2 * rhoinv * rhoinv * dp_deps2 + dp_drho2);
    const auto v2 = v02 - vg[dim];
    aml = v2 - c2;
    apl = v2 + c2;

    this_ap = max_wrapper(max_wrapper(apr, apl), double_t(0.0));
    this_am = min_wrapper(min_wrapper(amr, aml), double_t(0.0));
    const auto amp_mask = (this_ap - this_am == 0.0);
    for (int f = 0; f < nf_; f++) {
        double_t fr = v * U[f * face_offset + index];
        double_t fl = v2 * U[f * face_offset + flipped_index];

        if (f == sx_i + dim) {
            fr += p;
            fl += p2;
        } else if (f == egas_i) {
            fr += v0 * p;
            fl += v02 * p2;
        }
        if (dim == 0) {
            // levi_civita 1 2 0
            if (f == lx_i + 1) {
                fr += x[2] * p;
                fl += x[2] * p2;
            } else if (f == lx_i + 2) {
                // levi_civita 2 1 0
                fr -= x[1] * p;
                fl -= x[1] * p2;
            }
        } else if (dim == 1) {
            // levi_civita 0 2 1
            if (f == lx_i + 0) {
                fr -= x[2] * p;
                fl -= x[2] * p2;
                // 2 0 1
            } else if (f == lx_i + 2) {
                fr += x[0] * p;
                fl += x[0] * p2;
            }
        } else if (dim == 2) {
            if (f == lx_i) {
                // levi_civita 0 1 2
                fr += x[1] * p;
                fl += x[1] * p2;
                // 1 0 2
            } else if (f == lx_i + 1) {
                fr -= x[0] * p;
                fl -= x[0] * p2;
            }
        }

        if (!skippable(amp_mask)) {
            const double_t flux_tmp1 =
                (this_ap * fl - this_am * fr +
                    this_ap * this_am *
                        (U[f * face_offset + index] -
                            U[f * face_offset + flipped_index])) /
                (this_ap - this_am);
            const double_t flux_tmp2 = (fl + fr) / 2.0;
            select_wrapper(this_flux[f], amp_mask, flux_tmp2, flux_tmp1);
        } else {
            this_flux[f] = (this_ap * fl - this_am * fr +
                               this_ap * this_am *
                                   (U[f * face_offset + index] -
                                       U[f * face_offset + flipped_index])) /
                (this_ap - this_am);
        }
    }

    am = min_wrapper(am, this_am);
    ap = max_wrapper(ap, this_ap);
    return max_wrapper(ap, double_t(-am));
}

/// Fills masks required for the flux kernel. Container needs to be able to hold NDIM * 1000 elements
template<typename mask_buffer_t>
void fill_masks(mask_buffer_t &masks) {
    //boost::container::vector<bool> masks(NDIM * 10 * 10 * 10);
    constexpr int length = INX + 2;
    constexpr int length_short = INX + 1;
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
    return;
}
