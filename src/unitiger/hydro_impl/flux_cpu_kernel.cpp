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

#endif
