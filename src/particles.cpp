#define PARTICLES_CPP

#include "octotiger/particles.hpp"
#include "octotiger/options.hpp"

std::vector<particle> load_particles() {
	std::vector<particle> particles;
	for (integer part_i = 0; part_i < opts().Part_M.size(); part_i++) {
		space_vector part_pos;
		part_pos[XDIM] = opts().Part_X[part_i];
		if (NDIM > 1) {
			part_pos[YDIM] = opts().Part_Y[part_i];
                        if (NDIM > 2) {
	                        part_pos[ZDIM] = opts().Part_Z[part_i];
                        }
                }
                particles.push_back(particle(opts().Part_M[part_i], part_pos, part_i, 1));
        }
	return particles;
}
std::vector<particle> load_particles(space_vector bmin, space_vector bmax) {
        std::vector<particle> particles;
        for (integer part_i = 0; part_i < opts().Part_M.size(); part_i++) {
                space_vector part_pos;
		space_vector part_vel;
                part_pos[XDIM] = opts().Part_X[part_i];
		part_vel[XDIM] = opts().Part_VX[part_i];
                if (NDIM > 1) {
                        part_pos[YDIM] = opts().Part_Y[part_i];
			part_vel[YDIM] = opts().Part_VY[part_i];
                        if (NDIM > 2) {
                                part_pos[ZDIM] = opts().Part_Z[part_i];
				part_vel[ZDIM] = opts().Part_VZ[part_i];
                        }
                }
                particle p = particle(opts().Part_M[part_i], part_pos, part_vel, part_i, 1); 
                if (p.is_in_boundary(bmin, bmax)) { 
                        particles.push_back(p);
                }
        }
        return particles;
}
std::vector<particle> load_particles(std::vector<particle> particles, space_vector bmin, space_vector bmax) {
        std::vector<particle> particles_in_b;
        for (auto p : particles) {
                if (p.is_in_boundary(bmin, bmax)) {
                        particles_in_b.push_back(p);
                }
        }
        return particles_in_b;
}
std::vector<integer> get_particles_inds(std::vector<particle> particles, integer cell_index) {
        std::vector<integer> inds;
        for (integer i = 0; i < particles.size(); i++) {
                const particle p = particles[i];
                const auto p_inds = p.containing_cell_index;
                const integer pindex = gindex(p_inds[XDIM], p_inds[YDIM], p_inds[ZDIM]);
                if (pindex == cell_index) {
                        inds.push_back(i);
                }
        }
        return inds;
}
int_simd_vector contain_particles(const std::vector<particle> particles, const std::array<simd_vector, NDIM> cell_X, const real dx) {
        int_simd_vector contain_parts;
        for (integer i = 0; i != simd_len; ++i) {
                space_vector bmin, bmax;
                for (integer d = 0; d < NDIM; d++) {
                        bmin[d] = cell_X[d][i] - 0.5 * dx;
                        bmax[d] = bmin[d] + dx;
                }
                const auto parts_in_cell = load_particles(particles, bmin, bmax);
                if (parts_in_cell.size() > 0) {
                        contain_parts[i] = 1;
                } else {
                        contain_parts[i] = 0;
                }
        }
        return contain_parts;
}
