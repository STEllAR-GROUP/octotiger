#ifndef PARTICLES_HPP_
#define PARTICLES_HPP_

#include "octotiger/taylor.hpp"
#include "octotiger/config/export_definitions.hpp"
#include "octotiger/space_vector.hpp"
#include "octotiger/real.hpp"

struct particle {
	space_vector pos;
	space_vector vel;
	real mass;
	expansion L; //expansion coefficient
	space_vector L_c; // AM correction
	real pot;
	space_vector g;
	integer part_type;
	integer id;
	std::array<integer, NDIM> containing_cell_index;
	std::array<integer, NDIM> containing_parent_cell_index;
	space_vector pos_rel_cell_center;

	particle() = default;
	~particle() = default;
	particle(real m, space_vector x, integer pid, integer ptype = 0) {
		(*this).mass = m;
		for (integer d = 0; d < NDIM; ++d) {
			(*this).pos[d] = x[d];
			(*this).vel[d] = 0.0;
		}
		(*this).part_type = ptype;
		(*this).id = pid;
         //       printf("created particle m: %e=%e, pos: (%e,%e,%e)=(%e,%e,%e)\n", (*this).mass, m, (*this).pos[0], (*this).pos[1], (*this).pos[2], x[0], x[1], x[2]);
	}
        particle(real m, space_vector x, space_vector v, integer ptype = 0) {
                (*this).mass = m;
                for (integer d = 0; d < NDIM; ++d) {
                        (*this).pos[d] = x[d];
                        (*this).vel[d] = v[d];
		}
		(*this).part_type = ptype;
	//	printf("created particle m: %e=%e, pos: (%e,%e,%e)=(%e,%e,%e)\n", (*this).mass, m, (*this).pos[0], (*this).pos[1], (*this).pos[2], x[0], x[1], x[2]);
        }
        particle(real m, space_vector x, space_vector v, integer pid, integer ptype = 0) {
                (*this).mass = m;
                for (integer d = 0; d < NDIM; ++d) {
                        (*this).pos[d] = x[d];
                        (*this).vel[d] = v[d];
                }
                (*this).part_type = ptype;
                (*this).id = pid;
                //printf("created particle m: %e=%e, pos: (%e,%e,%e)=(%e,%e,%e)\n", (*this).mass, m, (*this).pos[0], (*this).pos[1], (*this).pos[2], x[0], x[1], x[2]);        
        }
	bool is_in_boundary(space_vector bmin, space_vector bmax) {
		real xp = (*this).pos[XDIM];
		if ((bmin[XDIM] <= xp) && (xp < bmax[XDIM])) {
			if (NDIM > 1) {
				xp = (*this).pos[YDIM];
				if ((bmin[YDIM] <= xp) && (xp < bmax[YDIM])) {
					if (NDIM > 2) {
						xp = (*this).pos[ZDIM];
						if ((bmin[ZDIM] <= xp) && (xp < bmax[ZDIM])) {
							return true;
						}
					} else {
						return true;
					}
				}
			} else {
				return true;
			}
		}
		return false;
	}
	void update_cell_indexes(std::array<integer, NDIM> cell, std::array<integer, NDIM> parent_cell) {
		for (integer d = 0; d < NDIM; ++d) {
                        (*this).containing_cell_index[d] = cell[d];
                        (*this).containing_parent_cell_index[d] = parent_cell[d];
                }
	}
	void update_rel_pos(space_vector rel) {
                for (integer d = 0; d < NDIM; ++d) {
                        (*this).pos_rel_cell_center[d] = rel[d];
                }

	}
	template<class Arc>
        void serialize(Arc& arc, const unsigned) {
                arc & pos;
                arc & vel;
                arc & mass;
		arc & id;
                arc & L;
		arc & L_c;
		arc & containing_cell_index;
		arc & pot;
		arc & g;
        }
};

OCTOTIGER_EXPORT std::vector<particle> load_particles();
OCTOTIGER_EXPORT std::vector<particle> load_particles(space_vector bmin, space_vector bmax);
OCTOTIGER_EXPORT std::vector<particle> load_particles(std::vector<particle> particles, space_vector bmin, space_vector bmax);
OCTOTIGER_EXPORT std::vector<integer> get_particles_inds(std::vector<particle> particles, integer cell_index);
OCTOTIGER_EXPORT int_simd_vector contain_particles(const std::vector<particle> particles, const std::array<simd_vector, NDIM> cell_X, const real dx);

#endif /* PARTICLES_HPP_ */
