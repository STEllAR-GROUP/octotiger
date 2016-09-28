/*
 * fmm.hpp
 *
 *  Created on: Sep 27, 2016
 *      Author: dmarce1
 */

#include "defs.hpp"
#include "geometry.hpp"
#include <taylor.hpp>
#include "simd.hpp"

#define NMX 4
#define NPX (INX/NMX)
#define NP3 (NPX*NPX*NPX)
#define NM3 (NMX*NMX*NMX)

#define ORDER 4
#define IWIDTH 2

struct force_box {
	std::vector<real> phi;
	std::vector<real> g;
	force_box() :
		phi(MN3), g(MN3) {
	}
};

class particle_box {
private:
	inline static integer pindex(integer i, integer j, integer k) {
		return (NPX * NPX) * i + NPX * j + k;
	}
	std::vector<real> M;
public:
	void set_mass(integer i, integer j, integer k, real m) {
		M[pindex(i, j, k)] = m;
	}
	force_box interact(const particle_box& other, const geo::direction& dir, integer di, real dx) {
		force_box g;
		for (integer i1 = 0; i1 != NPX; ++i1) {
			for (integer j1 = 0; j1 != NPX; ++j1) {
				for (integer k1 = 0; k1 != NPX; ++k1) {
					const integer iiif = pindex(i1, j1, k1);
					g.phi[iiif] = ZERO;
					g.g[iiif] = ZERO;
					for (integer i0 = 0; i0 != NPX; ++i0) {
						const real x = ((i1 - i0) + PNX * di * dir[XDIM]) * dx;
						for (integer j0 = 0; j0 != NPX; ++j0) {
							const real y = ((j1 - j0) + PNX * di * dir[YDIM]) * dx;
							for (integer k0 = 0; k0 != NPX; ++k0) {
								const integer iiim = pindex(i0, j0, k0);
								const real z = ((k1 - k0) + PNX * di * dir[ZDIM]) * dx;
								const real r = std::sqrt(x * x + y * y + z * z);
								if (r != 0.0) {
									g.phi[iiif] -= M[iiim] / r;
									g.g[iiif][XDIM] -= M[iiim] * x / (r * r * r);
									g.g[iiif][YDIM] -= M[iiim] * y / (r * r * r);
									g.g[iiif][ZDIM] -= M[iiim] * z / (r * r * r);
								}
							}
						}
					}
				}
			}
		}
		return g;
	}
	taylor<ORDER, real> get_multipole(real dx) const {
		taylor<ORDER, real> mpole;
		for (integer i = 0; i != NPX; ++i) {
			for (integer j = 0; j != NPX; ++j) {
				for (integer k = 0; k != NPX; ++k) {
					const integer iii = pindex(i, j, k);
					space_vector x;
					x[XDIM] = (real(i) - real(NPX - 1) / 2.0) * dx;
					x[YDIM] = (real(j) - real(NPX - 1) / 2.0) * dx;
					x[ZDIM] = (real(k) - real(NPX - 1) / 2.0) * dx;
					mpole.add_monopole(x, M[iii]);
				}
			}
		}
		return mpole;
	}
};

class fmm {
private:
	inline static integer mindex(integer i, integer j, integer k) {
		return (NMX * NMX) * i + NMX * j + k;
	}
	std::vector<particle_box> P;
	std::vector<force_box> G;
	std::vector<taylor<ORDER, real>> M;
	std::vector<taylor<ORDER, real>> L;
	std::vector<space_vector> com;
	space_vector origin;
	real dx;
public:
	fmm(const space_vector& _o, real _dx) :
		P(NM3), M(NM3), L(NM3), com(NM3), origin(_o), dx(_dx) {
	}
	void set_particles(const std::vector<real>& rho);
	std::vector<taylor<ORDER, real>> get_multipoles() const;
	void set_multipoles(const std::vector<taylor<ORDER, real>>& data, const geo::octant& ci);
	struct boundary_type {
		std::vector<taylor<ORDER, real>> M;
		std::vector<particle_box> P;
	};
	boundary_type get_boundary(const geo::direction& dir);
	void boundary_interactions(const boundary_type&);
	void self_interactions();
	void center_multipoles();
};
