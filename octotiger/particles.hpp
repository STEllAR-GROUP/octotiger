#ifndef PARTICLES_HPP_
#define PARTICLES_HPP_


#include "octotiger/space_vector.hpp"
#include "octotiger/real.hpp"
#include "octotiger/future.hpp"

using part_int = integer;
using fixed32 = double;
using group_int = integer;

struct group_particle {
	std::array<fixed32, NDIM> x;
	group_int g;
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & x;
		arc & g;
	}
};

struct output_particle {
	std::array<fixed32, NDIM> x;
	std::array<float, NDIM> v;
	char r;
	template<class A>
	void serialize(A && a, unsigned) {
		a & x;
		a & v;
		a & r;
	}
};

#define NTYPES_HEADER 2
#define PART_CACHE_SIZE 50

struct particle {
	space_vector pos;
	space_vector vel;
	real mass;
	expansion L; //expansion coefficient
	real pot;
	space_vector g;
	integer part_type;
	std::array<integer, NDIM> containing_cell_index;
	std::array<integer, NDIM> containing_parent_cell_index;
	space_vector pos_rel_cell_center;

	particle() = default;
	~particle() = default;
	particle(real m, space_vector x, integer ptype = 0) {
		(*this).mass = m;
		for (integer d = 0; d < NDIM; ++d) {
			(*this).pos[d] = x[d];
			(*this).vel[d] = 0.0;
		}
		(*this).part_type = ptype;
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
	bool is_in_boundary(space_vector bmin, space_vector bmax) {
		real xp = (*this).pos[XDIM];
		if ((bmin[XDIM] < xp) && (xp < bmax[XDIM])) {
			if (NDIM > 1) {
				xp = (*this).pos[YDIM];
				if ((bmin[YDIM] < xp) && (xp < bmax[YDIM])) {
					if (NDIM > 2) {
						xp = (*this).pos[ZDIM];
						if ((bmin[ZDIM] < xp) && (xp < bmax[ZDIM])) {
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
                arc & L;
		arc & containing_cell_index;
        }
};

struct particle_sample {
	std::array<fixed32, NDIM> x;
	std::array<float, NDIM> g;
	float p;
	template<class A>
	void serialize(A && a, unsigned) {
		for (int dim = 0; dim < NDIM; dim++) {
			a & x[dim];
			a & g[dim];
		}
		a & p;
	}
};

OCTOTIGER_EXPORT std::vector<particle> load_particles();
OCTOTIGER_EXPORT std::vector<particle> load_particles(space_vector bmin, space_vector bmax);
OCTOTIGER_EXPORT std::vector<particle> load_particles(std::vector<particle> particles, space_vector bmin, space_vector bmax);

#ifdef CHECK_BOUNDS
#define CHECK_PART_BOUNDS(i)                                                                                                                            \
	if( i < 0 || i >= particles_size()) {                                                                                                            \
		PRINT( "particle bound check failure %li should be between %li and %li\n", (long long) i, (long long) 0, (long long) particles_size());  \
		ALWAYS_ASSERT(false);                                                                                                                           \
	}
#else
#define CHECK_PART_BOUNDS(i)
#endif

static std::array<fixed32*, NDIM> particles_x;
static std::array<float, NDIM>* particles_v;
static char* particles_r;
static std::array<float*, NDIM> particles_g;
static float* particles_p;
static std::atomic<group_int>* particles_grp
#ifdef PARTICLES_CPP
= nullptr
#endif
;
static group_int* particles_lgrp;
static char* particles_tr;
static size_t particles_global_offset;

struct particle_global_range {
	int proc;
	std::pair<part_int, part_int> range;
};

part_int particles_size();
std::unordered_map<int, part_int> particles_groups_init();
void particles_groups_destroy();
void particles_resize(part_int, bool lock = true);
void particles_random_init();
void particles_pop_rungs();
void particles_push_rungs();
void particles_displace(double dx, double dy, double dz);
void particles_destroy();
void particles_sort_by_sph(std::pair<part_int, part_int> rng);
void particles_global_read_pos(particle_global_range, double* x, double* y, double* z, part_int offset);
void particles_global_read_pos_and_group(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, group_int* g, part_int offset);
part_int particles_sort(std::pair<part_int, part_int> rng, double xm, int xdim);
//gadget_io_header particles_read_gadget4(std::string);
#ifndef __CUDACC__
future<std::vector<int>> particles_get(int rank, const std::vector<std::pair<part_int, part_int>>& ranges);
#endif
void particles_cache_free();
void particles_group_cache_free();
std::pair<std::array<double, NDIM>, double> particles_enclosing_sphere(std::pair<part_int, part_int> rng);
//std::vector<output_particle> particles_get_sample(const range<double>& box);
std::vector<particle_sample> particles_sample(int cnt);
void particles_load(FILE* fp);
void particles_save(FILE* fp);
void particles_inc_group_cache_epoch();
void particles_global_read_rungs(particle_global_range range, char* r, part_int offset);
int particles_group_home(group_int);
#ifndef __CUDACC__
//hpx::mutex& particles_shared_mutex();
#endif
void particles_set_tracers(size_t count = 0);
std::vector<output_particle> particles_get_tracers();
void particles_memadvise_cpu();
void particles_global_read_vels(particle_global_range range, float* vx, float* vy, float* z, part_int offset);
void particles_memadvise_gpu();
std::vector<size_t> particles_rung_counts();
void particles_set_minrung(int minrung);
void particles_free();
void particles_save_glass(const char* filename);
void particles_sort_by_rung(int minrung);
double particles_active_pct();
std::pair<part_int, part_int> particles_current_range();

inline float& particles_pot(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_p[index];
}

inline fixed32& particles_pos(int dim, part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_x[dim][index];
}

inline float& particles_vel(int dim, part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_v[index][dim];
}

inline std::array<float, NDIM>* particles_vel_data() {
	return particles_v;
}

inline char& particles_rung(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_r[index];
}

inline float& particles_gforce(int dim, part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_g[dim][index];
}

inline group_int particles_group_init(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_global_offset + index;
}

inline std::atomic<group_int>& particles_group(part_int index) {
	CHECK_PART_BOUNDS(index);
	//ASSERT(particles_grp);
	return particles_grp[index];
}

inline group_int& particles_lastgroup(part_int index) {
	CHECK_PART_BOUNDS(index);
	//ASSERT(particles_lgrp);
	return particles_lgrp[index];
}

inline char& particles_tracer(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_tr[index];
}

inline particle particles_get_particle(part_int index) {
	static bool do_groups = false; //get_options().do_groups;
	static bool do_tracers = false; //get_options().do_tracers;
	CHECK_PART_BOUNDS(index);
	particle p;
	for (int dim = 0; dim < NDIM; dim++) {
		p.pos[dim] = particles_pos(dim, index);
		p.vel[dim] = particles_vel(dim, index);
	}
//	p.r = particles_rung(index);
//	if (do_groups) {
//		p.lg = particles_lastgroup(index);
//	}
//	if (do_tracers) {
//		p.t = particles_tracer(index);
//	}
	return p;
}

inline void particles_set_particle(particle p, part_int index) {
	static bool do_groups = false; //get_options().do_groups;
	static bool do_tracers = false; //get_options().do_tracers;
	CHECK_PART_BOUNDS(index);
	for (int dim = 0; dim < NDIM; dim++) {
		particles_pos(dim, index) = p.pos[dim];
		particles_vel(dim, index) = p.vel[dim];
	}
//	particles_rung(index) = p.r;
//	if (do_groups) {
//		particles_lastgroup(index) = p.lg;
//	}
//	if (do_tracers) {
//		particles_tracer(index) = p.t;
//	}
}

struct energies_t {
	double pot;
	double kin;
	double tckin;
	double xmom;
	double ymom;
	double cosmic;
	double zmom;
	double nmom;
	energies_t() {
		pot = kin = 0.f;
		xmom = ymom = zmom = nmom = 0.0;
		cosmic = 0.0;
		tckin = 0.0;
	}
	energies_t& operator+=(const energies_t& other) {
		pot += other.pot;
		kin += other.kin;
		cosmic += other.cosmic;
		xmom += other.xmom;
		ymom += other.ymom;
		zmom += other.zmom;
		nmom += other.nmom;
		tckin += other.tckin;
		return *this;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & xmom;
		arc & ymom;
		arc & cosmic;
		arc & zmom;
		arc & nmom;
		arc & pot;
		arc & kin;
		arc & tckin;
	}
};

energies_t particles_sum_energies();

#endif /* PARTICLES_HPP_ */
