//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_UNITIGER_HYDRO_HPP_
#define OCTOTIGER_UNITIGER_HYDRO_HPP_
#include <vector>

#include "safe_real.hpp"

//#define SAFE_MATH_ON
#include "../../octotiger/safe_math.hpp"

#ifdef NOHPX
#include <future>
using std::future;
using std::async;
using std::launch;
#endif

#include "octotiger/unitiger/cell_geometry.hpp"
#include "octotiger/unitiger/util.hpp"



struct timestep_t {
	double a;
	double x, y, z;
	double dt;
	int dim;
	std::vector<double> ur;
	std::vector<double> ul;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & a;
		arc & x;
		arc & y;
		arc & z;
		arc & dim;
		arc & dt;
		arc & ur;
		arc & ul;
	}
};

namespace hydro {

using x_type = std::vector<std::vector<safe_real>>;

using flux_type = std::vector<std::vector<std::vector<safe_real>>>;

template<int NDIM>
using recon_type =std::vector<std::vector<std::vector<safe_real>>>;

using state_type = std::vector<std::vector<safe_real>>;
}

template<int NDIM, int INX, class PHYSICS>
struct hydro_computer: public cell_geometry<NDIM, INX> {

	void reconstruct_ppm(std::vector<std::vector<safe_real>> &q, const std::vector<safe_real> &u, bool smooth, bool disc_detect,
			const std::vector<std::vector<double>> &disc);

	using geo = cell_geometry<NDIM,INX>;

	enum bc_type {
		OUTFLOW, PERIODIC
	};

	const hydro::recon_type<NDIM>& reconstruct(const hydro::state_type &U, const hydro::x_type&, safe_real);
//#ifdef OCTOTIGER_WITH_CUDA
	const hydro::recon_type<NDIM>& reconstruct_cuda(hydro::state_type &U, const hydro::x_type&, safe_real);
//#endif

	timestep_t flux(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q, hydro::flux_type &F, hydro::x_type &X, safe_real omega);
	timestep_t flux_experimental(const hydro::recon_type<NDIM> &Q, hydro::flux_type &F, hydro::x_type &X, safe_real omega);

	void post_process(hydro::state_type &U, const hydro::state_type &X, safe_real dx);

	void boundaries(hydro::state_type &U, const hydro::x_type &X);

	void advance(const hydro::state_type &U0, hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type &X, safe_real dx, safe_real dt,
			safe_real beta, safe_real omega);

	void output(const hydro::state_type &U, const hydro::x_type &X, int num, safe_real);

	void outputU(const hydro::state_type &U, int num, std::string test_type);

	void outputQ(const hydro::recon_type<NDIM> &Q, int num, std::string test_type);

	void outputF(const hydro::flux_type &Fl, int num, std::string test_type);

	int compareU(const hydro::state_type &U, int num, std::string test_type);

	int compareQ(const hydro::recon_type<NDIM> &Q, int num, std::string test_type);

	int compareF(const hydro::flux_type &Fl, int num, std::string test_type);

	void use_angmom_correction(int index);

	void use_smooth_recon(int field);

	void use_disc_detect(int field);

	void use_experiment(int num) {
		experiment = num;
	}

	std::vector<safe_real> get_field_sums(const hydro::state_type &U, safe_real dx);

	std::vector<safe_real> get_field_mags(const hydro::state_type &U, safe_real dx);

	hydro_computer();

	void set_bc(int face, bc_type bc) {
		bc_[face] = bc;
	}

	void set_bc(std::vector<bc_type> &&bc) {
		bc_ = std::move(bc);
	}

	inline int get_nf() const {return nf_;}
	inline int get_angmom_index() const {return angmom_index_;}
	inline const std::vector<bool>& get_smooth_field() const {return smooth_field_;}
	inline const std::vector<bool>& get_disc_detect() const {return disc_detect_;}

private:
	int experiment;
	int nf_;
	int angmom_index_;
	std::vector<bool> smooth_field_;
	std::vector<bool> disc_detect_;
	std::vector<bc_type> bc_;
}
;

#include <octotiger/unitiger/hydro_impl/hydro.hpp>

#endif /* OCTOTIGER_UNITIGER_HYDRO_HPP_ */
