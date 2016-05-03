/*
 * grid_scf.cpp
 *
 *  Created on: Oct 17, 2015
 *      Author: dmarce1
 */

#include "grid.hpp"
#include "node_server.hpp"
#include "lane_emden.hpp"
#include "node_client.hpp"
#include "options.hpp"
#include "eos.hpp"

const real bibi_mu = 0.5;
const real bibi_don_core_frac = 2.0 * 0.2374;
const real bibi_acc_core_frac = 2.0 * 0.2377;


typedef typename node_server::scf_update_action scf_update_action_type;
HPX_REGISTER_ACTION (scf_update_action_type);


typedef typename node_server::rho_mult_action rho_mult_action_type;
HPX_REGISTER_ACTION (rho_mult_action_type);


hpx::future<void> node_client::rho_mult(real f0, real f1) const {
	return hpx::async<typename node_server::rho_mult_action>(get_gid(), f0, f1);
}

hpx::future<real> node_client::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x) const {
	return hpx::async<typename node_server::scf_update_action>(get_gid(), com, omega, c1, c2, c1_x, c2_x);
}

void node_server::rho_mult(real f0, real f1) {
	std::vector<hpx::future<void>> futs;
	if (is_refined) {
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.rho_mult(f0, f1));
		}
	}
	for( integer i = 0; i != H_NX; ++i) {
		for( integer j = 0; j != H_NX; ++j) {
			for( integer k = 0; k != H_NX; ++k) {
				grid_ptr->hydro_value(frac0_i, i, j, k) *= f0;
				grid_ptr->hydro_value(frac1_i, i, j, k) *= f1;
				grid_ptr->hydro_value(rho_i,i,j,k) =
						grid_ptr->hydro_value(frac0_i, i, j, k) +grid_ptr->hydro_value(frac1_i, i, j, k) ;
			}
		}
	}
	for( auto&& fut : futs ) {
		fut.get();
	}
}

real node_server::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x) {
	grid::set_omega(omega);
	std::vector<hpx::future<real>> futs;
	real res;
	if (is_refined) {
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.scf_update(com, omega,c1,c2,c1_x,c2_x));
		}
		res = ZERO;
	} else {
		res = grid_ptr->scf_update(com, omega,c1,c2,c1_x,c2_x);
	}
	exchange_interlevel_hydro_data();
	collect_hydro_boundaries();
	for (auto&& fut : futs) {
		res += fut.get();
	}
	return res;
}



struct scf_parameters {
	real M1;
	real M2;
	real R1;
	real R2;
	real a;
	real fill1;
	real fill2;
	real omega;
	real G;
	real q;
	std::shared_ptr<eos> eos1;
	std::shared_ptr<eos> eos2;
	real l1_x;
	real c1_x;
	real c2_x;
	scf_parameters(real M1, real M2, real a, real fill1, real fill2) {
		this->M1 = M1;
		G = 1.0;
		this->M2 = M2;
		this->a = a;
		this->fill1 = 1.0;
		this->fill2 = 1.0;
		const real V1 = find_V(M1 / M2) * std::pow(a, 3.0) * fill1;
		const real V2 = find_V(M2 / M1) * std::pow(a, 3.0) * fill2;
		const real c = 4.0 * M_PI / 3.0;
		R1 = std::pow(V1 / c, 1.0 / 3.0);
		R2 = std::pow(V2 / c, 1.0 / 3.0);
		q = M2 / M1;
		c1_x = -a * M2 / (M1 + M2);
		c2_x = +a * M1 / (M1 + M2);
		l1_x = a * (0.5 - 0.227 * log10(q)) + c1_x;
		omega = std::sqrt((G * (M1 + M2)) / (a * a * a));
		eos2 = std::make_shared < bipolytropic_eos
				> (M2, R2, 3.0, 1.5, bibi_don_core_frac, bibi_don_core_frac
						* bibi_mu);
		eos1 = std::make_shared < bipolytropic_eos
				> (M1, R1, 3.0, 1.5, bibi_acc_core_frac, bibi_acc_core_frac
						* bibi_mu);
		printf("%e %e %e %e\n", c1_x, c2_x, R1, R2);
	}
};

static scf_parameters& initial_params() {
	static scf_parameters a(1.0, 0.1, 0.8, 0.5, 0.5);
	return a;
}

real grid::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x) {
	const real w0 = 1.0/5.0;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer D = G_BW - H_BW;
				const integer iiih = hindex(i, j, k);
				const integer iiig = gindex(i + D, j + D, k + D);
				const real x = X[XDIM][iiih];
				const real y = X[YDIM][iiih];
				const real z = X[ZDIM][iiih];
				const real R = std::sqrt(std::pow(x - com, 2) + y * y);
				real rho = U[rho_i][iiih];
				const real phi_eff = G[phi_i][iiig] - 0.5 * std::pow(omega * R, 2);
				const real fx = G[gx_i][iiig] + (x - com) * std::pow(omega, 2);
				const real fy = G[gy_i][iiig] + y * std::pow(omega, 2);
				const real fz = G[gz_i][iiig];

				bool is_donor_side = x > initial_params().l1_x + com;
				real C  = is_donor_side ? c2 : c1;
				real x0  = is_donor_side ? c2_x : c1_x;
				real g = (x - x0 - com)*fx + y*fy + z*fz;
				auto this_eos = is_donor_side ? initial_params().eos2 : initial_params().eos1;

				real new_rho, eint;
				if( g < 0.0 ) {
					new_rho = std::max(this_eos->enthalpy_to_density(std::max(C - phi_eff,0.0)),1.0e-10);
				} else {
					new_rho = 1.0e-10;
				}
				rho = (1.0-w0)*rho + w0*new_rho;
				eint = this_eos->pressure(rho) / (fgamma-1.0);

				U[rho_i][iiih] = rho;
				U[frac0_i][iiih] = is_donor_side ? 0.0 : rho;
				U[frac1_i][iiih] = is_donor_side ? rho : 0.0;
				U[sx_i][iiih] = -omega * y * rho;
				U[sy_i][iiih] = +omega * (x - com) * rho;
				U[sz_i][iiih] = 0.0;
				U[egas_i][iiih] = eint + std::pow(R * omega, 2) * rho / 2.0;
				U[tau_i][iiih] = std::pow(eint, 1.0 / fgamma);
				U[zx_i][iiih] = 0.0;
				U[zy_i][iiih] = 0.0;
				U[zz_i][iiih] = dx * dx * omega * rho / 6.0;
			}
		}
	}

	return 0.0;
}

real interpolate(real x1, real x2, real x3, real x4, real y1, real y2, real y3,
		real y4, real x) {
//	printf( "%e %e %e %e %e\n", x1, x2, x3, x4, x);
	x1 -= x2;
	x3 -= x2;
	x4 -= x2;
	x -= x2;

	real a, b, c, d;

	a = y2;

	b = (x3 * x4) / (x1 * (x1 - x3) * (x1 - x4)) * y1;
	b += -(1.0 / x1 + (x3 + x4) / (x3 * x4)) * y2;
	b += (x1 * x4) / ((x1 - x3) * x3 * (x4 - x3)) * y3;
	b += (x1 * x3) / ((x1 - x4) * x4 * (x3 - x4)) * y4;

	c = -(x3 + x4) / (x1 * (x1 - x3) * (x1 - x4)) * y1;
	c += (x1 + x3 + x4) / (x1 * x3 * x4) * y2;
	c += (x1 + x4) / (x3 * (x1 - x3) * (x3 - x4)) * y3;
	c += (x3 + x1) / (x4 * (x1 - x4) * (x4 - x3)) * y4;

	d = y1 / (x1 * (x1 - x3) * (x1 - x4));
	d -= y2 / (x1 * x3 * x4);
	d += y3 / (x3 * (x3 - x1) * (x3 - x4));
	d += y4 / ((x1 - x4) * (x3 - x4) * x4);

	return a + b * x + c * x * x + d * x * x * x;

}

bool find_root(std::function<real(real)>& func, real xmin, real xmax,
		real& root) {
	real xmid;
	while ((xmax - xmin) > 1.0e-10) {
		xmid = (xmax + xmin) / 2.0;
		if (func(xmid) * func(xmax) < 0.0) {
			xmin = xmid;
		} else {
			xmax = xmid;
		}
	}
	root = xmid;
}

void node_server::run_scf() {

	const auto interp_force =
			[](const line_of_centers_t& loc, real omega, real x) {
				real a, b, cr, cl, x0;
				real g_ll, g_l, g_r, g_rr;
				real xl, xll, xr, xrr;
				for( std::size_t i = 8; i != loc.size() - 4; i += 4) {
					const integer il = i - 4;
					const integer ir = i;
					xl = loc[il].first;
					xr = loc[ir].first;
					if( (x - xl)*(x-xr) <= 0.0 ) {
						const integer ill = il - 4;
						const integer irr = ir + 4;
						xll = loc[ill].first;
						xrr = loc[irr].first;
						g_ll = loc[ill].second[gx_i+NF];
						g_rr = loc[irr].second[gx_i+NF];
						g_l = loc[il].second[gx_i+NF];
						g_r = loc[ir].second[gx_i+NF];
						break;
					}
				}
				return interpolate(xll, xl, xr, xrr, g_ll, g_l, g_r, g_rr, x) + omega*omega*x;
			};

	const auto interp_phi =
			[](const line_of_centers_t& loc, real omega, real x) {
				real a, b, cr, cl, x0;
				real phi_ll, phi_l, phi_r, phi_rr;
				real xl, xll, xr, xrr;
				for( std::size_t i = 8; i != loc.size() - 4; i += 4) {
					const integer il = i - 4;
					const integer ir = i;
					xl = loc[il].first;
					xr = loc[ir].first;
					if( (x - xl)*(x-xr) <= 0.0 ) {
						const integer ill = il - 4;
						const integer irr = ir + 4;
						xll = loc[ill].first;
						xrr = loc[irr].first;
						phi_ll = loc[ill].second[pot_i]/ loc[ill].second[rho_i];
						phi_rr = loc[irr].second[pot_i]/ loc[irr].second[rho_i];
						phi_l = loc[il].second[pot_i]/ loc[il].second[rho_i];
						phi_r = loc[ir].second[pot_i]/ loc[ir].second[rho_i];
						break;
					}
				}
				return interpolate(xll, xl, xr, xrr, phi_ll, phi_l, phi_r, phi_rr, x) - 0.5* omega*omega*x*x;
			};

	solve_gravity(false);
	char* ptr;
	real omega = initial_params().omega;
	for (integer i = 0; i != 1000; ++i) {
		asprintf(&ptr, "X.scf.%i.silo", int(i));
		auto& params = initial_params();
		if( i % 5 == 0)
		output(ptr);
		free(ptr);
		auto diags = diagnostics();
		real this_m = diags.grid_sum[rho_i];
		real f0 = params.M1 / diags.grid_sum[frac0_i];
		real f1 = params.M2 / diags.grid_sum[frac1_i];
		real f = (params.M1+params.M2) / diags.grid_sum[rho_i];
//		f = (f + 1.0)/2.0;
	//	printf( "%e %e \n", f0, f1);
		rho_mult(f0,f1);
		params.eos1->set_d0(f0*params.eos1->d0());
		params.eos2->set_d0(f1*params.eos2->d0());
		solve_gravity(false);

		auto axis = grid_ptr->find_axis();
		auto loc = line_of_centers(axis);

		real l1_x, c1_x, c2_x;
		real l1_phi;
		auto find_zeros = [&](real & l1_x, real& c1_x, real& c2_x, real omega) {
			const real dx = params.a / 100.0;
			bool l1_found = false;
			real x0;
			std::function<real(real)> f([&](real x) {
						return interp_force(loc,omega,x);
					});
			for( real x = 0.0; x < params.a; x += dx ) {
				const real xl = x - dx / 2.0;
				const real xr = x + dx / 2.0;
				real vl = interp_force(loc,omega,xl);
				real vr = interp_force(loc,omega,xr);
				if( vl * vr <= 0.0 ) {
					bool rc = find_root(f, xl-dx/2.0, xr+dx/2.0, x0);
					if( l1_found ) {
						c2_x = x0;
						break;
					} else {
						l1_x = x0;
						l1_phi = interp_phi(loc,omega,x0);
					}
					l1_found = true;
				}
			}
			for( real x = 0.0; x > -params.a; x -= dx ) {
				const real xl = x - dx / 2.0;
				const real xr = x + dx / 2.0;
				real vl = interp_force(loc,omega,xl);
				real vr = interp_force(loc,omega,xr);
				if( vl * vr <= 0.0 ) {
					bool rc = find_root(f, xl-dx/2.0, xr+dx/2.0, x0);
					c1_x = x0;
					break;
				}
			}
		};

		real com = axis.second[0];
		std::function<real(real)> ff([&](real omega){
			real c1_x, c2_x, l1_x;
			find_zeros(l1_x, c1_x,c2_x,omega);
			return std::abs(c2_x - c1_x) -  params.a;
		});

		real new_omega;
		find_root(ff, omega/2.0, 2.0* omega, new_omega);
		omega = new_omega;
		//	omega = 0.5*(new_omega+omega);
		find_zeros(l1_x,c1_x,c2_x, omega);

		real phi_1 = interp_phi(loc, omega, c1_x);
		real phi_2 = interp_phi(loc, omega, c2_x);

		real h_1 = params.eos1->h0();
		real h_2 = params.eos2->h0();

		real fill = 1.0;
		real c_1 = l1_phi*fill + phi_1*(1.0-fill);
		real c_2 = l1_phi*fill + phi_2*(1.0-fill);
		params.eos1->set_h0(c_1 - phi_1);
		params.eos2->set_h0(c_2 - phi_2);
		auto e1 = std::dynamic_pointer_cast<bipolytropic_eos>(params.eos1);
		auto e2 = std::dynamic_pointer_cast<bipolytropic_eos>(params.eos2);

		real M1 = diags.grid_sum[frac0_i];
		real M2 = diags.grid_sum[frac1_i];
		printf( "%e %e %e %e %e %e %e %e %e %e %e %e\n", c1_x, c2_x, l1_x, l1_phi, M1, M2, omega, e1->s0(), e2->s0(), e2->get_frac(), f0, f1 );
		if( i % 10 == 0) {
			regrid(me.get_gid(), false);
		}


	//	real old_frac = e1->get_frac();
	//	printf( "s1 : %e s2 : %e\n", e1->s0(), e2->s0() );
		std::function<double(double)> fff = [&](real frac) {
			e2->set_frac(frac);
			return e1->s0() - e2->s0();
		};
		real new_frac;
		find_root( fff, 0.0, 1.0, new_frac );
//		e1->set_frac(old_frac);
	//	printf( "NF: %e\n", new_frac);
		scf_update(com, omega, c_1, c_2, c1_x, c2_x);
		solve_gravity(false);


	}
}

std::vector<real> scf_binary(real x, real y, real z, real) {
	std::vector<real> u(NF, real(0));
	static auto& params = initial_params();
	std::shared_ptr<const eos> this_eos;
	real rho, r, ei;
	if (x < params.l1_x) {
		r = std::sqrt(std::pow(x - params.c1_x, 2) + y * y + z * z);
		this_eos = params.eos1;
	} else {
		r = std::sqrt(std::pow(x - params.c2_x, 2) + y * y + z * z);
		this_eos = params.eos2;
	}
	rho = std::max(this_eos->density_at(r), 1.0e-10);
	ei = this_eos->pressure(rho) / (fgamma - 1.0);
	u[rho_i] = rho;
	u[frac0_i] = x > params.l1_x ? 0.0 : rho;
	u[frac1_i] = x > params.l1_x ? rho : 0.0;
	u[egas_i] = ei + 0.5 * (x * x + y * y) * params.omega * params.omega;
	u[sx_i] = -y * params.omega * rho;
	u[sy_i] = +x * params.omega * rho;
	u[sz_i] = 0.0;
	u[tau_i] = std::pow(ei, 1.0 / fgamma);
	return u;
}
