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

//0.5=.313
//0.6 .305

const real targ_frac = 0.06666666666666666;
const real bibi_mu = 1.0/2.16;
const real bibi_acc_core_frac = 0.5;
const real bibi_don_core_frac = 0.1;
const real rho_floor = 1.0e-12;

const real nc1 = 5.0;
const real nc2 = 5.0;
const real ne1 = 1.5;
const real ne2 = 1.5;

typedef typename node_server::scf_update_action scf_update_action_type;
HPX_REGISTER_ACTION (scf_update_action_type);


typedef typename node_server::rho_mult_action rho_mult_action_type;
HPX_REGISTER_ACTION (rho_mult_action_type);


hpx::future<void> node_client::rho_mult(real f0, real f1) const {
	return hpx::async<typename node_server::rho_mult_action>(get_gid(), f0, f1);
}

hpx::future<real> node_client::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, accretor_eos e1, donor_eos e2) const {
	return hpx::async<typename node_server::scf_update_action>(get_gid(), com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
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
				grid_ptr->hydro_value(spc_ac_i, i, j, k) *= f0;
				grid_ptr->hydro_value(spc_dc_i, i, j, k) *= f1;
				grid_ptr->hydro_value(spc_ae_i, i, j, k) *= f0;
				grid_ptr->hydro_value(spc_de_i, i, j, k) *= f1;
				grid_ptr->hydro_value(rho_i,i,j,k) = 0.0;
				for( integer si = 0; si != NSPECIES; ++si) {
					grid_ptr->hydro_value(rho_i,i,j,k) +=grid_ptr->hydro_value(spc_i+si, i, j, k);
				}
			}
		}
	}
	for( auto&& fut : futs ) {
		fut.get();
	}
}

real node_server::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, accretor_eos e1, donor_eos e2) {
	grid::set_omega(omega);
	std::vector<hpx::future<real>> futs;
	real res;
	if (is_refined) {
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.scf_update(com, omega,c1,c2,c1_x,c2_x, l1_x,e1,e2));
		}
		res = ZERO;
	} else {
		res = grid_ptr->scf_update(com, omega,c1,c2,c1_x,c2_x, l1_x,e1,e2);
	}
	exchange_interlevel_hydro_data();
	collect_hydro_boundaries();
	for (auto&& fut : futs) {
		res += fut.get();
	}
	current_time += 1.0e-100;
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
	std::shared_ptr<accretor_eos> eos1;
	std::shared_ptr<donor_eos> eos2;
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
#ifdef BIBI
		eos2 = std::make_shared < donor_eos
				> (M2, R2, nc2, ne2, bibi_don_core_frac, bibi_don_core_frac
						* bibi_mu);
		eos1 = std::make_shared < accretor_eos
				> (M1, R1, nc1, ne1, bibi_acc_core_frac, bibi_acc_core_frac
						* bibi_mu);
#endif

#ifdef CWD
		eos2 = std::make_shared < donor_eos
				> ();
		eos1 = std::make_shared < accretor_eos
				> ();
		eos1->initialize(M1,R1);
		eos2->initialize(M2,R2);
#endif

		printf("%e %e %e %e\n", c1_x, c2_x, R1, R2);
	}
};

//0.15=0.77
//0.30=0.41
//0.33=0.35
static scf_parameters& initial_params() {
	static scf_parameters a(1.5, 0.35, 1.0, 0.5, 0.5);
	return a;
}

real grid::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, accretor_eos eos_1, donor_eos eos_2) {
	const real w0 = 1.0/10.0;
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

				bool is_donor_side = x > l1_x + com;
				real C  = is_donor_side ? c2 : c1;
				real x0  = is_donor_side ? c2_x : c1_x;
				real g = (x - x0 - com)*fx + y*fy + z*fz;
				auto this_eos = is_donor_side ? eos_2 : eos_1;

				real new_rho, eint;
				if( g < 0.0 ) {
					new_rho = std::max(this_eos.enthalpy_to_density(std::max(C - phi_eff,0.0)),rho_floor);
				} else {
					new_rho = rho_floor;
				}
				rho = (1.0-w0)*rho + w0*new_rho;
				eint = this_eos.pressure(rho) / (fgamma-1.0);

				U[rho_i][iiih] = rho;
#ifdef BIBI
				if( C - phi_eff > 0.0 && g < 0.0) {
					U[spc_ac_i][iiih] = rho > this_eos.dE() ? (is_donor_side ? 0.0 : rho) : 0.0;
					U[spc_dc_i][iiih] = rho > this_eos.dE() ? (is_donor_side ? rho : 0.0) : 0.0;
					U[spc_ae_i][iiih] = rho <= this_eos.dE() ? (is_donor_side ? 0.0 : rho) : 0.0;
					U[spc_de_i][iiih] = rho <= this_eos.dE() ? (is_donor_side ? rho : 0.0) : 0.0;
					U[spc_vac_i][iiih] = 0.0;
				} else {
					U[spc_ac_i][iiih] =
					U[spc_dc_i][iiih] =
					U[spc_ae_i][iiih] =
					U[spc_de_i][iiih] =  0.0;
					U[spc_vac_i][iiih] = rho_floor;
				}
#endif

#ifdef CWD
				U[spc_ac_i][iiih] = (is_donor_side ? 0.0 : rho);
				U[spc_dc_i][iiih] = (is_donor_side ? rho : 0.0);
				U[spc_ae_i][iiih] = 0.0;
				U[spc_de_i][iiih] = 0.0;
#endif
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
	while ((xmax - xmin) > rho_floor) {
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

	const auto interp_rho =
			[](const line_of_centers_t& loc, real x) {
				real a, b, cr, cl, x0;
				real rho_ll, rho_l, rho_r, rho_rr;
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
						rho_ll = loc[ill].second[rho_i];
						rho_rr = loc[irr].second[rho_i];
						rho_l = loc[il].second[rho_i];
						rho_r = loc[ir].second[rho_i];
						break;
					}
				}
				return interpolate(xll, xl, xr, xrr,rho_ll,rho_l, rho_r, rho_rr, x);
			};

	solve_gravity(false);
	char* ptr;
	real omega = initial_params().omega;
	for (integer i = 0; i != 100; ++i) {
		asprintf(&ptr, "X.scf.%i.silo", int(i));
		auto& params = initial_params();
		set_omega_and_pivot();
		if( i % 5 == 0)
		output(ptr);
		free(ptr);
		auto diags = diagnostics();


		std::vector<space_vector> moment_coms(NSPECIES);
		moment_coms[spc_ac_i - spc_i] = diags.primary_com;
		moment_coms[spc_ae_i - spc_i] = diags.primary_com;
		moment_coms[spc_dc_i - spc_i] = diags.secondary_com;
		moment_coms[spc_de_i - spc_i] = diags.secondary_com;
		moment_coms[spc_vac_i - spc_i] = diags.grid_com;
		auto moments1 = frac_moments(moment_coms);
		moment_coms[spc_ac_i - spc_i] =
		moment_coms[spc_ae_i - spc_i] =
		moment_coms[spc_dc_i - spc_i] =
		moment_coms[spc_de_i - spc_i] = diags.grid_com;
		auto moments2 = frac_moments(moment_coms);
		auto Iorb = moments2;
		for( integer si = 0; si != NSPECIES; ++si) {
			Iorb[si] -= moments1[si];
		}
		auto Ispin = moments1;

		real iorb = 0.0;
		real is1 = 0.0;
		real is2 = 0.0;
		iorb += Iorb[spc_ac_i-spc_i];
		iorb += Iorb[spc_ae_i-spc_i];
		iorb += Iorb[spc_dc_i-spc_i];
		iorb += Iorb[spc_de_i-spc_i];
		is1 += Ispin[spc_ac_i-spc_i];
		is1 += Ispin[spc_ae_i-spc_i];
		is2 += Ispin[spc_dc_i-spc_i];
		is2 += Ispin[spc_de_i-spc_i];


		real j1 = 0.0;
		real j2 = 0.0;
		real m1 = diags.primary_sum[rho_i];
		real m2 = diags.secondary_sum[rho_i];
		j1 -= diags.primary_com_dot[XDIM] * (diags.primary_com[YDIM] - diags.grid_com[YDIM]) * m1;
		j1 += diags.primary_com_dot[YDIM] * (diags.primary_com[XDIM] - diags.grid_com[XDIM]) * m1;
		j2 -= diags.secondary_com_dot[XDIM] * (diags.secondary_com[YDIM] - diags.grid_com[YDIM]) * m2;
		j2 += diags.secondary_com_dot[YDIM] * (diags.secondary_com[XDIM] - diags.grid_com[XDIM]) * m2;
		const real jorb = j1 + j2;
		j1 = diags.primary_sum[zz_i] - j1;
		j2 = diags.secondary_sum[zz_i] - j2;
		real spin_ratio = (is1+is2)/(iorb);
		real this_m = diags.grid_sum[rho_i];
		real f0 = params.M1 / (diags.grid_sum[spc_ac_i]+diags.grid_sum[spc_ae_i]);
		real f1 = params.M2 / (diags.grid_sum[spc_dc_i]+diags.grid_sum[spc_de_i]);
		real f = (params.M1+params.M2) / diags.grid_sum[rho_i];
//		f = (f + 1.0)/2.0;
	//	printf( "%e %e \n", f0, f1);
		rho_mult(f0,f1);
		solve_gravity(false);

		auto axis = grid_ptr->find_axis();
		auto loc = line_of_centers(axis);

		real l1_x, c1_x, c2_x, l2_x, l3_x;
		real l1_phi, l2_phi, l3_phi;
		auto find_zeros = [&](real & l1_x, real& c1_x, real& c2_x, real omega) {
			const real dx = params.a / 100.0;
			bool l1_found = false;
			bool c1_found = false;
			bool c2_found = false;
			real x0;
			std::function<real(real)> f([&](real x) {
						return interp_force(loc,omega,x);
					});
			for( real x = 0.0; x < 3.0 * params.a; x += dx ) {
				const real xl = x - dx / 2.0;
				const real xr = x + dx / 2.0;
				real vl = interp_force(loc,omega,xl);
				real vr = interp_force(loc,omega,xr);
				if( vl * vr <= 0.0 ) {
					bool rc = find_root(f, xl-dx/2.0, xr+dx/2.0, x0);
					if( l1_found ) {
						if( c2_found ) {
							l2_phi = interp_phi(loc,omega,x0);
							l2_x = x0;
							break;
						} else {
							c2_found = true;
							c2_x = x0;
						}
					} else {
						l1_x = x0;
						l1_phi = interp_phi(loc,omega,x0);
						l1_found = true;
					}
				}
			}
			for( real x = 0.0; x > -3.0*params.a; x -= dx ) {
				const real xl = x - dx / 2.0;
				const real xr = x + dx / 2.0;
				real vl = interp_force(loc,omega,xl);
				real vr = interp_force(loc,omega,xr);
				if( vl * vr <= 0.0 ) {
					bool rc = find_root(f, xl-dx/2.0, xr+dx/2.0, x0);
					if( c1_found ) {
						l3_phi = interp_phi(loc,omega,x0);
						l3_x = x0;
						break;
					} else {
						c1_found = true;
						c1_x = x0;
					}
				}
			}
		};

		real com = axis.second[0];
		std::function<real(real)> ff([&](real omega){
			real c1_x, c2_x, l1_x;
			find_zeros(l1_x, c1_x,c2_x,omega);
			return std::abs(c2_x) -  params.a * params.M1 / (params.M2+params.M1);
		});

		real new_omega;
		find_root(ff, omega/2.0, 2.0* omega, new_omega);
		omega = new_omega;
		//	omega = 0.5*(new_omega+omega);
		find_zeros(l1_x,c1_x,c2_x, omega);
		real rho1 = interp_rho(loc,c1_x);
		real rho2 = interp_rho(loc,c2_x);
		params.eos1->set_d0(rho1);
		params.eos2->set_d0(rho2);

		real phi_1 = interp_phi(loc, omega, c1_x);
		real phi_2 = interp_phi(loc, omega, c2_x);

		real h_1 = params.eos1->h0();
		real h_2 = params.eos2->h0();

		real fill = 1.0;
		real phi = std::min(l3_phi,l2_phi);
		real c_1 = phi*fill + phi*(1.0-fill);
		real c_2 = phi*fill + phi*(1.0-fill);
		params.eos1->set_h0(c_1 - phi_1);
		params.eos2->set_h0(c_2 - phi_2);
		auto e1 = params.eos1;
		auto e2 = params.eos2;

		real M1 = diags.grid_sum[spc_ac_i]+diags.grid_sum[spc_ae_i];
		real M2 = diags.grid_sum[spc_dc_i]+diags.grid_sum[spc_de_i];
		real core_frac_1 = diags.grid_sum[spc_ac_i]/M1;
		real core_frac_2 = diags.grid_sum[spc_dc_i]/M2;
		const real eptot = diags.grid_sum[pot_i];
		const real ektot = diags.grid_sum[egas_i] - 0.5*eptot;
		const real virial = (2.0*ektot + 0.5 * eptot) / (2.0*ektot - 0.5*eptot);
		const real v1 = diags.primary_volume;
		const real v2 = diags.secondary_volume;
		const real vfactor = 4.0 / 3.0 * M_PI;
		const real r1 = std::pow(v1/vfactor,1.0/3.0);
		const real r2 = std::pow(v2/vfactor,1.0/3.0);
		const real g1 = std::sqrt(is1 / (r1*r1) / m1);
		const real g2 = std::sqrt(is2 / (r2*r2) / m2);
		if( i % 5 == 0 )
			printf( "%13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s\n", "rho1", "rho2", "c1_x", "c2_x", "l1_phi", "l2_phi", "l3_phi", "M1", "M2", "omega", "virial", "core_frac_1", "core_frac_2", "e1frac", "iorb", "is1", "is2","spin_ratio", "r1", "r2", "g1", "g2"  );
		printf( "%13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e\n", rho1, rho2, c1_x, c2_x,  l1_phi, l2_phi,  l3_phi, M1, M2, omega,   virial, core_frac_1, core_frac_2, e1->get_frac(), iorb, is1, is2, spin_ratio, r1, r2, g1, g2 );
		if( i % 10 == 0) {
			regrid(me.get_gid(), false);
		}



	//	real old_frac = e1->get_frac();
	//	printf( "s1 : %e s2 : %e\n", e1->s0(), e2->s0() );
		real e1f = e1->get_frac();
		e1f = 0.9 * e1f + 0.1 * std::pow( e1f, targ_frac / core_frac_1);
		e1->set_frac(e1f);

#ifdef BIBI
		std::function<double(double)> fff = [&](real frac) {
			e2->set_frac(frac);
			return e1->s0() - e2->s0();
		};
		real new_frac;
		find_root( fff, 0.0, 1.0, new_frac );
#endif
//		e1->set_frac(old_frac);
	//	printf( "NF: %e\n", new_frac);
		scf_update(com, omega, c_1, c_2, c1_x, c2_x, l1_x, *e1, *e2);
		solve_gravity(false);


	}
}

std::vector<real> scf_binary(real x, real y, real z, real) {
	std::vector<real> u(NF, real(0));
	static auto& params = initial_params();
	std::shared_ptr<bipolytropic_eos> this_eos;
	real rho, r, ei;
	if (x < params.l1_x) {
		r = std::sqrt(std::pow(x - params.c1_x, 2) + y * y + z * z);
		this_eos = std::dynamic_pointer_cast<bipolytropic_eos>(params.eos1);
	} else {
		r = std::sqrt(std::pow(x - params.c2_x, 2) + y * y + z * z);
		this_eos = std::dynamic_pointer_cast<bipolytropic_eos>(params.eos2);
	}
	rho = std::max(this_eos->density_at(r), rho_floor);
	ei = this_eos->pressure(rho) / (fgamma - 1.0);
	u[rho_i] = rho;
	u[spc_ac_i] = rho > this_eos->dE() ? (x > params.l1_x ? 0.0 : rho) : 0.0;
	u[spc_dc_i] = rho > this_eos->dE() ? (x > params.l1_x ? rho : 0.0) : 0.0;
	u[spc_ae_i] = rho <= this_eos->dE() ? (x > params.l1_x ? 0.0 : rho) : 0.0;
	u[spc_de_i] = rho <= this_eos->dE() ? (x > params.l1_x ? rho : 0.0) : 0.0;
	u[egas_i] = ei + 0.5 * (x * x + y * y) * params.omega * params.omega;
	u[sx_i] = -y * params.omega * rho;
	u[sy_i] = +x * params.omega * rho;
	u[sz_i] = 0.0;
	u[tau_i] = std::pow(ei, 1.0 / fgamma);
	return u;
}
