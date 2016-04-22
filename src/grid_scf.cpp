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

#define BIBI

real mass_ratio = 0.7;
const real donor_radius = 0.15;
const real rho_floor = 1.0e-10;
const real bibi_mu = 0.5;
const real bibi_don_core_frac = 2.0 * 0.2374;
const real bibi_acc_core_frac = 2.0 * 0.2377;
const real rho_acc_max = 1.0;
const real rho_don_max = 0.836;

real RA, RD;


real xpt_A = 0.950980392156863;
real xpt_B = 0.182352941176471;
real xpt_C = -0.166666666666667;
real MA = 1.689579196657952E-002;
real MD = 1.689579196657952E-002;



//HPX_PLAIN_ACTION(scf_binary_init, scf_binary_init_action);

const real G = ONE;

struct global_vars_t {
	real xcom;
	real omega;
	real donor_center;
	real accretor_center;
	real donor_con;
	real accretor_con;
	real l1_x;
	real accretor_mass;
	real donor_mass;
	real donor_central_enthalpy;
	real accretor_central_enthalpy;
	template<class Arc>
	void serialize(Arc& arc, const unsigned) {
		arc & xcom;
		arc & omega;
		arc & donor_center;
		arc & accretor_center;
		arc & donor_con;
		arc & accretor_con;
		arc & l1_x;
		arc & accretor_mass;
		arc & donor_mass;
		arc & donor_central_enthalpy;
		arc & accretor_central_enthalpy;
	}
};

static std::once_flag init_flag;

global_vars_t global;

typedef typename node_server::scf_params_action scf_params_action_type;
HPX_REGISTER_ACTION (scf_params_action_type);

typedef typename node_server::scf_update_action scf_update_action_type;
HPX_REGISTER_ACTION (scf_update_action_type);

hpx::future<scf_data_t> node_client::scf_params() const {
	return hpx::async<typename node_server::scf_params_action>(get_gid());
}

hpx::future<real> node_client::scf_update(bool b) const {
	return hpx::async<typename node_server::scf_update_action>(get_gid(), b);
}

real node_server::scf_update(bool mom_only) {
	std::vector<hpx::future<real>> futs;
	real res;
	if (is_refined) {
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.scf_update(mom_only));
		}
		res = ZERO;
	} else {
		res = grid_ptr->scf_update(mom_only);
	}
	exchange_interlevel_hydro_data();
	collect_hydro_boundaries();
	for (auto&& fut : futs) {
		res += fut.get();
	}
	return res;
}

scf_data_t node_server::scf_params() {
	scf_data_t data;
	std::vector<hpx::future<scf_data_t>> futs;
	if (is_refined) {
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.scf_params());
		}
	}

	if (is_refined) {
		for (auto& fut : futs) {
			data.accumulate(fut.get());
		}
	} else {
		data = grid_ptr->scf_params();
	}
	return data;
}

void static_initialize();

static eos& donor_eos() {
	static_initialize();
#ifndef BIBI
	static polytropic_eos a(MD, RD);
#else
	static bipolytropic_eos a(MD, RD, 3.0, 1.5, bibi_don_core_frac, bibi_don_core_frac * bibi_mu);
#endif
	return a;
}
static eos& accretor_eos() {
	static_initialize();
#ifndef BIBI
	static polytropic_eos a(MA, RA);
#else
	static bipolytropic_eos a(MA, RA, 3.0, 1.5, bibi_acc_core_frac, bibi_acc_core_frac * bibi_mu);
#endif
	return a;
}

scf_data_t grid::scf_params() {

	scf_data_t data;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer D = G_BW - H_BW;
				const integer iiih = hindex(i, j, k);
				const integer iiig = gindex(i + D, j + D, k + D);
				const real x = X[XDIM][iiih];
				const real y = X[YDIM][iiih];
				const real z = X[ZDIM][iiih];
				const real R = std::sqrt(std::pow(x - global.xcom, 2) + y * y);
				const real& rho = U[rho_i][iiih];
				const real phi = G[phi_i][iiig];
				const real phi_eff = phi - 0.5 * std::pow(global.omega * R, 2);
				//	const bool acc_frac = (U[frac0_i][iiih] > U[frac1_i][iiih]) && (rho > 10.0 * rho_floor);
				//	const bool don_frac = (U[frac0_i][iiih] < U[frac1_i][iiih]) && (rho > 10.0 * rho_floor);
				const bool don_frac = X[XDIM][iiih] >= xpt_B && (rho > 10.0 * rho_floor);
				const bool acc_frac = X[XDIM][iiih] <= xpt_C && (rho > 10.0 * rho_floor);
				const real dm = rho * dx * dx * dx;
				const real dV = dx * dx * dx;
				real p;
			//	bool instar = don_frac || acc_frac;
				p = 0.0;
			//	real h = 0.0;
				data.m += dm;
				data.m_x += dm * x;
				if (don_frac) {
					data.donor_mass += dm;
					if (std::abs(y) < dx && std::abs(z) < dx) {
						const real h = donor_eos().density_to_enthalpy(rho);
						if (h > data.donor_central_enthalpy) {
							data.donor_phi_min = phi;
							data.donor_x = x;
							data.donor_central_enthalpy = h;
							data.donor_central_density = rho;
						}
					}
			//		h = donor_eos().density_to_enthalpy(rho);
					p = donor_eos().pressure(rho);
//					p = ZERO;
					data.donor_phi_max = std::max(data.donor_phi_max, phi_eff);
				} else if (acc_frac) {
					data.accretor_mass += dm;
					if (std::abs(y) < dx && std::abs(z) < dx) {
						const real h = accretor_eos().density_to_enthalpy(rho);
						if (h > data.accretor_central_enthalpy) {
							data.accretor_phi_min = phi;
							data.accretor_x = x;
							data.accretor_central_enthalpy = h;
							data.accretor_central_density = rho;
						}
					}
		//			h = accretor_eos().density_to_enthalpy(rho);
					p = accretor_eos().pressure(rho);
					data.accretor_phi_max = std::max(data.accretor_phi_max, phi_eff);
				} else {
			//		h = 0.0;
				}
				const real ekin = 3.0 * p + std::pow(R * global.omega, 2) * rho;
				const real epot = 0.5 * phi * rho;
				data.virial_sum += (ekin + epot) * dV;
				data.virial_norm += (std::abs(ekin) + std::abs(epot)) * dV;
				if (y > 0.0 && z > 0.0) {
					if (std::abs(y) < dx && std::abs(z) < dx) {
						const real a0 = x + dx / 2.0 - xpt_A;
						const real b0 = x + dx / 2.0 - xpt_B;
						const real c0 = x + dx / 2.0 - xpt_C;
						const real phi_p = U[pot_i][iiih + H_DNX] / U[rho_i][iiih + H_DNX];
						const real phi0 = 0.5 * (phi_p + phi);
						if (std::abs(a0) < dx / 2.0) {
							data.phiA = phi0;
							data.xA = x + dx / 2.0;
						}
						if (std::abs(b0) < dx / 2.0) {
							data.phiB = phi0;
							data.xB = x + dx / 2.0;
						}
						if (std::abs(c0) < dx / 2.0) {
							data.phiC = phi0;
							data.xC = x + dx / 2.0;
						}
					}
				}

				if (std::abs(y) < dx && std::abs(z) < dx) {

					if (x > global.accretor_center && x < global.donor_center) {
						if (phi_eff > data.l1_phi) {
							data.l1_phi = phi_eff;
							data.l1_x = x;
						}
					}
				}
			}
		}
	}
	return data;
}

void static_initialize() {
	std::call_once(init_flag, []() {
		const real dx = 1.0/255.0;
//		xpt_A =  0.775590551181102;
//		xpt_B = 0.145669291338583;
//		xpt_C = -0.145669291338583;
			//	MA = 7.868152748811249E-003;
			//	MD = 5.485139898262405E-003;


			global.donor_center = (xpt_A - xpt_B)/2.0+xpt_B;
			global.accretor_center = -((xpt_A + xpt_C)/2.0-xpt_C);
			RD = std::abs(global.donor_center - xpt_B);
			RA = std::abs(global.accretor_center - xpt_C);
			mass_ratio = MD / MA;
//			global.donor_center = (xpt_A + xpt_B) / 2.0;
			real a = (global.donor_center)*(mass_ratio + 1.0);
//			global.accretor_center = -mass_ratio / (mass_ratio + 1.0) * a;
//			RD = (xpt_A - xpt_B)/2.0;
//			RA = xpt_C -global.accretor_center;
			//	printf( "%e %e %e\n", RA, RD, a);
			global.omega = std::sqrt(G * (MA + MD) / pow(a, 3));
			global.accretor_mass = MA;
			global.donor_mass = MD;

			//		printf( "%e %e %e\n", xpt_A, xpt_B, xpt_C);
		});
}

const real w0 = 1.0 / 2.0;
real grid::scf_update(bool mom_only) {
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer D = G_BW - H_BW;
				const integer iiih = hindex(i, j, k);
				const integer iiig = gindex(i + D, j + D, k + D);
				const real x = X[XDIM][iiih];
				const real y = X[YDIM][iiih];
				const real z = X[ZDIM][iiih];
				const real R = std::sqrt(std::pow(x - global.xcom, 2) + y * y);
				const real& rho = U[rho_i][iiih];
				const real phi_eff = G[phi_i][iiig] - 0.5 * std::pow(global.omega * R, 2);
				const real fx = G[gx_i][iiig] + (x - global.xcom) * std::pow(global.omega, 2);
				const real fy = G[gy_i][iiig] + y * std::pow(global.omega, 2);
				const real fz = G[gz_i][iiig];
				//	const bool acc_frac = (U[frac0_i][iiih] > U[frac1_i][iiih]) && (rho > 10.0 * rho_floor);
				//	const bool don_frac = (U[frac0_i][iiih] < U[frac1_i][iiih]) && (rho > 10.0 * rho_floor);
				const eos* eos_ptr;
				real* frac_ptr;
				real* not_frac_ptr;
				real phi0;
				bool star = true;
				if (x <= xpt_C && fx * (x - global.accretor_center) + fy * y + fz * z <= 0) {
					phi0 = global.accretor_con;
					eos_ptr = &accretor_eos();
#ifndef BIBI
					frac_ptr = &U[frac0_i][iiih];
					not_frac_ptr = &U[frac1_i][iiih];
#else
					if (rho >= accretor_eos().d0() * bibi_acc_core_frac) {
						frac_ptr = &U[frac0_i][iiih];
						not_frac_ptr = &U[frac1_i][iiih];
					} else {
						frac_ptr = &U[frac1_i][iiih];
						not_frac_ptr = &U[frac0_i][iiih];
					}
#endif
				} else if (x >= xpt_B && x <= xpt_A && fx * (x - global.donor_center) + fy * y + fz * z <= 0) {
					phi0 = global.donor_con;
					eos_ptr = &donor_eos();
#ifndef BIBI
					frac_ptr = &U[frac1_i][iiih];
					not_frac_ptr = &U[frac0_i][iiih];
#else
					if (rho >= donor_eos().d0() * bibi_don_core_frac) {
						frac_ptr = &U[frac0_i][iiih];
						not_frac_ptr = &U[frac1_i][iiih];
					} else {
						frac_ptr = &U[frac1_i][iiih];
						not_frac_ptr = &U[frac0_i][iiih];
					}
#endif
				} else {
					star = false;
				}
				real new_rho, eint;
				if (star) {
					const real h = std::max(phi0 - phi_eff, 0.0);
					const real d = std::max(eos_ptr->enthalpy_to_density(h), rho_floor);
					new_rho = rho * (1.0 - w0) + w0 * d;
					*frac_ptr = new_rho - rho_floor / 2.0;
					*not_frac_ptr = rho_floor / 2.0;
					eint = eos_ptr->pressure(rho) / (fgamma - 1.0);
				} else {
					new_rho = rho * (1.0 - w0) + rho_floor * w0;
					U[frac0_i][iiih] = U[frac1_i][iiih] = rho_floor / 2.0;
					eint = 1.0e-10;
				}
				U[rho_i][iiih] = new_rho;
				U[sx_i][iiih] = -global.omega * y * rho;
				U[sy_i][iiih] = +global.omega * (x - global.xcom) * rho;
				U[sz_i][iiih] = 0.0;
				U[egas_i][iiih] = eint + std::pow(R * global.omega, 2) * rho / 2.0;
				U[tau_i][iiih] = std::pow(eint, 1.0 / fgamma);
				U[zx_i][iiih] = 0.0;
				U[zy_i][iiih] = 0.0;
				U[zz_i][iiih] = dx * dx * global.omega * rho / 6.0;
			}
		}
	}

	return 0.0;
}

void set_global_vars_local(const global_vars_t& gv) {
	global = gv;
	accretor_eos().set_h0(gv.accretor_central_enthalpy);
	donor_eos().set_h0(gv.donor_central_enthalpy);
//	accretor_eos().M0 *= MA / gv.accretor_mass;
//	donor_eos().M0 *= MD / gv.donor_mass;
}

HPX_PLAIN_ACTION(set_global_vars_local, set_global_vars_action);

void set_global_vars(const global_vars_t& gv) {
	auto tmp = gv;
	auto localities = hpx::find_all_localities();
	std::list<hpx::future<void>> futs;
	for (auto& locality : localities) {
		futs.push_back(hpx::async<set_global_vars_action>(locality, tmp));
	}
	for (auto&& fut : futs) {
		fut.get();
	}
}

std::vector<real> scf_binary(real x, real y, real z, real) {
	std::vector<real> u(NF, real(0));
	donor_eos().set_d0(rho_don_max);
	accretor_eos().set_d0(rho_acc_max);
	const real ra = std::sqrt(std::pow(x - global.accretor_center, 2) + y * y + z * z);
	const real rd = std::sqrt(std::pow(x - global.donor_center, 2) + y * y + z * z);
	//	const real da = accretor_eos().density_at(ra);
	//	const real dd = donor_eos().density_at(rd);
	const real da = rho_acc_max * exp(-std::pow(ra / RA, 2) * 13.815510558);
	const real dd = rho_don_max * exp(-std::pow(rd / RD, 2) * 13.815510558);
//1	const real dd = donor_eos().density_at(rd);
	u[rho_i] = std::max(da + dd, rho_floor);
#ifndef BIBI
	u[frac0_i] = std::max(da, rho_floor / 2.0);
	u[frac1_i] = std::max(dd, rho_floor / 2.0);
#else
	real thresh;
	if (da > dd) {
		thresh = accretor_eos().d0() * bibi_acc_core_frac;
	} else {
		thresh = donor_eos().d0() * bibi_don_core_frac;
	}
	const real rho = da + dd;
//	printf( "%e\n", thresh);
	if (rho > thresh) {
		u[frac0_i] = std::max(rho, rho_floor / 2.0);
		u[frac1_i] = rho_floor / 2.0;
	} else {
		u[frac1_i] = std::max(rho, rho_floor / 2.0);
		u[frac0_i] = rho_floor / 2.0;
	}

#endif
	return u;
}

void node_server::run_scf() {

	solve_gravity(false);
	auto d = scf_params();
	global.accretor_central_enthalpy = accretor_eos().h0();
	global.donor_central_enthalpy = donor_eos().h0();
	global.donor_center = d.donor_x;
	global.accretor_center = d.accretor_x;
	global.l1_x = d.l1_x;
	set_global_vars(global);
//	printf("%e\n", global.omega);
	for (integer i = 0; i != 401; ++i) {
		if (i % 100 == 0 && i != 0) {
			save_to_file(std::string("X.chk"));
			output("X.scf.silo");
			//	SYSTEM(
			//			std::string("./octotiger -problem=dwd -restart=X.chk -output=X.") + std::to_string(i)
			//					+ std::string(".silo"));
		}
		solve_gravity(false);
		auto d = scf_params();
		global.donor_center = d.donor_x;
		global.accretor_center = d.accretor_x;
		global.l1_x = d.l1_x;
		const real a = xpt_A - global.xcom;
		const real b = xpt_B - global.xcom;
		const real c = xpt_C - global.xcom;
		global.omega = std::sqrt(2.0 * (d.phiA - d.phiB) / (a * a - b * b));
		//	const real dcb = d.phiB - 0.5 * std::pow(global.omega * xpt_B, 2);
		global.donor_con = d.phiA - 0.5 * std::pow(global.omega * a, 2);
		global.accretor_con = d.phiC - 0.5 * std::pow(global.omega * c, 2);
		global.xcom = d.m_x / d.m;
		//	printf( "%e %e\n", dca, dcb);
		if (i % 25 == 0) {
			regrid(me.get_gid(), false);
			printf("      %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n", "xcom",
					"q", "omega", "C_A", "C_D", "M_A", "M_D", "xA", "xB", "xC", "l1_x", "virial", "rho_A", "rho_D",
					"HA", "HD");
		}
		//	printf( "%e %e\n", accretor_eos().d0(), donor_eos().d0());
		printf("%5i %12e %12e %12e %12e %12e %12e %12e %12e %12e %12e %12e %12e %12e %12e %12e %12e  \n", int(i),
				global.xcom, d.donor_mass / d.accretor_mass, global.omega, global.accretor_con, global.donor_con,
				d.accretor_mass, d.donor_mass, d.xA, d.xB, d.xC, d.l1_x, d.virial_sum / d.virial_norm,
				d.accretor_central_density, d.donor_central_density,
				accretor_eos().density_to_enthalpy(d.accretor_central_density),
				accretor_eos().density_to_enthalpy(d.accretor_central_density));

		d.accretor_phi_min -= 0.5 * std::pow(global.omega * (d.accretor_x - global.xcom), 2);
		d.donor_phi_min -= 0.5 * std::pow(global.omega * (d.donor_x - global.xcom), 2);
		global.accretor_central_enthalpy = global.accretor_con - d.accretor_phi_min;
		global.donor_central_enthalpy = global.donor_con - d.donor_phi_min;

		set_global_vars(global);
		scf_update(false);
	}
}
