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

//#define MULTICORE

real R_a, R_b, R_c;
real w0 = 0.1;

HPX_PLAIN_ACTION(scf_binary_init, scf_binary_init_action);

const real c_r = 3.65375;
const real c_den = 5.99071;
const real M_acc = 1.0 / 1000.0;
const real M_don = 0.2 / 1000.0;
const real R_don = 0.06;
const real rho_floor = 1.0e-10;

const real G = 1.0;
const real n = 1.5;

struct global_vars_t {
	real omega, K;
	real C_acc, C_don, x_com;
	real x_acc, x_don;
	real x_drift, y_drift, z_drift;
	template<class Arc>
	void serialize(Arc& arc, const unsigned) {
		arc & K;
		arc & omega;
		arc & C_acc;
		arc & C_don;
		arc & x_com;
		arc & x_acc;
		arc & x_don;
		arc & x_drift;
		arc & y_drift;
		arc & z_drift;
	}
};

global_vars_t global;

real alpha_acc;
real alpha_don, R_acc, rho_acc, rho_don;

//HPX_REGISTER_PLAIN_ACTION(scf_binary_init_action);

__attribute__((constructor))
void scf_binary_init() {
	real q = M_don / M_acc;
	alpha_don = R_don / c_r;
	rho_don = c_den * 3.0 * M_don / std::pow(R_don, 3) / 4.0 / M_PI;
	global.K = 8.0 * M_PI * std::pow(alpha_don, 2) / 5.0 * std::pow(rho_don, 1.0 / 3.0);
	rho_acc = 288.0 * G * M_acc * M_acc * M_PI * c_den * c_den / 125.0 * std::pow(global.K * c_r * c_r, -3);
	alpha_acc = 5.0 * global.K * c_r / (4 * std::pow(6. * G * G * M_acc * M_PI * M_PI * c_den, 1. / 3.));
	R_acc = alpha_acc * c_r;
	double normalized_roche_volume = find_V(q);
	double roche_radius = std::pow(normalized_roche_volume / (4.0 / 3.0 * M_PI), 1.0 / 3.0);
	real a = R_don / roche_radius;
	global.x_acc = -q / (q + 1.0) * a;
	global.x_don = +1.0 / (q + 1.0) * a;
	real omega = std::sqrt(G * (M_acc + M_don) / pow(a, 3));

	R_a = global.x_don + R_don;
	R_b = global.x_don - R_don;

//	printf("rho    %e  |  %e\n", rho_acc, rho_don);
//	printf("M      %e  |  %e\n", M_acc, M_don);
//	printf("R      %e  |  %e\n", R_acc, R_don);
//	printf("x_com  %e  |  %e\n", global.x_acc, global.x_don);
//	printf("alpha  %e  |  %e\n", alpha_acc, alpha_don);
//	printf(" a     %e\n", a);
//	printf(" omega %e\n", omega);
//	exit(0);

}

std::vector<real> scf_binary(real x, real y, real z) {
	real r_acc = std::sqrt(std::pow(x - global.x_acc, 2) + y * y + z * z);
	real r_don = std::sqrt(std::pow(x - global.x_don, 2) + y * y + z * z);
	std::vector<real> u(NF, real(0));
	real alpha, rho0, r;
	real* rho_frac;
	if (r_acc < R_acc) {
		alpha = alpha_acc;
		rho0 = rho_acc;
		r = r_acc;
		rho_frac = &(u[acc_i]);
	} else if (r_don < R_don) {
		alpha = alpha_don;
		rho0 = rho_don;
		r = r_don;
		rho_frac = &(u[don_i]);
	} else {
		u[rho_i] = rho_floor;
		return u;
	}
	const real theta = lane_emden(r / alpha, r / alpha / 100.0);
	u[rho_i] = std::pow(theta, 1.5) * rho0;
	u[acc_i] = u[don_i] = ZERO;
	*rho_frac = u[rho_i];
	return u;
}

typedef typename node_server::scf_params_action scf_params_action_type;
HPX_REGISTER_ACTION(scf_params_action_type);

typedef typename node_server::scf_update_action scf_update_action_type;
HPX_REGISTER_ACTION(scf_update_action_type);

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
//	collect_hydro_boundaries();
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

scf_data_t grid::scf_params() {
	const real n = 1.5;

	real R, x, y, phi;
	scf_data_t scf_data;
	scf_data.x_com = ZERO;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				const integer iiig = gindex(i + G_BW - H_BW, j + G_BW - H_BW, k + G_BW - H_BW);
				x = X[XDIM][iii];
				y = X[YDIM][iii];
				real& rho = U[rho_i][iii];
				R = std::sqrt(std::pow(x - global.x_com, 2) + y * y);
				phi = G[phi_i][iiig];
				bool is_don = U[acc_i][iii] < U[don_i][iii];
				real ekin = HALF * std::pow(global.omega * R, 2) * rho + global.K * n * std::pow(rho, ONE + ONE / n);
				real epot = U[pot_i][iii];
				real phi_eff = G[phi_i][iiig] - HALF * global.omega * global.omega * R * R;
				scf_data.virial_num += (TWO * ekin + HALF * epot) * dx * dx * dx;
				scf_data.virial_den += std::abs(HALF * U[pot_i][iii]) * dx * dx * dx;
				scf_data.x_com += rho * dx * dx * dx * x;
				scf_data.sx_sum += U[sx_i][iii] * dx * dx * dx;
				scf_data.sy_sum += U[sy_i][iii] * dx * dx * dx;
				scf_data.sz_sum += U[sz_i][iii] * dx * dx * dx;
				if (U[acc_i][iii] > scf_data.rho_max_acc && !is_don) {
					scf_data.phi_eff_acc = phi_eff;
					scf_data.rho_max_acc = rho;
					scf_data.x_acc = x;
				}
				if (U[don_i][iii] > scf_data.rho_max_don && is_don) {
					scf_data.phi_eff_don = phi_eff;
					scf_data.rho_max_don = rho;
					scf_data.x_don = x;
				}
				if (is_don) {
					scf_data.m_don += rho * dx * dx * dx;
				} else {
					scf_data.m_acc += rho * dx * dx * dx;
				}
				if (std::abs(X[YDIM][iii]) < dx && std::abs(X[ZDIM][iii]) < dx) {
					if (std::abs(X[XDIM][iii] - R_a) <= HALF * dx) {
						scf_data.phi_a = phi;
					}
					if (global.x_acc < X[XDIM][iii] && global.x_don > X[XDIM][iii]) {
						if (phi_eff > scf_data.l1) {
							scf_data.l1 = phi_eff;
							scf_data.phi_l1 = G[phi_i][iiig];
							scf_data.lx = X[XDIM][iii];
						}
					}
				}
			}
		}
	}
	scf_data.x_com /= (scf_data.m_don + scf_data.m_acc);
	return scf_data;
}

real grid::scf_update(bool mom_only) {
	const real n = 1.5;
	real omega = global.omega;
	real C_acc = global.C_acc;
	real C_don = global.C_don;
	real res = ZERO;
	real R, x, y, phi, z;
	scf_data_t scf_data;
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer iii = hindex(i, j, k);
				const integer iiig = gindex(i + G_BW - H_BW, j + G_BW - H_BW, k + G_BW - H_BW);
				real& rho = U[rho_i][iii];

#ifndef MULTICORE
				auto eos_h2rho = [=](real h) {
					return std::pow(std::max(ZERO, h) / ((n + ONE) * global.K), n);
				};
				auto eos_rho2ene = [=](real rho) {
					return global.K * std::pow( rho, 1.0 + 1.0/n ) / (fgamma-1.0);
				};
#else
				const real rho_E = 0.5;
				const real rho_C = 1.0;
				real n_E = 1.5;
				real n_C = 3.0;
				auto eos_h2rho = [=](real h) {
					h = std::max(ZERO,h);
					real P0 = global.K * std::pow(rho_E, 1.0 + 1.0 / n_E);
					real H_E = (P0 / rho_E)*(1.0+n_E);
					real H_C = (P0 / rho_C)*(1.0+n_C);
					real r;
					if( h < H_E) {
						r = rho_E * std::pow(h / H_E, n_E);
					} else {
						r = rho_C * std::pow((h-H_E + H_C) / H_C, n_C);
					}
					return r;
				};
				auto eos_rho2ene = [=](real rho) {
					real p;
					real P0 = global.K * std::pow(rho_E, 1.0 + 1.0 / n_E);
					if( rho <= rho_E ) {
						p = P0 * std::pow( rho/rho_E, 1.0 + 1.0 / n_E);
					} else if( rho >= rho_C) {
						p = P0 * std::pow( rho/rho_C, 1.0 + 1.0 / n_C);
					} else {
						p = P0;
					}
					return p / (fgamma-1.0);
				};
#endif

				if (!mom_only) {
					x = X[XDIM][iii];
					y = X[YDIM][iii];
					z = X[ZDIM][iii];
					R = std::sqrt(std::pow(x - global.x_com, 2) + y * y);
					phi = G[phi_i][iiig];
					const real phi_eff = phi - HALF * omega * omega * R * R;
					real new_rho;
					real* rho_frac;
					real gx = G[gx_i][iiig] + (x - global.x_com) * omega * omega;
					real gy = G[gy_i][iiig] + y * omega * omega;
					real gz = G[gz_i][iiig];
					bool is_don = std::abs(x - global.x_don) < std::abs(x - global.x_acc);
					if (!is_don) {
						x -= global.x_acc;
						if (x * gx + y * gy + z * gz < ZERO) {
							new_rho = eos_h2rho(C_acc - phi_eff);
						} else {
							new_rho = ZERO;
						}
						rho_frac = &(U[acc_i][iii]);
					} else {
						x -= global.x_don;
						if (x * gx + y * gy + z * gz < ZERO) {
							new_rho = eos_h2rho(C_don - phi_eff);
						} else {
							new_rho = ZERO;
						}
						rho_frac = &(U[don_i][iii]);
					}
					new_rho = std::max(new_rho, rho_floor);
					res += std::abs((new_rho - rho) * dx * dx * dx);
					rho = (ONE - w0) * rho + w0 * new_rho;
					U[don_i][iii] = U[acc_i][iii] = ZERO;
					*rho_frac = rho;

					real e_int = eos_rho2ene(rho);
					real e_kin = HALF * rho * std::pow(global.omega * R, 2);
					U[egas_i][iii] = e_int + e_kin;
					U[tau_i][iii] = std::pow(e_int, ONE / (ONE + ONE / n));
					U[sx_i][iii] = -y * global.omega * rho;
					U[sy_i][iii] = +(X[XDIM][iii] - global.x_com) * global.omega * rho;
				} else {
					U[sx_i][iii] -= global.x_drift * rho;
					U[sy_i][iii] -= global.y_drift * rho;
				}
				U[sz_i][iii] = ZERO;
				U[zx_i][iii] = ZERO;
				U[zy_i][iii] = ZERO;
				U[zz_i][iii] = dx * dx * global.omega * rho / 6.0;
			}
		}
	}
	return res;
}

void set_global_vars_local(const global_vars_t& gv) {
	global = gv;
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

void node_server::run_scf() {
	real MAX = 0.25;
	real MIN = 1.0 / 1000.0;
	w0 = MIN;
	real res = ONE, old_res = ONE;
	set_global_vars(global);
	solve_gravity(false);
	scf_data_t data = scf_params();
	bool done = false;
	for (integer i = 0; i != 1001; ++i) {
		const real new_omega = std::sqrt(TWO * (data.phi_a - data.phi_l1) / (R_a * R_a - data.lx * data.lx));
		data.phi_eff_acc += HALF * global.omega * global.omega * data.x_acc * data.x_acc;
		global.omega = new_omega;
		data.phi_eff_acc -= HALF * global.omega * global.omega * data.x_acc * data.x_acc;
		R_b = HALF * (data.lx + R_b);
		global.K = (data.l1 - data.phi_eff_don) / (n + 1.0) * std::pow(rho_don, -1.0 / n);
		global.C_don = data.l1;
		global.C_acc = data.phi_eff_acc + global.K * (2.5) * std::pow(rho_acc, 1.0 / 1.5);
		global.x_acc = data.x_acc;
		global.x_don = data.x_don;
		global.x_com = data.x_com;
		set_global_vars(global);
		real virial_error = data.virial_num / data.virial_den;
		if (i % 50 == 0 || done) {
			if (done) {
				save_to_file(std::string("X.chk"));
				//		SYSTEM(std::string("./hpx X.chk X.") + std::to_string(i) + std::string(".silo"));
			}
			regrid(me.get_gid(), false);
			printf("\n   s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s\n", "rho_max_acc",
					"rho_max_don", "omega", "X_com", "C_acc", "C_don", "virial", "xdrif", "ydrift", "zdrift", "q", "w0",
					"res");
			FILE* fp = fopen("scf.dat", "at");
			fprintf(fp, "\n   s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s\n", "rho_max_acc",
					"rho_max_don", "omega", "X_com", "C_acc", "C_don", "virial", "xdrif", "ydrift", "zdrift", "q", "w0",
					"res");
			fclose(fp);
		}
		res = scf_update(false);
		data = scf_params();
		res /= data.m_don + data.m_acc;
		global.x_drift = data.sx_sum / (data.m_don + data.m_acc);
		global.y_drift = data.sy_sum / (data.m_don + data.m_acc);
		global.z_drift = data.sz_sum / (data.m_don + data.m_acc);
		set_global_vars(global);
		scf_update(true);
		solve_gravity(false);
		data = scf_params();
		global.x_drift = data.sx_sum / (data.m_don + data.m_acc);
		global.y_drift = data.sy_sum / (data.m_don + data.m_acc);
		global.z_drift = data.sz_sum / (data.m_don + data.m_acc);
		printf("%3i %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e\n", int(i), data.rho_max_acc,
				data.rho_max_don, global.omega, global.x_com, global.C_acc, global.C_don, virial_error, global.x_drift,
				global.y_drift, global.z_drift, data.m_don / data.m_acc, w0, res);
		FILE* fp = fopen("scf.dat", "at");
		fprintf(fp, "%3i %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e\n", int(i), data.rho_max_acc,
				data.rho_max_don, global.omega, global.x_com, global.C_acc, global.C_don, virial_error, global.x_drift,
				global.y_drift, global.z_drift, data.m_don / data.m_acc, w0, res);
		fclose(fp);
		if (res < 1.0e-3) {
			if (done) {
				break;
			}
			if (i > 50) {
				done = true;
			}
		} else {
			done = false;
		}
		real dw0p = std::min(exp((w0 - MAX) / MAX) / 100.0, (MAX - w0) / 4.0);
		real dw0m = std::min(exp((MIN - w0) / MAX) / 100.0, (w0 - MIN) / 4.0);
		//	printf( "%e %e %e\n", w0, dw0p, dw0m);
		if (res < old_res) {
			w0 += dw0p;
		} else if (res > old_res) {
			w0 -= dw0m;
		}
		w0 = std::max(MIN, std::min(MAX, w0));
		old_res = res;
	}
}
