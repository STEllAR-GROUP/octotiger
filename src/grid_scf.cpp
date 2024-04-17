//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/defs.hpp"
#include "octotiger/particles.hpp"
#include "octotiger/eos.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/grid_scf.hpp"
#include "octotiger/lane_emden.hpp"
#include "octotiger/node_client.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/profiler.hpp"
#include "octotiger/real.hpp"
#include "octotiger/util.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>

constexpr integer spc_ac_i = spc_i;
constexpr integer spc_ae_i = spc_i + 1;
constexpr integer spc_dc_i = spc_i + 2;
constexpr integer spc_de_i = spc_i + 3;
constexpr integer spc_vac_i = spc_i + 4;

// w0 = speed of convergence. Adjust lower if nan
const real w0init = 1.0 / 2.0;
const real w0max = 0.5;
const real iter2max = 25.0;
const int itermax = 256;
real w0 = w0init;

namespace scf_options {

/********** V1309 SCO ****************/
static real async1 = -0.0e-2;
static real async2 = -0.0e-2;
static bool equal_struct_eos = false; // If true, EOS of accretor will be set to that of donor
static real M1 = 1.54; // Mass of primary
static real M2 = 0.17; // Mass of secondaries
static real nc1 = 5.0; // Primary core polytropic index
static real nc2 = 5.0; // Secondary core polytropic index
static real ne1 = 3.0; // Primary envelope polytropic index // Ignored if equal_struct_eos=true
static real ne2 = 1.5; // Secondary envelope polytropic index
static real mu1 = 2.1598; // Primary ratio of molecular weights // Ignored if equal_struct_eos=true
static real mu2 = 2.1598; // Primary ratio of molecular weights
static real a = 6.36; // approx. orbital sep
static real core_frac1 = 1.0 / 10.0; // Desired core fraction of primary // Ignored if equal_struct_eos=true
static real core_frac2 = 2.0 / 3.0; // Desired core fraction of secondary - IGNORED FOR CONTACT binaries
static real fill1 = 0.99; // 1d Roche fill factor for primary (ignored if contact fill is > 0.0) //  - IGNORED FOR CONTACT binaries  // Ignored if equal_struct_eos=true
static real fill2 = 0.99; // 1d Roche fill factor for secondary (ignored if contact fill is > 0.0) // - IGNORED FOR CONTACT binaries
static real contact_fill = 0.1; //  Degree of contact - IGNORED FOR NON-CONTACT binaries // SET to ZERO for equal_struct_eos=true
static std::string filename1 = "";
static std::string filename2 = "";

//static real async1 = -0.0e-2;
//static real async2 = -0.0e-2;
//static bool equal_struct_eos = true; // If true, EOS of accretor will be set to that of donor
//static real M1 = 1.00; // Mass of primary
//static real M2 = 0.70; // Mass of secondaries
//static real nc1 = 1.5; // Primary core polytropic index
//static real nc2 = 1.5; // Secondary core polytropic index
//static real ne1 = 1.5; // Primary envelope polytropic index // Ignored if equal_struct_eos=true
//static real ne2 = 1.5; // Secondary envelope polytropic index
//static real mu1 = 1; // Primary ratio of molecular weights // Ignored if equal_struct_eos=true
//static real mu2 = 1; // Primary ratio of molecular weights
//static real a = 1.0; // approx. orbital sep
//static real core_frac1 = 1.0 / 10.0; // Desired core fraction of primary // Ignored if equal_struct_eos=true
//static real core_frac2 = 2.0 / 3.0; // Desired core fraction of secondary - IGNORED FOR CONTACT binaries
//static real fill1 = 0.99; // 1d Roche fill factor for primary (ignored if contact fill is > 0.0) //  - IGNORED FOR CONTACT binaries  // Ignored if equal_struct_eos=true
//static real fill2 = 0.99; // 1d Roche fill factor for secondary (ignored if contact fill is > 0.0) // - IGNORED FOR CONTACT binaries
//static real contact_fill = 0.0; //  Degree of contact - IGNORED FOR NON-CONTACT binaries // SET to ZERO for equal_struct_eos=true

//namespace scf_options {
//static real async1 = -0.0e-2;
//static real async2 = -0.0e-2;
//static bool equal_struct_eos = true; // If true, EOS of accretor will be set to that of donor
//static real M1 = 0.6; // Mass of primary
//static real M2 = 0.3; // Mass of sfecondaries
//static real nc1 = 2.5; // Primary core polytropic index
//static real nc2 = 1.5; // Secondary core polytropic index
//static real ne1 = 1.5; // Primary envelope polytropic index // Ignored if equal_struct_eos=true
//static real ne2 = 1.5; // Secondary envelope polytropic index
//static real mu1 = 1.0; // Primary ratio of molecular weights // Ignored if equal_struct_eos=true
//static real mu2 = 1.0; // Primary ratio of molecular weights
//static real a = 1.00; // approx. orbital sep
//static real core_frac1 = 0.9; // Desired core fraction of primary // Ignored if equal_struct_eos=true
//static real core_frac2 = 0.9; // Desired core fraction of secondary - IGNORED FOR CONTACT binaries
//static real fill1 = 1.0; // 1d Roche fill factor for primary (ignored if contact fill is > 0.0) //  - IGNORED FOR CONTACT binaries  // Ignored if equal_struct_eos=true
//static real fill2 = 1.0; // 1d Roche fill factor for secondary (ignored if contact fill is > 0.0) // - IGNORED FOR CONTACT binaries
//static real contact_fill = 0.00; //  Degree of contact - IGNORED FOR NON-CONTACT binaries // SET to ZERO for equal_struct_eos=true

#define READ_LINE(s) 		\
	else if( cmp(ptr,#s) ) { \
		s = read_float(ptr); \
		if( hpx::get_locality_id() == 0 ) printf( #s "= %e\n", double(s)); \
	}

#define READ_LINE_AS_STRING(s)            \
        else if( cmp(ptr,#s) ) { \
                s = read_string(ptr); \
                if( hpx::get_locality_id() == 0 ) printf( #s "= %s\n", s.c_str()); \
        }

void read_option_file() {
	FILE *fp = fopen("scf.init", "rt");
	if (fp != nullptr) {
		if (hpx::get_locality_id() == 0)
			printf("SCF option file found\n");
		const auto cmp = [](char *ptr, const char *str) {
			return strncmp(ptr, str, strlen(str)) == 0;
		};
		const auto read_float = [](char *ptr) {
			while (*ptr != '\0' && *ptr != '=') {
				++ptr;
			}
			if (*ptr == '=') {
				++ptr;
			}
			return std::stof(ptr);
		};
                const auto read_string = [](char *ptr) {
                        while (*ptr != '\0' && *ptr != '=') {
                                ++ptr;
                        }
                        if (*ptr == '=') {
                                ++ptr;
                        }
			auto tmp = std::string(ptr);
			tmp.resize(tmp.find_last_not_of(" \t\f\v\n\r") + 1);
                        return tmp;
                };
		while (!feof(fp)) {
			char buffer[1024];
			if (fgets(buffer, 1023, fp) != nullptr) {
				char *ptr = buffer;
				while (isspace(*ptr) && *ptr != '\0') {
					++ptr;
				}
				if (isspace(*ptr)) {
					++ptr;
				}
				//if (false) {
				//}
				READ_LINE(equal_struct_eos)
				READ_LINE(contact_fill)
				READ_LINE(core_frac1)
				READ_LINE(core_frac2)
				READ_LINE(async1)
				READ_LINE(async2)
				READ_LINE(fill1)
				READ_LINE(fill2)
				READ_LINE(nc1)
				READ_LINE(nc2)
				READ_LINE(ne1)
				READ_LINE(ne2)
				READ_LINE(mu1)
				READ_LINE(mu2)
				READ_LINE(M1)
				READ_LINE(M2)
				READ_LINE(a) 
				READ_LINE_AS_STRING(filename1)
				READ_LINE_AS_STRING(filename2) else if (strlen(ptr)) {
					if (hpx::get_locality_id() == 0)
						printf("unknown SCF option - %s\n", buffer);
				}
			}
		}
		fclose(fp);
	} else {
		if (hpx::get_locality_id() == 0)
			printf("SCF option file \"scf.init\" not found - using defaults\n");
	}

}

}

future<void> node_client::rho_move(real x) const {
	return hpx::async<typename node_server::rho_move_action>(get_unmanaged_gid(), x);
}

void node_server::rho_move(real x) {
	std::array<future<void>, NCHILD> futs;
	if (is_refined) {
		integer index = 0;
		for (auto &child : children) {
			futs[index++] = child.rho_move(x);
		}
	}
	const auto dx_min = 2.0 * opts().xscale / INX / (1 << opts().max_level);
	if (x >= 0.0) {
		grid_ptr->rho_move(std::min(w0 * x / 2.0, dx_min / 10.0));
	} else {
		grid_ptr->rho_move(std::max(w0 * x / 2.0, -dx_min / 10.0));
	}
	all_hydro_bounds();
	if (is_refined) {
		for (auto &f : futs) {
			GET(f);
		}
//		wait_all_and_propagate_exceptions(futs);
	}
}

using scf_update_action_type = typename node_server::scf_update_action;
HPX_REGISTER_ACTION (scf_update_action_type);

using rho_mult_action_type = typename node_server::rho_mult_action;
HPX_REGISTER_ACTION (rho_mult_action_type);

future<void> node_client::rho_mult(real f0, real f1, real f_p0, real f_p1) const {
	return hpx::async<typename node_server::rho_mult_action>(get_unmanaged_gid(), f0, f1, f_p0, f_p1);
}

future<real> node_client::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, struct_eos e1, struct_eos e2) const {
	return hpx::async<typename node_server::scf_update_action>(get_unmanaged_gid(), com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
}

void node_server::rho_mult(real f0, real f1, real f_p0, real f_p1) {
	std::array<future<void>, NCHILD> futs;
	if (is_refined) {
		integer index = 0;
		for (auto &child : children) {
			futs[index++] = child.rho_mult(f0, f1, f_p0, f_p1);
		}
	}
	grid_ptr->rho_mult(f0, f1, f_p0, f_p1);
	all_hydro_bounds();
	if (is_refined) {
		for (auto &f : futs) {
			GET(f);
		}

//		wait_all_and_propagate_exceptions(futs);
	}
}

real node_server::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, struct_eos e1, struct_eos e2) {
	grid::set_omega(omega);
	std::array<future<real>, NCHILD> futs;
	real res;
	if (is_refined) {
		integer index = 0;
		for (auto &child : children) {
			futs[index++] = child.scf_update(com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
		}
		res = ZERO;
	} else {
		res = grid_ptr->scf_update(com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
	}
	all_hydro_bounds();
	if (is_refined) {
		res = std::accumulate(futs.begin(), futs.end(), res, [](real res, future<real> &f) {
			return res + f.get();
		});
	}
	current_time += 1.0e-100;
	return res;
}

struct scf_parameters {
	real R1;
	real R2;
	real omega;
	real G;
	real q;
	std::shared_ptr<struct_eos> struct_eos1;
	std::shared_ptr<struct_eos> struct_eos2;
	real l1_x;
	real c1_x;
	real c2_x;
	scf_parameters() {
		if (scf_options::equal_struct_eos) {
			scf_options::contact_fill = 0.0;
		}
		const real M1 = scf_options::M1;
		const real M2 = scf_options::M2;
		const std::string filename1 = scf_options::filename1;
		const std::string filename2 = scf_options::filename2;
		const real fill1 = scf_options::fill1;
		const real contact = scf_options::contact_fill;
		const real a = scf_options::a;
		G = 1.0;
		const real c = 4.0 * M_PI / 3.0;
		q = M2 / M1;
		c1_x = -a * M2 / (M1 + M2);
		c2_x = +a * M1 / (M1 + M2);
		l1_x = a * (0.5 - 0.227 * log10(q)) + c1_x;
		omega = SQRT((G * (M1 + M2)) / (a * a * a));
		const real fill2 = scf_options::fill2;
		const real V1 = find_V(M1 / M2) * cube(a);
		const real V2 = find_V(M2 / M1) * cube(a);
		R1 = POWER(V1 / c, 1.0 / 3.0) * POWER(fill1, 5);
		R2 = POWER(V2 / c, 1.0 / 3.0) * POWER(fill2, 5);
		if (opts().eos == WD) {
			//	printf( "!\n");
			struct_eos2 = std::make_shared<struct_eos>(scf_options::M2, R2);
			struct_eos1 = std::make_shared<struct_eos>(scf_options::M1, *struct_eos2);
		} else {
			if (scf_options::equal_struct_eos) {
				struct_eos2 = std::make_shared<struct_eos>(scf_options::M2, R2, scf_options::nc2, scf_options::ne2, scf_options::core_frac2, scf_options::mu2);
				struct_eos1 = std::make_shared<struct_eos>(scf_options::M1, scf_options::nc1, *struct_eos2);
			} else {
				if (filename1 == "") {
					struct_eos1 = std::make_shared<struct_eos>(scf_options::M1, R1, scf_options::nc1, scf_options::ne1, scf_options::core_frac1, scf_options::mu1);
				} else {
					struct_eos1 = std::make_shared<struct_eos>(scf_options::M1, scf_options::core_frac1, filename1);
				}
				if (contact > 0.0 && !opts().v1309) {
					struct_eos2 = std::make_shared<struct_eos>(scf_options::M2, R2, scf_options::nc2, scf_options::ne2, scf_options::mu2, *struct_eos1);
				} else {
					if (filename2 == "") {					
						struct_eos2 = std::make_shared<struct_eos>(scf_options::M2, R2, scf_options::nc2, scf_options::ne2, scf_options::core_frac2,
							scf_options::mu2);
					} else {
						struct_eos2 = std::make_shared<struct_eos>(scf_options::M2, scf_options::core_frac2, filename2);
					}
				}
			}
		}
		printf( "R1: %e from eos: %e, R2: %e, from eos: %e, c1_x %e, c2_x %e, l1_x %e\n", R1, struct_eos1->get_R0(), R2, struct_eos2->get_R0(), c1_x, c2_x, l1_x);
	}
};

static scf_parameters& initial_params() {
	static scf_parameters a;
	return a;
}

real grid::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, struct_eos struct_eos_1, struct_eos struct_eos_2) {

	if (omega <= 0.0) {
		printf("OMEGA <= 0.0\n");
		abort();
	}
	real rho_int = 10.0 * rho_floor;
	rho_int = SQRT(rho_int * rho_floor);
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer D = -H_BW;
				const integer iiih = hindex(i, j, k);
				const integer iiig = gindex(i + D, j + D, k + D);
				const real x = X[XDIM][iiih];
				const real y = X[YDIM][iiih];
				const real z = X[ZDIM][iiih];
				const real R = SQRT(POWER(x - com, 2) + y * y);
				real rho = U[rho_i][iiih];
				real phi_eff = G[iiig][phi_i] - 0.5 * POWER(omega * R, 2);
				const real fx = G[iiig][gx_i] + (x - com) * POWER(omega, 2);
				const real fy = G[iiig][gy_i] + y * POWER(omega, 2);
				const real fz = G[iiig][gz_i];

				bool is_donor_side;
				real g;
				real g1 = (x - c1_x) * fx + y * fy + z * fz;
				real g2 = (x - c2_x) * fx + y * fy + z * fz;
				if (x >= l1_x + dx) {
					is_donor_side = true;
					g = g2;
				} else if (x <= l1_x - dx) {
					g = g1;
					is_donor_side = false;
				} else {
					if (g1 < g2) {
						is_donor_side = false;
						g = g1;
					} else {
						is_donor_side = true;
						g = g2;
					}
				}
				real C = is_donor_side ? c2 : c1;
				auto this_struct_eos = is_donor_side ? struct_eos_2 : struct_eos_1;
				real cx, ti_omega;
				if (!is_donor_side) {
					cx = c1_x;
					ti_omega = scf_options::async1 * omega;
				} else {
					cx = c2_x;
					ti_omega = scf_options::async2 * omega;
				}
				phi_eff -= 0.5 * ti_omega * ti_omega * R * R;
				phi_eff -= omega * ti_omega * R * R;
				phi_eff += (omega + ti_omega) * ti_omega * cx * x;
				real new_rho, eint;
				if (g <= 0.0) {
					ASSERT_NONAN(phi_eff);
					ASSERT_NONAN(C);
					new_rho = std::max(this_struct_eos.enthalpy_to_density(std::max(C - phi_eff, this_struct_eos.hfloor())), rho_floor);
				} else {
					new_rho = rho_floor;
				}
				ASSERT_NONAN(new_rho);
				rho = std::max((1.0 - w0) * rho + w0 * new_rho, rho_floor);
				if (new_rho < rho_int) {
					rho = rho_floor;
				}
				U[rho_i][iiih] = rho;
				const real rho0 = rho - rho_floor;
				ASSERT_NONAN(rho0);
				if (opts().eos == WD) {
					U[spc_ac_i][iiih] = rho > this_struct_eos.wd_core_cut ? (is_donor_side ? 0.0 : rho) : 0.0;
					U[spc_dc_i][iiih] = rho > this_struct_eos.wd_core_cut ? (is_donor_side ? rho : 0.0) : 0.0;
					U[spc_ae_i][iiih] = rho <= this_struct_eos.wd_core_cut ? (is_donor_side ? 0.0 : rho) : 0.0;
					U[spc_de_i][iiih] = rho <= this_struct_eos.wd_core_cut ? (is_donor_side ? rho : 0.0) : 0.0;
				} else {
					U[spc_ac_i][iiih] = rho > this_struct_eos.dE() ? (is_donor_side ? 0.0 : rho0) : 0.0;
					U[spc_dc_i][iiih] = rho > this_struct_eos.dE() ? (is_donor_side ? rho0 : 0.0) : 0.0;
					U[spc_ae_i][iiih] = rho <= this_struct_eos.dE() ? (is_donor_side ? 0.0 : rho0) : 0.0;
					U[spc_de_i][iiih] = rho <= this_struct_eos.dE() ? (is_donor_side ? rho0 : 0.0) : 0.0;
				}
				real sx, sy;
				U[spc_vac_i][iiih] = rho_floor;
				if (opts().eos == WD) {
					double abar = 0.0, zbar = 0.0;
					for (int s = 0; s < opts().n_species; s++) {
						abar += U[spc_i + s][iiih] / opts().atomic_mass[s];
						zbar += U[spc_i + s][iiih] * opts().atomic_number[s] / opts().atomic_mass[s];
					}
					abar = rho / abar;
					zbar *= abar / rho;
					eint = this_struct_eos.energy(rho);
				} else {
					eint = std::max(opts().star_egas_out, this_struct_eos.pressure(rho) / (fgamma - 1.0));
				}
				ASSERT_NONAN(eint);
				if (opts().v1309) {
					if (rho0 < this_struct_eos.get_cutoff_density()) {
						U[spc_de_i][iiih] = U[spc_ae_i][iiih] = U[spc_dc_i][iiih] = U[spc_ac_i][iiih] = 0.0;
						U[spc_vac_i][iiih] += rho0;
					}
				}
				sx = -omega * y * rho;
				sy = +omega * (x - com) * rho;
				sx += -ti_omega * y * rho;
				sy += +ti_omega * (x - cx) * rho;
				U[sz_i][iiih] = 0.0;
				if (rho == rho_floor) {
					sx = sy = 0.0;
					eint = -0.5 * rho_floor * G[iiig][phi_i];
					if (opts().eos == WD) {
						eint -= 3.0 * ztwd_pressure(rho);
					}
					eint = std::max(eint, 0.0);
					eint /= 3.0 * (fgamma - 1.0);
					if (opts().eos == WD) {
						eint += ztwd_energy(rho);
					}
					//			eint = 0.0;
				}
				real etherm = eint;
				if (opts().eos == WD) {
					etherm -= ztwd_energy(rho);
					etherm = std::max(1.0e-20, etherm);
				}

				U[sx_i][iiih] = sx;
				U[sy_i][iiih] = sy;
				U[tau_i][iiih] = POWER(etherm, 3.0 / 5.0);
				U[egas_i][iiih] = eint + (sx * sx + sy * sy) / 2.0 * INVERSE(rho);
				U[lx_i][iiih] = -z * sy;
				U[ly_i][iiih] = +z * sx;
				U[lz_i][iiih] = x * sy - y * sx;

				auto const parts_in_cell = get_particles_inds(particles, iiig);
				for (auto const p_ind : parts_in_cell) {
					auto &p = particles[p_ind];
					p.mass = p.mass * (1.0 - w0) + w0 * this_struct_eos.get_p_mass();
					p.vel[XDIM] = -omega * p.pos[YDIM];
					p.vel[YDIM] = omega * (p.pos[XDIM] - com);
					printf("updating particle %i, %e, loc: (%e, %e, %e), vel: (%e, %e, %e)\n", p.id, p.mass, p.pos[XDIM], p.pos[YDIM], p.pos[ZDIM], p.vel[XDIM], p.vel[YDIM], p.vel[ZDIM]);	
				}
			}
		}
	}
	init_z_field();
	if (opts().radiation) {
		rad_grid_ptr->initialize_erad(U[rho_i], U[tau_i]);
	}
	return 0.0;
}

void node_server::run_scf(std::string const &data_dir) {
	solve_gravity(false, false);
	real omega = initial_params().omega;
	real jorb0;
//	printf( "Starting SCF\n");
	grid::set_omega(omega);
	printf("Starting SCF\n");
	real l1_phi = 0.0, l2_phi, l3_phi;
	for (integer i = 0; i != itermax; ++i) {
//		profiler_output(stdout);
		char buffer[33];    // 21 bytes for int (max) + some leeway
		sprintf(buffer, "X.scf.%i", int(i));
		auto &params = initial_params();
		//	set_omega_and_pivot();
		if (i % opts().scf_output_frequency == 0) {
			if (!opts().disable_output) {
				output_all(this, buffer, i, i == 100 || i == 0);
			}
		}
		auto diags = diagnostics();
		real pmass1 = params.struct_eos1->get_p_mass(); 
		real pmass2 = params.struct_eos2->get_p_mass();
		real f1, f2, f1_p, f2_p;
		if (pmass1 == 0.0) {
			f1 = scf_options::M1 * INVERSE(diags.m[0]);
			f1_p = 1.0;
		} else {
			f1 = (scf_options::M1 - pmass1) * INVERSE(diags.m[0] - pmass1);; 
			f1_p = 1.0; //(scf_options::M1 - (diags.m[0] - pmass1)) * INVERSE(pmass1);
			//params.struct_eos2->set_p_mass(scf_options::M1 - (diags.m[0] - pmass1));
		}
		if (pmass2 == 0.0) {
			f2 = scf_options::M2 * INVERSE(diags.m[1]);
			f2_p = 1.0;
		} else {
			f2 = (scf_options::M2 - pmass2) * INVERSE(diags.m[1] - pmass2);
			f1_p = 1.0; //(scf_options::M2 - (diags.m[1] - pmass2)) * INVERSE(pmass2);
			//params.struct_eos2->set_p_mass(scf_options::M2 - (diags.m[1] - pmass2));
		}
		printf("M1: %e, M2: %e, R1: %e, R2: %e, com1: (%e, %e, %e), com2: (%e, %e, %e), grid_com: (%e, %e, %e), rho1: %e, rho2: %e\n", diags.m[0], diags.m[1], params.struct_eos1->get_R0(), params.struct_eos2->get_R0(),
			diags.com[0][0], diags.com[0][1], diags.com[0][2], 
			diags.com[1][0], diags.com[1][1], diags.com[1][2],
			diags.grid_com[0], diags.grid_com[1], diags.grid_com[2], 
			diags.rho_max[0], diags.rho_max[1]);
		real f = (scf_options::M1 + scf_options::M2) * INVERSE(diags.m[0] + diags.m[1]);
		f = (f + 1.0) / 2.0;
                solve_gravity(false, false);
                diags = diagnostics();
		rho_mult(f1, f2, f1_p, f2_p);
		solve_gravity(false, false);
		diags = diagnostics();
		rho_move(diags.grid_com[0]);
		solve_gravity(false, false);
		diags = diagnostics();
		real iorb = diags.z_mom_orb;
		real is1 = diags.z_moment[0];
		real is2 = diags.z_moment[1];
		real M1 = diags.m[0];
		real M2 = diags.m[1];
		real j1 = is1 * omega * (1.0 + scf_options::async1);
		real j2 = is2 * omega * (1.0 + scf_options::async2);
		real jorb = iorb * omega;
		if (i == 0) {
			jorb0 = jorb;
		}
		real spin_ratio = (j1 + j2) * INVERSE(jorb);
		//solve_gravity(false, false);
		auto axis = grid_ptr->find_axis();
		auto loc = line_of_centers(axis);

		real l1_x, c1_x, c2_x; //, l2_x, l3_x;

		real com = diags.grid_com[0]; //axis.second[0];
		real new_omega;
		new_omega = jorb0 * INVERSE(iorb);
		omega = new_omega;
		std::pair<real, real> rho1_max;
		std::pair<real, real> rho2_max;
		std::pair<real, real> l1_phi_pair;
		std::pair<real, real> l2_phi_pair;
		std::pair<real, real> l3_phi_pair;
		real phi_1, phi_2;
		//if (scf_options::filename1 == "") {
			line_of_centers_analyze(loc, omega, rho1_max, rho2_max, l1_phi_pair, l2_phi_pair, l3_phi_pair, phi_1, phi_2);
		//} else { -- experimental for now
		//	line_of_centers_analyze(loc, omega, rho1_max, rho2_max, l1_phi_pair, l2_phi_pair, l3_phi_pair, phi_1, phi_2, params.struct_eos1->get_RC());
		//}
		real rho1, rho2;
		if (rho1_max.first > rho2_max.first) {
			std::swap(phi_1, phi_2);
			std::swap(rho1_max, rho2_max);
		}
		c1_x = diags.com[0][XDIM];
		c2_x = diags.com[1][XDIM];
		rho1 = diags.rho_max[0];
		rho2 = diags.rho_max[1];
		l1_x = l1_phi_pair.first;
		l1_phi = l1_phi_pair.second;
		l2_phi = l2_phi_pair.second;
		l3_phi = l3_phi_pair.second;

		printf("grid com: (%e, %e, %e), com2: (%e, %e, %e)\n", diags.grid_com[0], diags.grid_com[1], diags.grid_com[2], axis.second[0], axis.second[1], axis.second[2]);
		params.struct_eos2->set_d0(rho2);
		if (opts().eos == WD) {
			params.struct_eos1->set_wd_T0(0.0, opts().atomic_mass[0], opts().atomic_number[0]);
			params.struct_eos2->set_wd_T0(0.0, opts().atomic_mass[3], opts().atomic_number[3]);
///			printf("wd_eps = %e %e\n", params.struct_eos1->wd_eps, params.struct_eos2->wd_eps);
		}
		if (scf_options::equal_struct_eos) {
			//	printf( "%e %e \n", rho1, rho1*f0);
			params.struct_eos1->set_d0_using_struct_eos(rho1, *(params.struct_eos2));
		} else {
			if (scf_options::filename1 == "") {
				params.struct_eos1->set_d0(rho1);
			}
		}
		static real rhoc1 = 1.0e-3 * rho1;
		if (opts().v1309) {
			rhoc1 *= INVERSE(POWER(spin_ratio * 3.0, w0));
			params.struct_eos1->set_cutoff_density(rhoc1);
			params.struct_eos2->set_cutoff_density(rhoc1);
		}

		real c_1, c_2;
		if (scf_options::equal_struct_eos) {
			const real alo2 = 1.0 - scf_options::fill2;
			const real ahi2 = scf_options::fill2;
			c_2 = phi_2 * alo2 + ahi2 * l1_phi;
			c_1 = params.struct_eos1->h0() + phi_1;
		} else {
			if (scf_options::contact_fill > 0.0) {
				const real alo = 1.0 - scf_options::contact_fill;
				const real ahi = scf_options::contact_fill;
				c_1 = c_2 = l1_phi * alo + ahi * std::min(l3_phi, l2_phi);
			} else {
				const real alo1 = 1.0 - scf_options::fill1;
				const real ahi1 = scf_options::fill1;
				const real alo2 = 1.0 - scf_options::fill2;
				const real ahi2 = scf_options::fill2;
				c_2 = phi_2 * alo2 + ahi2 * l1_phi;
		//		if (scf_options::filename1 == "") {
					c_1 = phi_1 * alo1 + ahi1 * l1_phi;
		//		} else { // -- experimental for now, assuming Rst == Rrl
		//			c_1 = std::min(phi_1 + params.struct_eos1->HE(), 1.0001*l1_phi);
		//		}
			}
		}
		if (opts().v1309) {
			c_1 += params.struct_eos1->hfloor();
			c_2 += params.struct_eos2->hfloor();
		}
		if (!scf_options::equal_struct_eos) {
			if (scf_options::filename1 == "") {
				params.struct_eos1->set_h0(c_1 - phi_1);
			}
		}
		printf("eos2 - old h0: %e, new h0: %e, c_2: %e, phi_2: %e, phi_l1: %e\n", params.struct_eos2->h0(), c_2 - phi_2, c_2, phi_2, l1_phi);
		printf("eos1 - old h0: %e, new h0: %e, c_1: %e, phi_1: %e, phi_l1: %e\n", params.struct_eos1->h0(), c_1 - phi_1, c_1, phi_1, l1_phi);
		params.struct_eos2->set_h0(c_2 - phi_2);
		//	printf( "---------\n");
		auto e1 = params.struct_eos1;
		auto e2 = params.struct_eos2;

		real core_frac_1 = diags.grid_sum[spc_ac_i] * INVERSE(M1);
		real core_frac_2 = diags.grid_sum[spc_dc_i] * INVERSE(M2);
		const real virial = diags.virial;
		real e1f, e2f;
		if (opts().eos == WD) {
			e1f = e1->wd_core_cut;
			e2f = e2->wd_core_cut;
			auto relerr = 2.0 * (scf_options::core_frac1 - core_frac_1) / (scf_options::core_frac1 + core_frac_1);
			if (relerr < 0.0) {
				e1f = (1.0 - w0) * e1f + w0 * e1f * (1.0 - 10.0 * relerr);
			} else {
				e1f = (1.0 - w0) * e1f + w0 * e1f / (1.0 + 10.0 * relerr);
			}
			relerr = 2.0 * (scf_options::core_frac2 - core_frac_2) / (scf_options::core_frac2 + core_frac_2);
			if (relerr < 0.0) {
				e2f = (1.0 - w0) * e2f + w0 * e2f * (1.0 - 10.0 * relerr);
			} else {
				e2f = (1.0 - w0) * e2f + w0 * e2f / (1.0 + 10.0 * relerr);
			}
			e1->wd_core_cut = e1f;
			e2->wd_core_cut = e2f;
		} else {
			if (!scf_options::equal_struct_eos) {
				e1f = e1->get_frac();
				if (core_frac_1 == 0.0) {
					e1f = 0.5 + 0.5 * e1f;
				} else {
					e1f = (1.0 - w0) * e1f + w0 * POWER(e1f, scf_options::core_frac1 * INVERSE( core_frac_1));
				}
				e1->set_frac(e1f);
			}
			e2f = e2->get_frac();

			if (scf_options::contact_fill <= 0.0) {
				if (core_frac_2 == 0.0) {
					e2f = 0.5 + 0.5 * e2f;
				} else {
					e2f = (1.0 - w0) * e2f + w0 * POWER(e2f, scf_options::core_frac2 * INVERSE(core_frac_2));
				}
				if (!scf_options::equal_struct_eos) {
					e2->set_frac(e2f);
				}
			} else {
				if (opts().v1309) {
					const real ne = scf_options::ne1;
					const real gamma = grid::get_fgamma();
					const real p0 = params.struct_eos1->P0();
					const real de = params.struct_eos1->dE();
					const real s1 = POWER(p0 * POWER(rhoc1 * INVERSE( de), 1.0 + 1.0 * INVERSE( ne)) / (gamma - 1.0), 1.0 / gamma) * INVERSE(rhoc1);
					printf("S = %e\n", s1);
					e2->set_entropy(s1);
				} else {
					e2->set_entropy(e1->s0());
				}
			}
			e1f = e1->get_frac();
			e2f = e2->get_frac();
		}
		params.struct_eos2->update_R0(); // important in the case of regridding
		real amin, jmin, mu;
		mu = M1 * M2 * INVERSE(M1 + M2);
		amin = SQRT(3.0 * (is1 + is2) * INVERSE( mu ));

		const real r0 = POWER(diags.stellar_vol[0] / (1.3333333333 * 3.14159), 1.0 / 3.0);
		const real r1 = POWER(diags.stellar_vol[1] / (1.3333333333 * 3.14159), 1.0 / 3.0);
		const real fi0 = diags.stellar_vol[0] * INVERSE(diags.roche_vol[0]);
		const real fi1 = diags.stellar_vol[1] * INVERSE(diags.roche_vol[1]);

		jmin = SQRT((M1 + M2)) * (mu * POWER(amin, 0.5) + (is1 + is2) * POWER(amin, -1.5));
		if (i % 5 == 0) {
			printf("   %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s\n", "rho1", "rho2", "M1", "M2",
					"is1", "is2", "omega", "virial", "core_frac_1", "core_frac_2", "jorb", "jmin", "amin", "jtot", "com", "spin_ratio", "iorb", "R1", "R2",
					"fill1", "fill2");
		}
		lprintf((opts().data_dir + "log.txt").c_str(),
				"%i %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e  %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e\n", i, rho1, rho2,
				M1, M2, is1, is2, omega, virial, core_frac_1, core_frac_2, jorb, jmin, amin, j1 + j2 + jorb, com, spin_ratio, iorb, r0, r1, fi0, fi1, w0, e1f,
				e2f);
		if (i % 10 == 0) {
			regrid(me.get_unmanaged_gid(), omega, -1, false);
		} else {
			grid::set_omega(omega);
		}
		if (opts().eos == WD) {
			set_AB(e2->A, e2->B());
		}
//		printf( "%e %e\n", grid::get_A(), grid::get_B());
		printf( "before scf update, rho1_x: %e, rho2_x: %e, l1_x: %e, c1: %e, c2: %e, omega: %e\n", rho1_max.first, rho2_max.first, l1_x, c_1, c_2, omega);
		scf_update(com, omega, c_1, c_2, rho1_max.first, rho2_max.first, l1_x, *e1, *e2);
		solve_gravity(false, false);
		w0 = std::min(w0max, w0 * POWER(w0max / w0init, 1.0 / iter2max));

	}
	if (opts().radiation) {
		if (opts().eos == WD) {
			set_cgs();
			all_hydro_bounds();
			grid_ptr->rad_init();
		}
	}
}

std::vector<real> scf_binary(real x, real y, real z, real dx) {
	const real fgamma = grid::get_fgamma();
	std::vector<real> u(opts().n_fields, real(0));
	static auto &params = initial_params();
	if (!opts().restart_filename.empty()) {
		return u;
	}
	std::shared_ptr<struct_eos> this_struct_eos;
	real r, ei;
	int M = 1;
	if (x < params.l1_x) {
		this_struct_eos = params.struct_eos1;
	} else {
		this_struct_eos = params.struct_eos2;
	}
	real rho = 0;
	const real R0 = this_struct_eos->get_R0();
	int nsamp = 0;
	for (double x0 = x - dx / 2.0 + dx / 2.0 / M; x0 < x + dx / 2.0; x0 += dx / M) {
		for (double y0 = y - dx / 2.0 + dx / 2.0 / M; y0 < y + dx / 2.0; y0 += dx / M) {
			for (double z0 = z - dx / 2.0 + dx / 2.0 / M; z0 < z + dx / 2.0; z0 += dx / M) {
				++nsamp;
				if (x < params.l1_x) {
					r = SQRT(std::pow(x0 - params.c1_x, 2) + y0 * y0 + z0 * z0);
				} else {
					r = SQRT(std::pow(x0 - params.c2_x, 2) + y0 * y0 + z0 * z0);
				}
				if (r <= R0) {
					rho += this_struct_eos->density_at(r, dx);
				}
			}
		}
	}
	rho = std::max(rho / nsamp, rho_floor);
	if (opts().eos == WD) {
		ei = this_struct_eos->energy(rho);
	} else {
		ei = this_struct_eos->pressure(rho) / (fgamma - 1.0);
	}
	u[rho_i] = rho;
	const real rho0 = rho - rho_floor;
	if (opts().eos == WD) {
		u[spc_ac_i] = rho > this_struct_eos->wd_core_cut ? (x > params.l1_x ? 0.0 : rho) : 0.0;
		u[spc_dc_i] = rho > this_struct_eos->wd_core_cut ? (x > params.l1_x ? rho : 0.0) : 0.0;
		u[spc_ae_i] = rho <= this_struct_eos->wd_core_cut ? (x > params.l1_x ? 0.0 : rho) : 0.0;
		u[spc_de_i] = rho <= this_struct_eos->wd_core_cut ? (x > params.l1_x ? rho : 0.0) : 0.0;
	} else {
		u[spc_ac_i] = rho > this_struct_eos->dE() ? (x > params.l1_x ? 0.0 : rho0) : 0.0;
		u[spc_dc_i] = rho > this_struct_eos->dE() ? (x > params.l1_x ? rho0 : 0.0) : 0.0;
		u[spc_ae_i] = rho <= this_struct_eos->dE() ? (x > params.l1_x ? 0.0 : rho0) : 0.0;
		u[spc_de_i] = rho <= this_struct_eos->dE() ? (x > params.l1_x ? rho0 : 0.0) : 0.0;
	}
	u[spc_vac_i] = rho_floor;
	u[egas_i] = ei + 0.5 * rho * (x * x + y * y) * params.omega * params.omega;
	u[sx_i] = -y * params.omega * rho;
	u[sy_i] = +x * params.omega * rho;
	u[sz_i] = 0.0;
	real etherm = ei;
	if (opts().eos == WD) {
		etherm -= ztwd_energy(rho);
		etherm = std::max(1.0e-10, etherm);
	}
	u[tau_i] = POWER(etherm, 3.0 / 5.0);
	return u;
}

std::vector<particle> scf_binary_particles(real x, real y, real z, real dx) {
        std::vector<particle> p(0);
	integer pid;
	real c_x;
        static auto &params = initial_params();
        if (!opts().restart_filename.empty()) {
                return p;
        }
        std::shared_ptr<struct_eos> this_struct_eos;
        if (x < params.l1_x) {
                this_struct_eos = params.struct_eos1;
		pid = 11;
		c_x = params.c1_x;
        } else {
                this_struct_eos = params.struct_eos2;
		pid = 12;
		c_x = params.c2_x;
        }
	real const c_y = 0.0;
	real const c_z = 0.0;
	real const pmass = this_struct_eos->get_p_mass();
	if (pmass > 0.0) {
		if ((c_y < y + 0.5*dx) && (c_y >=  y - 0.5*dx) && (c_z < z + 0.5*dx) && (c_z >= z - 0.5*dx) && (c_x < x + dx*0.5) && (c_x >= x - dx*0.5)) {
			space_vector p_pos;
	                p_pos[XDIM] = c_x;
	                if (NDIM > 1) {
        	                p_pos[YDIM] = c_y;
	                        if (NDIM > 2) {
        	                        p_pos[ZDIM] = c_z;
                        	}
	                }
	                p.push_back(particle(pmass, p_pos, pid, 1));
		}
	}
	return p;
}
