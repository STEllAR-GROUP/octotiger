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
#include "util.hpp"
#include "profiler.hpp"
extern options opts;

// w0 = speed of convergence. Adjust lower if nan
const real w0 = 1.0 / 4.0;
const real rho_floor = 1.0e-15;

namespace scf_options {

static constexpr real async1 = -0.0e-2;
static constexpr real async2 = -0.0e-2;
static constexpr bool equal_struct_eos = true; // If true, EOS of accretor will be set to that of donor
static constexpr real M1 = 1.0;// Mass of primary
static constexpr real M2 = 0.2;// Mass of secondaries
static constexpr real nc1 = 2.5;// Primary core polytropic index
static constexpr real nc2 = 1.5;// Secondary core polytropic index
static constexpr real ne1 = 1.5;// Primary envelope polytropic index // Ignored if equal_struct_eos=true
static constexpr real ne2 = 1.5;// Secondary envelope polytropic index
static constexpr real mu1 = 1.0;// Primary ratio of molecular weights // Ignored if equal_struct_eos=true
static constexpr real mu2 = 1.0;// Primary ratio of molecular weights
static constexpr real a = 1.00;// approx. orbital sep
static constexpr real core_frac1 = 0.9;// Desired core fraction of primary // Ignored if equal_struct_eos=true
static constexpr real core_frac2 = 0.9;// Desired core fraction of secondary - IGNORED FOR CONTACT binaries
static constexpr real fill1 = 1.0;// 1d Roche fill factor for primary (ignored if contact fill is > 0.0) //  - IGNORED FOR CONTACT binaries  // Ignored if equal_struct_eos=true
static constexpr real fill2 = 1.0;// 1d Roche fill factor for secondary (ignored if contact fill is > 0.0) // - IGNORED FOR CONTACT binaries
static real contact_fill = 0.00; //  Degree of contact - IGNORED FOR NON-CONTACT binaries // SET to ZERO for equal_struct_eos=true
// Contact fill factor
}
;

//0.5=.313
//0.6 .305

hpx::future<void> node_client::rho_move(real x) const {
	return hpx::async<typename node_server::rho_move_action>(get_unmanaged_gid(), x);
}

void node_server::rho_move(real x) {
	std::array<hpx::future<void>, NCHILD> futs;
	if (is_refined) {
        integer index = 0;
		for (auto& child : children) {
			futs[index++] = child.rho_move(x);
		}
	}
	grid_ptr->rho_move(x);
	all_hydro_bounds();

    wait_all_and_propagate_exceptions(futs);
}

typedef typename node_server::scf_update_action scf_update_action_type;
HPX_REGISTER_ACTION (scf_update_action_type);

typedef typename node_server::rho_mult_action rho_mult_action_type;
HPX_REGISTER_ACTION (rho_mult_action_type);

hpx::future<void> node_client::rho_mult(real f0, real f1) const {
	return hpx::async<typename node_server::rho_mult_action>(get_unmanaged_gid(), f0, f1);
}

hpx::future<real> node_client::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, struct_eos e1, struct_eos e2) const {
	return hpx::async<typename node_server::scf_update_action>(get_unmanaged_gid(), com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
}

void node_server::rho_mult(real f0, real f1) {
	std::array<hpx::future<void>, NCHILD> futs;
	if (is_refined) {
        integer index = 0;
		for (auto& child : children) {
			futs[index++] = child.rho_mult(f0, f1);
		}
	}
	grid_ptr->rho_mult(f0, f1);
	all_hydro_bounds();

    wait_all_and_propagate_exceptions(futs);
}

real node_server::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, struct_eos e1, struct_eos e2) {
	grid::set_omega(omega);
	std::array<hpx::future<real>, NCHILD> futs;
	real res;
	if (is_refined) {
        integer index = 0;
		for (auto& child : children) {
			futs[index++] = child.scf_update(com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
		}
		res = ZERO;
	} else {
		res = grid_ptr->scf_update(com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
	}
	all_hydro_bounds();
    res = std::accumulate(
        futs.begin(), futs.end(), res,
        [](real res, hpx::future<real> & f)
        {
            return res + f.get();
        });
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
		const real fill1 = scf_options::fill1;
		const real contact = scf_options::contact_fill;
		const real a = scf_options::a;
		G = 1.0;
		const real c = 4.0 * M_PI / 3.0;
		q = M2 / M1;
		c1_x = -a * M2 / (M1 + M2);
		c2_x = +a * M1 / (M1 + M2);
		l1_x = a * (0.5 - 0.227 * log10(q)) + c1_x;
		omega = std::sqrt((G * (M1 + M2)) / (a * a * a));
		const real fill2 = scf_options::fill2;
		const real V1 = find_V(M1 / M2) * cube(a) * cube(fill1);
		const real V2 = find_V(M2 / M1) * cube(a) * cube(fill2);
		R1 = std::pow(V1 / c, 1.0 / 3.0);
		R2 = std::pow(V2 / c, 1.0 / 3.0);
		if (opts.eos == WD) {
			struct_eos2 = std::make_shared < struct_eos > (scf_options::M2, R2);
			struct_eos1 = std::make_shared < struct_eos > (scf_options::M1, *struct_eos2);
		} else {
			if (scf_options::equal_struct_eos) {
				struct_eos2 = std::make_shared < struct_eos > (scf_options::M2, R2, scf_options::nc2, scf_options::ne2, scf_options::core_frac2, scf_options::mu2);
				struct_eos1 = std::make_shared < struct_eos > (scf_options::M1, scf_options::nc1, *struct_eos2);
			} else {
				struct_eos1 = std::make_shared < struct_eos > (scf_options::M1, R1, scf_options::nc1, scf_options::ne1, scf_options::core_frac1, scf_options::mu1);
				if (contact > 0.0) {
					struct_eos2 = std::make_shared < struct_eos > (scf_options::M2, R2, scf_options::nc2, scf_options::ne2, scf_options::mu2, *struct_eos1);
				} else {
					struct_eos2 = std::make_shared < struct_eos > (scf_options::M2, R2, scf_options::nc2, scf_options::ne2, scf_options::core_frac2, scf_options::mu2);
				}
			}
		}
	}
};

//0.15=0.77
//0.30=0.41
//0.33=0.35
static scf_parameters& initial_params() {
	static scf_parameters a;
	return a;
}

real grid::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, struct_eos struct_eos_1, struct_eos struct_eos_2) {
	PROF_BEGIN;
	if (omega <= 0.0) {
		printf("OMEGA <= 0.0\n");
		abort();
	}
	for (integer i = H_BW; i != H_NX - H_BW; ++i) {
		for (integer j = H_BW; j != H_NX - H_BW; ++j) {
			for (integer k = H_BW; k != H_NX - H_BW; ++k) {
				const integer D = -H_BW;
				const integer iiih = hindex(i, j, k);
				const integer iiig = gindex(i + D, j + D, k + D);
				const real x = X[XDIM][iiih];
				const real y = X[YDIM][iiih];
				const real z = X[ZDIM][iiih];
				const real R = std::sqrt(std::pow(x - com, 2) + y * y);
				real rho = U[rho_i][iiih];
				real phi_eff = G[iiig][phi_i] - 0.5 * std::pow(omega * R, 2);
				const real fx = G[iiig][gx_i] + (x - com) * std::pow(omega, 2);
				const real fy = G[iiig][gy_i] + y * std::pow(omega, 2);
				const real fz = G[iiig][gz_i];

				bool is_donor_side;
				real g;
				real g1 = (x - c1_x) * fx + y * fy + z * fz;
				real g2 = (x - c2_x) * fx + y * fy + z * fz;
				if (x >= l1_x /*+ 10.0*dx*/) {
					is_donor_side = true;
					g = g2;
				} else if (x <= l1_x /*- 10.0*dx*/) {
					g = g1;
					is_donor_side = false;
				} /*else {
				 if( g1 < g2 ) {
				 is_donor_side = false;
				 g = g1;
				 } else {
				 is_donor_side = true;
				 g = g2;
				 }
				 }*/
				real C = is_donor_side ? c2 : c1;
				//			real x0 = is_donor_side ? c2_x : c1_x;
				auto this_struct_eos = is_donor_side ? struct_eos_2 : struct_eos_1;
				real cx, ti_omega; //, Rc;
				if (!is_donor_side) {
					cx = c1_x;
					ti_omega = scf_options::async1 * omega;
				} else {
					cx = c2_x;
					ti_omega = scf_options::async2 * omega;
				}
				//	Rc = std::sqrt( x*x + cx*cx - 2.0*x*cx + y*y );
				phi_eff -= 0.5 * ti_omega * ti_omega * R * R;
				phi_eff -= omega * ti_omega * R * R;
				phi_eff += (omega + ti_omega) * ti_omega * cx * x;
				real new_rho, eint;
				const auto smallest = 1.0e-20;
				if (g <= 0.0) {
					ASSERT_NONAN(phi_eff);
					ASSERT_NONAN(C);
					new_rho = std::max(this_struct_eos.enthalpy_to_density(std::max(C - phi_eff, smallest)), rho_floor);
				} else {
					new_rho = rho_floor;
				}
				ASSERT_NONAN(new_rho);
				rho = std::max((1.0 - w0) * rho + w0 * new_rho, rho_floor);
				if( opts.eos == WD ) {
					eint = this_struct_eos.energy(rho);
				} else {
					eint = std::max(ei_floor, this_struct_eos.pressure(rho) / (fgamma - 1.0));
				}
				U[rho_i][iiih] = rho;
				const real rho0 = rho - rho_floor;
				if( opts.eos == WD ) {
					U[spc_ac_i][iiih] = (is_donor_side ? 0.0 : rho0);
					U[spc_dc_i][iiih] = (is_donor_side ? rho0 : 0.0);
					U[spc_ae_i][iiih] = 0.0;
					U[spc_de_i][iiih] = 0.0;
				} else {
					U[spc_ac_i][iiih] = rho > this_struct_eos.dE() ? (is_donor_side ? 0.0 : rho0) : 0.0;
					U[spc_dc_i][iiih] = rho > this_struct_eos.dE() ? (is_donor_side ? rho0 : 0.0) : 0.0;
					U[spc_ae_i][iiih] = rho <= this_struct_eos.dE() ? (is_donor_side ? 0.0 : rho0) : 0.0;
					U[spc_de_i][iiih] = rho <= this_struct_eos.dE() ? (is_donor_side ? rho0 : 0.0) : 0.0;
				}
				U[spc_vac_i][iiih] = rho_floor;

				U[sx_i][iiih] = -omega * y * rho;
				U[sy_i][iiih] = +omega * (x - com) * rho;
				U[sx_i][iiih] += -ti_omega * y * rho;
				U[sy_i][iiih] += +ti_omega * (x - cx) * rho;
				U[sz_i][iiih] = 0.0;
				if( opts.eos == WD ) {
					U[tau_i][iiih] = 1.0e-40;
					eint += std::pow(U[tau_i][iiih],1.0/fgamma);
				} else {
					U[tau_i][iiih] = std::pow(eint, 1.0 / fgamma);
				}
				U[egas_i][iiih] = eint + std::pow(R * omega, 2) * rho / 2.0;
				U[zx_i][iiih] = 0.0;
				U[zy_i][iiih] = 0.0;
				U[zz_i][iiih] = dx * dx * omega * rho / 6.0;
			}
		}
	}
	PROF_END;
	return 0.0;
}

real interpolate(real x1, real x2, real x3, real x4, real y1, real y2, real y3, real y4, real x) {
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

void node_server::run_scf() {

	solve_gravity(false);
	real omega = initial_params().omega;
	real jorb0;
	grid::set_omega(omega);
	for (integer i = 0; i != 100; ++i) {
//		profiler_output(stdout);
        char buffer[33];    // 21 bytes for int (max) + some leeway
        sprintf(buffer, "X.scf.%i.silo", int(i));
		auto& params = initial_params();
		//	set_omega_and_pivot();
		if (i % 100 == 0 && i != 0) {
			output(buffer, i, false);
			save_to_file("scf.chk");
		}
		auto diags = diagnostics();
		real f0 = scf_options::M1 / (diags.primary_sum[rho_i]);
		real f1 = scf_options::M2 / (diags.secondary_sum[rho_i]);
		real f = (scf_options::M1 + scf_options::M2) / diags.grid_sum[rho_i];
//		f = (f + 1.0)/2.0;
		//	printf( "%e %e \n", f0, f1);
		rho_mult(f0, f1);
		diags = diagnostics();
		rho_move(diags.grid_com[rho_i] / 2.0);
		diags = diagnostics();
		real iorb = diags.z_moment;
		real is1 = diags.primary_z_moment;
		real is2 = diags.secondary_z_moment;
		iorb -= is1 + is2;
		real M1 = diags.primary_sum[rho_i];
		real M2 = diags.secondary_sum[rho_i];
		real j1 = is1 * omega * (1.0 + scf_options::async1);
		real j2 = is2 * omega * (1.0 + scf_options::async2);
		real jorb = iorb * omega;
		if (i == 0) {
			jorb0 = jorb;
		}
		real spin_ratio = (j1 + j2) / (jorb);
		real this_m = diags.grid_sum[rho_i];
		solve_gravity(false);

		auto axis = grid_ptr->find_axis();
		auto loc = line_of_centers(axis);

		real l1_x, c1_x, c2_x; //, l2_x, l3_x;
		real l1_phi, l2_phi, l3_phi;

		real com = axis.second[0];
		real new_omega;
		new_omega = jorb0 / iorb;
		omega = new_omega;
		std::pair < real, real > rho1_max;
		std::pair < real, real > rho2_max;
		std::pair < real, real > l1_phi_pair;
		std::pair < real, real > l2_phi_pair;
		std::pair < real, real > l3_phi_pair;
		real phi_1, phi_2;
		line_of_centers_analyze(loc, omega, rho1_max, rho2_max, l1_phi_pair, l2_phi_pair, l3_phi_pair, phi_1, phi_2);
		real rho1, rho2;
		if (rho1_max.first > rho2_max.first) {
			std::swap(phi_1, phi_2);
			std::swap(rho1_max, rho2_max);
		}
		c1_x = diags.primary_com[XDIM];
		c2_x = diags.secondary_com[XDIM];
		rho1 = std::max(diags.field_max[spc_ac_i], diags.field_max[spc_ae_i]);
		rho2 = std::max(diags.field_max[spc_dc_i], diags.field_max[spc_de_i]);
		l1_x = l1_phi_pair.first;
		l1_phi = l1_phi_pair.second;
		l2_phi = l2_phi_pair.second;
		l3_phi = l3_phi_pair.second;

		//	printf( "++++++++++++++++++++%e %e %e %e \n", rho1, rho2, c1_x, c2_x);
		params.struct_eos2->set_d0(rho2 * f1);
		if (scf_options::equal_struct_eos) {
			//	printf( "%e %e \n", rho1, rho1*f0);
			params.struct_eos1->set_d0_using_struct_eos(rho1 * f0, *(params.struct_eos2));
		} else {
			params.struct_eos1->set_d0(rho1 * f0);
		}

		real h_1 = params.struct_eos1->h0();
		real h_2 = params.struct_eos2->h0();

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
				c_1 = phi_1 * alo1 + ahi1 * l1_phi;
				c_2 = phi_2 * alo2 + ahi2 * l1_phi;
				//		c_1 = l1_phi;
				//		c_2 = l1_phi;
			}
		}
		//	printf( "%e %e %e\n", l1_phi, l2_phi, l3_phi);
		if (!scf_options::equal_struct_eos) {
			params.struct_eos1->set_h0(c_1 - phi_1);
		}
		params.struct_eos2->set_h0(c_2 - phi_2);
		auto e1 = params.struct_eos1;
		auto e2 = params.struct_eos2;

		real core_frac_1 = diags.grid_sum[spc_ac_i] / M1;
		real core_frac_2 = diags.grid_sum[spc_dc_i] / M2;
		const real eptot = diags.grid_sum[pot_i];
		const real ektot = diags.grid_sum[egas_i] - 0.5 * eptot;
		const real virial = (2.0 * ektot + 0.5 * eptot) / (2.0 * ektot - 0.5 * eptot);
		const real v1 = diags.primary_volume;
		const real v2 = diags.secondary_volume;
		const real vfactor = 4.0 / 3.0 * M_PI;
		const real r1 = std::pow(v1 / vfactor, 1.0 / 3.0);
		const real r2 = std::pow(v2 / vfactor, 1.0 / 3.0);
		const real g1 = std::sqrt(is1 / (r1 * r1) / M1);
		const real g2 = std::sqrt(is2 / (r2 * r2) / M2);
		const real etot = diags.grid_sum[egas_i] + 0.5 * diags.grid_sum[pot_i];
		real e1f;
		if (opts.eos == WD) {
			if (!scf_options::equal_struct_eos) {
				e1f = e1->get_frac();
				if (core_frac_1 == 0.0) {
					e1f = 0.5 + 0.5 * e1f;
				} else {
					e1f = (1.0 - w0) * e1f + w0 * std::pow(e1f, scf_options::core_frac1 / core_frac_1);
				}
				e1->set_frac(e1f);
			}
			real e2f = e2->get_frac();
			if (scf_options::contact_fill <= 0.0) {
				if (core_frac_2 == 0.0) {
					e2f = 0.5 + 0.5 * e2f;
				} else {
					e2f = (1.0 - w0) * e2f + w0 * std::pow(e2f, scf_options::core_frac2 / core_frac_2);
				}
				e2->set_frac(e2f);
			} else {
				e2->set_entropy(e1->s0());
			}
			e1f = e1->get_frac();
			e2f = e2->get_frac();
		}
		real amin, jmin, mu;
		mu = M1 * M2 / (M1 + M2);
		amin = std::sqrt(3.0 * (is1 + is2) / mu);

		jmin = std::sqrt((M1 + M2)) * (mu * std::pow(amin, 0.5) + (is1 + is2) * std::pow(amin, -1.5));
		if (i % 5 == 0)
			printf("   %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s\n", "rho1", "rho2", "M1", "M2",
				"omega", "virial", "core_frac_1", "core_frac_2", "jorb", "jmin", "amin", "jtot", "com", "spin_ratio", "r1", "r2", "iorb", "pvol", "proche", "svol",
				"sroche");
		lprintf("log.txt", "%i %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e\n", i, rho1, rho2, M1, M2,
			omega, virial, core_frac_1, core_frac_2, jorb, jmin, amin, j1 + j2 + jorb, com, spin_ratio, r1, r2, iorb, diags.primary_volume, diags.roche_vol1,
			diags.secondary_volume, diags.roche_vol2);
		if (i % 10 == 0) {
			regrid(me.get_unmanaged_gid(), false);
		}
		grid::set_omega(omega);
		if( opts.eos == WD ) {
			grid::set_AB(e2->A, e2->B());
		}
//		printf( "%e %e\n", grid::get_A(), grid::get_B());
		scf_update(com, omega, c_1, c_2, rho1_max.first, rho2_max.first, l1_x, *e1, *e2);
		solve_gravity(false);

	}
}

std::vector<real> scf_binary(real x, real y, real z, real dx) {
	const real fgamma = grid::get_fgamma();
	std::vector < real > u(NF, real(0));
	static auto& params = initial_params();
	std::shared_ptr<struct_eos> this_struct_eos;
	real rho, r, ei;
	if (x < params.l1_x) {
		this_struct_eos = params.struct_eos1;
	} else {
		this_struct_eos = params.struct_eos2;
	}
	rho = 0;
	const real R0 = this_struct_eos->get_R0();
	int M = int(dx / R0) + 1;
	int nsamp = 0;
	for (double x0 = x - dx / 2.0 + dx / 2.0 / M; x0 < x + dx; x0 += dx / M) {
		for (double y0 = y - dx / 2.0 + dx / 2.0 / M; y0 < y + dx; y0 += dx / M) {
			for (double z0 = z - dx / 2.0 + dx / 2.0 / M; z0 < z + dx; z0 += dx / M) {
				++nsamp;
				if (x < params.l1_x) {
					r = std::sqrt(std::pow(x0 - params.c1_x, 2) + y0 * y0 + z0 * z0);
				} else {
					r = std::sqrt(std::pow(x0 - params.c2_x, 2) + y0 * y0 + z0 * z0);
				}
				if (r <= R0) {
					rho += this_struct_eos->density_at(r, dx);
				}
			}
		}
	}
//	grid::set_AB(this_struct_eos->A, this_struct_eos->B());
	rho = std::max(rho / nsamp, rho_floor);
	if( opts.eos == WD ) {
		ei = this_struct_eos->energy(rho);
	} else {
		ei = this_struct_eos->pressure(rho) / (fgamma - 1.0);
	}
	u[rho_i] = rho;
	if( opts.eos == WD ) {
		u[spc_ac_i] = x > params.l1_x ? 0.0 : rho;
		u[spc_dc_i] = x > params.l1_x ? rho : 0.0;
		u[spc_ae_i] = 0.0;
		u[spc_de_i] = 0.0;
	} else {
		u[spc_ac_i] = rho > this_struct_eos->dE() ? (x > params.l1_x ? 0.0 : rho) : 0.0;
		u[spc_dc_i] = rho > this_struct_eos->dE() ? (x > params.l1_x ? rho : 0.0) : 0.0;
		u[spc_ae_i] = rho <= this_struct_eos->dE() ? (x > params.l1_x ? 0.0 : rho) : 0.0;
		u[spc_de_i] = rho <= this_struct_eos->dE() ? (x > params.l1_x ? rho : 0.0) : 0.0;
	}
	u[egas_i] = ei + 0.5 * (x * x + y * y) * params.omega * params.omega;
	u[sx_i] = -y * params.omega * rho;
	u[sy_i] = +x * params.omega * rho;
	u[sz_i] = 0.0;
	if( opts.eos == WD ) {
		u[tau_i] = std::pow(ei, 1.0 / fgamma);
	} else {
		u[tau_i] = std::pow(1.0e-15, 1.0 / fgamma);
	}
	return u;
}
