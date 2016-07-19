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
extern options opts;

// w0 = speed of convergence. Adjust lower if nan
const real w0 = 1.0 / 4.0;
const real rho_floor = 1.0e-15;

namespace scf_options {

static constexpr bool equal_eos = false; // If true, EOS of accretor will be set to that of donor
static constexpr real M1 = 1.0;// Mass of primary
static constexpr real M2 = 0.5;// Mass of secondary
static constexpr real nc1 = 3.0;// Primary core polytropic index
static constexpr real nc2 = 3.0;// Secondary core polytropic index
static constexpr real ne1 = 1.5;// Primary envelope polytropic index // Ignored if equal_eos=true
static constexpr real ne2 = 1.5;// Secondary envelope polytropic index
static constexpr real mu1 = 1.0;// Primary ratio of molecular weights // Ignored if equal_eos=true
static constexpr real mu2 = 1.0;// Primary ratio of molecular weights
static constexpr real a = 1.00;// approx. orbital sep
static constexpr real core_frac1 = 0.98;// Desired core fraction of primary // Ignored if equal_eos=true
static constexpr real core_frac2 = 0.72;// Desired core fraction of secondary - IGNORED FOR CONTACT binaries
static constexpr real fill1 = 0.999;// 1d Roche fill factor for primary (ignored if contact fill is > 0.0) //  - IGNORED FOR CONTACT binaries  // Ignored if equal_eos=true
static constexpr real fill2 = 0.999;// 1d Roche fill factor for secondary (ignored if contact fill is > 0.0) // - IGNORED FOR CONTACT binaries
static real contact_fill = 0.5;//  Degree of contact - IGNORED FOR NON-CONTACT binaries // SET to ZERO for equal_eos=true
// Contact fill factor
};

//0.5=.313
//0.6 .305

typedef typename node_server::scf_update_action scf_update_action_type;
HPX_REGISTER_ACTION(scf_update_action_type);

typedef typename node_server::rho_mult_action rho_mult_action_type;
HPX_REGISTER_ACTION(rho_mult_action_type);

typedef typename node_server::rho_move_action rho_move_action_type;
HPX_REGISTER_ACTION(rho_move_action_type);

hpx::future<void> node_client::rho_mult(real f0, real f1) const {
	return hpx::async<typename node_server::rho_mult_action>(get_gid(), f0, f1);
}

hpx::future<void> node_client::rho_move(real x) const {
	return hpx::async<typename node_server::rho_move_action>(get_gid(), x);
}

hpx::future<real> node_client::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, accretor_eos e1, donor_eos e2) const {
	return hpx::async<typename node_server::scf_update_action>(get_gid(), com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
}

void node_server::rho_move(real x) {
	std::vector<hpx::future<void>> futs;
	if (is_refined) {
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.rho_move(x));
		}
	}
	const real dx = 2.0 * opts.xscale / real(INX) / real(1 << my_location.level());
	real w = x / dx;
	w = std::max(-0.5, std::min(0.5, w));
	for (integer i = 1; i != H_NX - 1; ++i) {
		for (integer j = 1; j != H_NX - 1; ++j) {
			for (integer k = 1; k != H_NX - 1; ++k) {
				for (integer si = spc_i; si != NSPECIES + spc_i; ++si) {
					grid_ptr->hydro_value(si, i, j, k) += w * grid_ptr->hydro_value(si, i + 1, j, k);
					grid_ptr->hydro_value(si, i, j, k) -= w * grid_ptr->hydro_value(si, i - 1, j, k);
					grid_ptr->hydro_value(si, i, j, k) = std::max(grid_ptr->hydro_value(si, i, j, k), 0.0);
				}
				grid_ptr->hydro_value(rho_i, i, j, k) = 0.0;
				for (integer si = 0; si != NSPECIES; ++si) {
					grid_ptr->hydro_value(rho_i, i, j, k) += grid_ptr->hydro_value(spc_i + si, i, j, k);
				}
				grid_ptr->hydro_value(rho_i, i, j, k) = std::max(grid_ptr->hydro_value(rho_i, i, j, k), rho_floor);
			}
		}
	}
	exchange_interlevel_hydro_data();
	collect_hydro_boundaries();
	for (auto&& fut : futs ) {
		fut.get();
	}
}

void node_server::rho_mult(real f0, real f1) {
	std::vector<hpx::future<void>> futs;
	if (is_refined) {
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.rho_mult(f0, f1));
		}
	}
	for (integer i = 0; i != H_NX; ++i) {
		for (integer j = 0; j != H_NX; ++j) {
			for (integer k = 0; k != H_NX; ++k) {
				grid_ptr->hydro_value(spc_ac_i, i, j, k) *= f0;
				grid_ptr->hydro_value(spc_dc_i, i, j, k) *= f1;
				grid_ptr->hydro_value(spc_ae_i, i, j, k) *= f0;
				grid_ptr->hydro_value(spc_de_i, i, j, k) *= f1;
				grid_ptr->hydro_value(rho_i, i, j, k) = 0.0;
				for (integer si = 0; si != NSPECIES; ++si) {
					grid_ptr->hydro_value(rho_i, i, j, k) += grid_ptr->hydro_value(spc_i + si, i, j, k);
				}
			}
		}
	}
	exchange_interlevel_hydro_data();
	collect_hydro_boundaries();
	for (auto&& fut : futs ) {
		fut.get();
	}
}

real node_server::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, accretor_eos e1, donor_eos e2) {
	grid::set_omega(omega);
	std::vector < hpx::future < real >> futs;
	real res;
	if (is_refined) {
		futs.reserve(NCHILD);
		for (auto& child : children) {
			futs.push_back(child.scf_update(com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2));
		}
		res = ZERO;
	} else {
		res = grid_ptr->scf_update(com, omega, c1, c2, c1_x, c2_x, l1_x, e1, e2);
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
	real R1;
	real R2;
	real omega;
	real G;
	real q;
	std::shared_ptr<accretor_eos> eos1;
	std::shared_ptr<donor_eos> eos2;
	real l1_x;
	real c1_x;
	real c2_x;
	scf_parameters() {
		if (scf_options::equal_eos) {
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
		const real V1 = find_V(M1 / M2) * std::pow(a, 3.0) * std::pow(fill1, 3.0);
		const real V2 = find_V(M2 / M1) * std::pow(a, 3.0) * std::pow(fill2, 3.0);
		R1 = std::pow(V1 / c, 1.0 / 3.0);
		R2 = std::pow(V2 / c, 1.0 / 3.0);
		if (scf_options::equal_eos) {
			eos2 = std::make_shared < donor_eos > (scf_options::M2, R2, scf_options::nc2, scf_options::ne2, scf_options::core_frac2, scf_options::mu2);
			eos1 = std::make_shared < accretor_eos > (scf_options::M1, scf_options::nc2, *eos2);
		} else {
			eos1 = std::make_shared < accretor_eos > (scf_options::M1, R1, scf_options::nc1, scf_options::ne1, scf_options::core_frac1, scf_options::mu1);
			if (contact > 0.0) {
				eos2 = std::make_shared < donor_eos > (scf_options::M2, R2, scf_options::nc2, scf_options::ne2, scf_options::mu2, *eos1);
			} else {
				eos2 = std::make_shared < donor_eos > (scf_options::M2, R2, scf_options::nc2, scf_options::ne2, scf_options::core_frac2, scf_options::mu2);
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

real grid::scf_update(real com, real omega, real c1, real c2, real c1_x, real c2_x, real l1_x, accretor_eos eos_1, donor_eos eos_2) {
	if (omega <= 0.0) {
		printf("OMEGA <= 0.0\n");
		abort();
	}
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
	//			const real fx = G[gx_i][iiig] + (x - com) * std::pow(omega, 2);
	//			const real fy = G[gy_i][iiig] + y * std::pow(omega, 2);
	//			const real fz = G[gz_i][iiig];

				real fx = -(G[phi_i][iiig + G_DNX] - G[phi_i][iiig - G_DNX]) / (2.0 * dx);
				real fy = -(G[phi_i][iiig + G_DNY] - G[phi_i][iiig - G_DNY]) / (2.0 * dx);
				real fz = -(G[phi_i][iiig + G_DNZ] - G[phi_i][iiig - G_DNZ]) / (2.0 * dx);
				fx -= (std::pow(x - com + dx, 2.0) - std::pow(x - com - dx, 2.0)) * std::pow(omega, 2) / (2.0 * dx);
				fy -= ((y + dx) * (y + dx) - (y - dx) * (y - dx)) * std::pow(omega, 2) / (2.0 * dx);

				bool is_donor_side = x > l1_x + com;
				real C = is_donor_side ? c2 : c1;
				real x0 = is_donor_side ? c2_x : c1_x;
				real g = (x - x0 - com) * fx + y * fy + z * fz;
				auto this_eos = is_donor_side ? eos_2 : eos_1;

				real new_rho, eint;
				const auto smallest = 1.0e-20;
				if (g < 0.0) {
					ASSERT_NONAN(phi_eff);
					ASSERT_NONAN(C);
					new_rho = std::max(this_eos.enthalpy_to_density(std::max(C - phi_eff, smallest)), rho_floor);
				} else {
					new_rho = rho_floor;
				}
				ASSERT_NONAN(new_rho);
				rho = std::max((1.0 - w0) * rho + w0 * new_rho, rho_floor);
				eint = std::max(ei_floor, this_eos.pressure(rho) / (fgamma - 1.0));

				U[rho_i][iiih] = rho;
				const real rho0 = rho - rho_floor;
				U[spc_ac_i][iiih] = rho > this_eos.dE() ? (is_donor_side ? 0.0 : rho0) : 0.0;
				U[spc_dc_i][iiih] = rho > this_eos.dE() ? (is_donor_side ? rho0 : 0.0) : 0.0;
				U[spc_ae_i][iiih] = rho <= this_eos.dE() ? (is_donor_side ? 0.0 : rho0) : 0.0;
				U[spc_de_i][iiih] = rho <= this_eos.dE() ? (is_donor_side ? rho0 : 0.0) : 0.0;
				U[spc_vac_i][iiih] = rho_floor;

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
	char* ptr;
	real omega = initial_params().omega;
	real jorb0;
	grid::set_omega(omega);
	for (integer i = 0; i != 250; ++i) {
		if (asprintf(&ptr, "X.scf.%i.silo", int(i)))
			;
		auto& params = initial_params();
		//	set_omega_and_pivot();
		if (i % 5 == 0)
			output(ptr);
		free(ptr);
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
		real j1 = is1 * omega;
		real j2 = is2 * omega;
		real jorb = iorb * omega;
		if (i == 0) {
			jorb0 = jorb;
		}
		real spin_ratio = (j1 + j2) / (jorb);
		real this_m = diags.grid_sum[rho_i];
		solve_gravity(false);

		auto axis = grid_ptr->find_axis();
		auto loc = line_of_centers(axis);

		real l1_x, c1_x, c2_x, l2_x, l3_x;
		real l1_phi, l2_phi, l3_phi;

		real com = axis.second[0];
		real new_omega;
		new_omega = jorb0 / iorb;
		omega = new_omega;
		std::pair<real, real> rho1_max;
		std::pair<real, real> rho2_max;
		std::pair<real, real> l1_phi_pair;
		std::pair<real, real> l2_phi_pair;
		std::pair<real, real> l3_phi_pair;
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
		params.eos2->set_d0(rho2 * f1);
		if (scf_options::equal_eos) {
			params.eos1->set_d0_using_eos(rho1 * f0, *(params.eos2));
		} else {
			params.eos1->set_d0(rho1 * f0);
		}

		real h_1 = params.eos1->h0();
		real h_2 = params.eos2->h0();

		real c_1, c_2;
		if (scf_options::equal_eos) {
			const real alo2 = 1.0 - scf_options::fill2;
			const real ahi2 = scf_options::fill2;
			c_2 = phi_2 * alo2 + ahi2 * l1_phi;
			c_1 = params.eos1->h0() + phi_1;
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
		if (!scf_options::equal_eos) {
			params.eos1->set_h0(c_1 - phi_1);
		}
		params.eos2->set_h0(c_2 - phi_2);
		auto e1 = params.eos1;
		auto e2 = params.eos2;

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
		if (!scf_options::equal_eos) {
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
		if (i % 5 == 0)
			printf("%13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s\n", "rho1", "rho2", "M1", "M2",
				"omega", "virial", "core_frac_1", "core_frac_2", "jorb", "c1_x", "c2_x", "jtot", "etot", "spin_ratio", "g1", "g2", "e1f", "e2f", "iorb", "is1",
				"is2");
		lprintf("log.txt", "%13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e %13e\n", rho1, rho2, M1, M2,
			omega, virial, core_frac_1, core_frac_2, jorb, c1_x, c2_x, j1 + j2 + jorb, etot, spin_ratio, g1, g2, e1f, e2f, iorb, is1, is2);
		if (i % 10 == 0) {
			regrid(me.get_gid(), false);
		}
		grid::set_omega(omega);

		scf_update(com, omega, c_1, c_2, rho1_max.first, rho2_max.first, l1_x, *e1, *e2);
		solve_gravity(false);

	}
}

std::vector<real> scf_binary(real x, real y, real z, real dx) {
	std::vector<real> u(NF, real(0));
	static auto& params = initial_params();
	std::shared_ptr<bipolytropic_eos> this_eos;
	real rho, r, ei;
	if (x < params.l1_x) {
		r = std::sqrt(std::pow(x - params.c1_x, 2) + y * y + z * z);
		this_eos = std::dynamic_pointer_cast < bipolytropic_eos > (params.eos1);
	} else {
		r = std::sqrt(std::pow(x - params.c2_x, 2) + y * y + z * z);
		this_eos = std::dynamic_pointer_cast < bipolytropic_eos > (params.eos2);
	}
	rho = std::max(this_eos->density_at(r, dx), rho_floor);
	ei = this_eos->pressure(rho) / (fgamma - 1.0);
	u[rho_i] = rho;
	u[spc_ac_i] = rho > this_eos->dE() ? (x > params.l1_x ? 0.0 : rho) : 0.0;
	u[spc_dc_i] = rho > this_eos->dE() ? (x > params.l1_x ? rho : 0.0) : 0.0;
	u[spc_ae_i] = rho <= this_eos->dE() ? (x > params.l1_x ? 0.0 : rho) : 0.0;
	u[spc_de_i] = rho <= this_eos->dE() ? (x > params.l1_x ? rho : 0.0) : 0.0;
//	if( rho > rho_floor)
//	printf( "%e %e %e %e\n", rho, this_eos->dE(), this_eos->dC(), this_eos->d0() );
//	printf( "%e %e\n", this_eos->m0(), this_eos->r0());

	//if( rho < this_eos->dE() && rho > rho_floor) {
//	}
	u[egas_i] = ei + 0.5 * (x * x + y * y) * params.omega * params.omega;
	u[sx_i] = -y * params.omega * rho;
	u[sy_i] = +x * params.omega * rho;
	u[sz_i] = 0.0;
	u[tau_i] = std::pow(ei, 1.0 / fgamma);
	return u;
}
