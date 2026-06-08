#include "octotiger/grid.hpp"
#include "octotiger/math/AutoDiff.hpp"
#include "octotiger/radiation/rad_grid.hpp"

#include <numeric>

RadiationSource radiationSource(real E0, Vector<real, NDIM> F0, real τ0, Vector<real, NDIM> S0, real ρ, real μ, real κ, real χ, real dt) {
	RadiationSource update;
	auto const c = physcon().c;
	auto const ic = inv(c);
	auto const c2 = sqr(c);
	auto const kB = physcon().kb;
	auto const aR = physcon().sigma * 4_R * ic;
	auto const m = physcon().mh;
	auto const Γ = grid::get_fgamma();
	auto const ic2 = sqr(ic);
	auto const Rg = m * (Γ - 1_R) * inv(kB);

	auto const iρ = inv(ρ);
	auto const e0 = std::pow(τ0, Γ);
	auto const Eg0 = e0 + 0.5_R * iρ * S0.dot(S0);
	auto const β = iρ * ic * S0;
	auto const σ = χ - κ;
	auto const A = aR * sqr(sqr(iρ * μ * Rg));
	auto const tol2 = sqr(e0 + E0) * std::numeric_limits<real>::epsilon();
	real dx;
	real x = 0_R;
	Vector<real, NDIM> F;
	bool &converged = update.converged;
	auto &cnt = update.iters;
	do {
		auto const E = E0 + x;
		auto const e = e0 - x;
		auto const B = A * sqr(sqr(e));
		auto const dBdx = -4_R * A * e * sqr(e);
		F = (F0 + dt * c2 * (β * (σ * E + κ * B))) * inv(1_R + dt * c * χ);
		auto const dFdx = dt * c2 * β * (σ + κ * dBdx) * inv(1_R + dt * c * χ);
		auto const f = x + dt * (c * κ * (E - B) + (σ - κ) * F.dot(β));
		auto const dfdx = 1_R + dt * (c * κ * (1_R - dBdx) + (σ - κ) * dFdx.dot(β));
		dx = -f * inv(dfdx);
		x += dx;
		cnt++;
		converged = (sqr(dx) < tol2);
	} while (!converged && (cnt < 20));
	auto const E = E0 + x;
	auto const e = e0 - x;
	auto const τ = pow(e, inv(Γ));
	auto const S = S0 + ic2 * (F0 - F);
	auto const Eg = e + 0.5_R * iρ * S.dot(S);
	update.dEr_dt = (E - E0) * inv(dt);
	update.dEg_dt = (Eg - Eg0) * inv(dt);
	update.dFr_dt = (F - F0) * inv(dt);
	update.dS_dt = (S - S0) * inv(dt);
	update.dtau_dt = (τ - τ0) * inv(dt);
	return update;
}

void test_rad_imp_cell() {
	auto const c = physcon().c;
	auto const ic2 = inv(sqr(c));
	auto const aR = physcon().sigma * inv(c) * 4_R;
	auto const fgamma = grid::get_fgamma();
	auto const kb = physcon().kb;
	auto const mh = physcon().mh;
	auto const mu = 1_R;

	auto norm = [](Vector<real, NDIM> const &v) {
		return sqrt(v.dot(v));
	};

	auto makeS = [&](real rho, real beta) {
		Vector<real, NDIM> S;
		for (auto d = 0; d < NDIM; d++) {
			S[d] = 0_R;
		}
		S[0] = rho * beta * c;
		return S;
	};

	auto makeGasEnergy = [&](real rho, real T, Vector<real, NDIM> const &S) {
		auto const Ei = rho * kb * T / (mu * mh * (fgamma - 1_R));
		auto const Ek = 0.5_R * S.dot(S) / rho;
		return Ei + Ek;
	};

	struct Test {
		char const *name;
		char const *why;
		real rho;
		real T;
		real ErFactor;
		real fluxFrac;
		real betaGas;
		real kappa;
		real chi;
		real dtFactor;
	};

	std::vector<Test> tests = {
		{"LTE fixed point", "Near LTE with tiny nonzero flux. Should barely change.", 1.0e-15_R, 1.0e4_R, 1.0_R, 1.0e-12_R, 0.0_R, 1.0e-8_R, 1.0e-8_R, 1.0e-3_R},
		{"Thin weak coupling", "Small c*kappa*dt and c*chi*dt. Should move very little.", 1.0e-15_R, 1.0e4_R, 2.0_R, 0.2_R, 0.0_R, 1.0e-12_R, 1.0e-12_R, 1.0e-4_R},
		{"Thick radiation hot", "Strong absorption with Er > aT^4. Radiation energy should decrease.", 1.0e-15_R, 1.0e4_R, 10.0_R, 1.0e-10_R, 0.0_R, 1.0e-8_R, 1.0e-8_R, 1.0_R},
		{"Thick gas hot", "Strong absorption with Er < aT^4. Radiation energy should increase.", 1.0e-15_R, 1.0e4_R, 0.1_R, 1.0e-10_R, 0.0_R, 1.0e-8_R, 1.0e-8_R, 1.0_R},
		{"Pure-ish flux damping", "chi large, kappa tiny. Flux should damp with little thermal change.", 1.0e-15_R, 1.0e4_R, 1.0_R, 0.8_R, 0.0_R, 1.0e-16_R, 1.0e-8_R, 10.0_R},
		{"High gas momentum", "Tests beta terms and momentum exchange.", 1.0e-15_R, 1.0e4_R, 1.0_R, 0.7_R, 1.0e-1_R, 1.0e-8_R, 1.0e-8_R, 1.0_R},
		{"Near free streaming", "Tests f close to one.", 1.0e-15_R, 1.0e4_R, 1.0_R, 0.95_R, 0.0_R, 1.0e-8_R, 1.0e-8_R, 1.0_R},
		{"Very stiff thermal exchange", "Large c*kappa*dt and large thermal disequilibrium. Tests nonlinear T^4 stiffness.", 1.0e-15_R, 1.0e4_R, 100.0_R, 1.0e-10_R, 0.0_R, 1.0e-8_R, 1.0e-8_R, 100.0_R},
		{"Fast gas, beamed radiation", "Large beta and large f together; stresses beta dot F.", 1.0e-15_R, 1.0e4_R, 2.0_R, 0.95_R, 0.2_R, 1.0e-8_R, 1.0e-8_R, 1.0_R},
		{"Very stiff flux damping", "Large c*chi*dt. Flux should be strongly damped.", 1.0e-15_R, 1.0e4_R, 1.0_R, 0.5_R, 0.0_R, 1.0e-16_R, 1.0e-8_R, 100.0_R}
	};

	printf("\n================ rad_imp_cell pointwise tests ================\n");

	for (auto const &test : tests) {
		auto const ErLTE = aR * sqr(sqr(test.T));
		auto const Er0 = test.ErFactor * ErLTE;

		auto const S0 = makeS(test.rho, test.betaGas);
		auto const Eg0 = makeGasEnergy(test.rho, test.T, S0);
		auto const Ei0 = Eg0 - 0.5_R * S0.dot(S0) / test.rho;
		auto const tau0 = pow(Ei0, inv(fgamma));

		Vector<real, NDIM> Fphys0;
		for (auto d = 0; d < NDIM; d++) {
			Fphys0[d] = 0_R;
		}
		Fphys0[0] = test.fluxFrac * c * Er0;

		auto const dt = test.dtFactor / (c * std::max(test.kappa, test.chi));

		printf("\n---------------------------------------------------------------\n");
		printf("TEST: %s\n", test.name);
		printf("WHY:  %s\n", test.why);
		printf("before call: rho=%.6e T=%.6e Er/E_LTE=%.6e f=%.6e beta=%.6e\n",
			double(test.rho), double(test.T), double(test.ErFactor),
			double(test.fluxFrac), double(test.betaGas));
		printf("coupling: kappa=%.6e chi=%.6e c*kappa*dt=%.6e c*chi*dt=%.6e\n",
			double(test.kappa), double(test.chi),
			double(c * test.kappa * dt), double(c * test.chi * dt));

		auto const Etot0 = Eg0 + Er0;

		Vector<real, NDIM> Mtot0;
		for (auto d = 0; d < NDIM; d++) {
			Mtot0[d] = S0[d] + Fphys0[d] * ic2;
		}

		auto const update = radiationSource(
			Er0, Fphys0, tau0, S0,
			test.rho, mu, test.kappa, test.chi, dt
		);

		auto const Er1 = Er0 + update.dEr_dt * dt;
		auto const Eg1 = Eg0 + update.dEg_dt * dt;

		Vector<real, NDIM> Fphys1;
		Vector<real, NDIM> S1;

		for (auto d = 0; d < NDIM; d++) {
			Fphys1[d] = Fphys0[d] + update.dFr_dt[d] * dt;
			S1[d] = S0[d] + update.dS_dt[d] * dt;
		}

		auto const Etot1 = Eg1 + Er1;

		Vector<real, NDIM> Mtot1;
		for (auto d = 0; d < NDIM; d++) {
			Mtot1[d] = S1[d] + Fphys1[d] * ic2;
		}

		real momErrMax = 0_R;
		for (auto d = 0; d < NDIM; d++) {
			momErrMax = std::max(momErrMax, std::abs(Mtot1[d] - Mtot0[d]));
		}

		auto const f0 = norm(Fphys0) / (c * Er0);
		auto const f1 = norm(Fphys1) / (c * Er1);

		printf("converged=%d iter=%d\n", int(update.converged), int(update.iters));
		printf("Er: %.17e -> %.17e   dEr=%.17e\n", double(Er0), double(Er1), double(update.dEr_dt * dt));
		printf("Eg: %.17e -> %.17e\n", double(Eg0), double(Eg1));
		printf("|Fphys|: %.17e -> %.17e\n", double(norm(Fphys0)), double(norm(Fphys1)));
		printf("f=|F|/(cEr): %.17e -> %.17e\n", double(f0), double(f1));
		printf("energy conservation error: %.17e\n", double(Etot1 - Etot0));
		printf("max momentum conservation error: %.17e\n", double(momErrMax));

		bool pass = true;
		pass = pass && update.converged;
		pass = pass && Er1 > 0_R;
		pass = pass && Eg1 > 0_R;
		pass = pass && f1 <= 1_R + 1.0e-10_R;
		pass = pass && std::abs(Etot1 - Etot0) <= 1.0e-10_R * (std::abs(Etot0) + 1_R);
		pass = pass && momErrMax <= 1.0e-10_R * (norm(Mtot0) + 1_R);

		printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
	}

	printf("\n===============================================================\n");
}