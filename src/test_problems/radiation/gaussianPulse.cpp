#include <memory>

#include <hpx/hpx_init.hpp>

#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/hpxfft/hpxfft.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/radiation/opacities.hpp"

using complex = std::complex<Real>;

static Real c;
static Real centralRadiationEnergyDensity;
static Real centralInteralEnergyDensity;
static Real centralDensity;
static Real timeOffset;
static Real extinctionCoefficient;

static bool initialized = false;

static void initialize() {
	if (initialized) return;
	initialized = true;
	CgsToCode const convert;
	auto const gamma = opts().gas_gamma;
	c = physcon().c;
	centralRadiationEnergyDensity = convert.energyDensity(opts().rad_diff_Er0);
	timeOffset = convert.time(opts().rad_diff_t0);
	centralDensity = convert.massDensity(opts().rad_diff_rho0);
	centralInteralEnergyDensity = centralRadiationEnergyDensity * 1e-20;
	extinctionCoefficient = centralDensity * (opacityAbsorption(centralDensity, 0_R) + opacityScattering(centralDensity, 0_R));
}

static Real Eh(Real r) {
	Real const D = c / (3_R * extinctionCoefficient);
	return centralRadiationEnergyDensity * std::exp(-0.25 * r * r / (D * timeOffset));
}

static Vector<Real, NDIM> fft_k(int xi, int xj, int xk) {
	Vector<Real, NDIM> k;
	CgsToCode const convert;
	auto const L = convert.length(2.0 * opts().xscale);
	auto const N = OCTOTIGER_GRIDDIM << opts().max_level;
	xi = (2 * xi <= N) ? xi : (xi - N);
	xj = (2 * xj <= N) ? xj : (xj - N);
	xk = (2 * xk <= N) ? xk : (xk - N);
	k[0] = 2.0 * M_PI * Real(xi) / L;
	k[1] = 2.0 * M_PI * Real(xj) / L;
	k[2] = 2.0 * M_PI * Real(xk) / L;
	return k;
}

static auto compute_ErHat(Real k, Real t) {
	FpeGuard fpeGuard{};
	Real const D = c / (3_R * extinctionCoefficient);
	Real const v = c * std::numbers::inv_sqrt3;
	Real const α = c * extinctionCoefficient * 0.5_R;
	Real const ErHat0 = centralRadiationEnergyDensity * std::pow(4_R * pi_R * D * timeOffset, 1.5_R);
	Real const k2 = k * k;
	Real const disc = sqr(α) - sqr(v) * sqr(k);
	Real const e = std::exp(-(D * timeOffset * sqr(k) + α * t));
	complex E, dEdt;
	if (disc > 0_R) {
		Real const λ = std::sqrt(disc);
		Real const ap = α + λ;
		Real const am = sqr(v) * k2 / ap;
		Real const q = α / λ;
		Real const g = std::exp(-D * timeOffset * k2);
		Real const e1 = std::exp(-am * t);
		Real const e2 = std::exp(-ap * t);
		E = ErHat0 * g * 0.5_R * ((1_R + q) * e1 + (1_R - q) * e2);
		dEdt = -ErHat0 * g * 0.5_R * (am * (1_R + q) * e1 + ap * (1_R - q) * e2);
	} else if (disc < 0_R) {
		Real const μ = std::sqrt(-disc);
		Real const c = std::cos(μ * t);
		Real const s = std::sin(μ * t);
		E = ErHat0 * e * (c + (α / μ) * s);
		dEdt = ErHat0 * e * (α * c - μ * s) - α * E;
	} else {
		E = ErHat0 * e * (1_R + α * t);
		dEdt = α * (ErHat0 * e - E);
	}
	return std::pair(E, dEdt);
}

static auto computeEr(int xi, int yi, int zi, Real t) {
	FpeGuard fpeGuard{};
	constexpr auto I = complex(0, 1);
	static hpx::mutex mtx_;
	static Real ct = INFINITY;
	static std::vector<complex> Er;
	static Vector<std::vector<complex>, NDIM> Fr;
	CgsToCode const convert;
	auto const L = convert.length(2.0 * opts().xscale);
	size_t const N = OCTOTIGER_GRIDDIM << opts().max_level;
	auto const x0 = Vector<Real, NDIM>({-0.5_R * L, -0.5_R * L, -0.5_R * L});
	auto const N3 = N * N * N;
	{
		bool doFFt;
		std::lock_guard<hpx::mutex> guard(mtx_);
		doFFt = bool(t != ct);
		if (Er.size() != N3) Er.resize(N3);
		for (auto &fr : Fr) {
			if (fr.size() == N3) continue;
			fr.resize(N3);
		}
		if (doFFt) {
			IFFT ifft(std::vector<size_t>({N, N, N}));
			Real const dx = L / N;
			Real const V = sqr(L) * L;
			std::vector<complex> Ehat(N3);
			Vector<std::vector<complex>, NDIM> Fhat;
			for (auto &fhat : Fhat) {
				fhat.resize(N3);
			}
			auto const norm = Real(N3) / V;
			for (int ki = 0; ki < N; ki++) {
				for (int kj = 0; kj < N; kj++) {
					for (int kk = 0; kk < N; kk++) {
						auto const k = fft_k(ki, kj, kk);
						auto [E, dEdt] = compute_ErHat(abs(k), t);
						auto const phase = std::exp(I * k.dot(x0));
						E *= phase;
						dEdt *= phase;
						auto const index = kk + N * (kj + N * ki);
						Ehat[index] = E * norm;
						auto const k2 = k.dot(k);
						if (k2) {
							auto const ik2 = 1_R / k2;
							for (int d = 0; d < NDIM; d++) {
								Fhat[d][index] = I * dEdt * k[d] * ik2 * norm;
							}
						} else {
							Fhat[0][index] = Fhat[1][index] = Fhat[2][index] = 0_R;
						}
					}
				}
			}
			Er = ifft(Ehat).get();
			for (int d = 0; d < NDIM; d++) {
				Fr[d] = ifft(Fhat[d]).get();
			}
			auto const center = N / 2 + N * (N / 2 + N * (N / 2));

			printf("FFT center = %e\n", Er[center].real());
			printf("Exact center = %e\n", centralRadiationEnergyDensity);
			ct = t;
		}
	}
	// Assumes all analytic evaluations for a given t complete before evaluations
	// for a different t begin. Therefore Er/Fr are not read concurrently with
	// a recomputation for another t.
	auto const index = zi + N * (yi + N * xi);
	auto const F = Vector<Real, NDIM>({Fr[0][index].real(), Fr[1][index].real(), Fr[2][index].real()});
	auto const E = Er[index].real();
	return std::pair(E, F);
}

constexpr int periodicWrap(int i, int n) {
	while (i < 0) {
		i += n;
	}
	while (i >= n) {
		i -= n;
	}
	return i;
};

std::vector<real> analyticGaussianPulse(real x_, real y_, real z_, real t) {
	FpeGuard fpeGuard{};
	{
		static hpx::mutex mtx_;
		std::lock_guard<hpx::mutex> guard(mtx_);
		initialize();
	}
	CgsToCode const convert;
	auto const gamma = opts().gas_gamma;
	std::vector<real> G(opts().n_fields, 0_R);
	std::vector<real> R(NRF, 0_R);
	auto const L = convert.length(2.0 * opts().xscale);
	x_ = convert.length(x_);
	y_ = convert.length(y_);
	z_ = convert.length(z_);
	auto const N = OCTOTIGER_GRIDDIM << opts().max_level;
	auto const dx = L / N;
	size_t const xi = size_t(periodicWrap((x_ + 0.5 * L) / dx, N));
	size_t const yi = size_t(periodicWrap((y_ + 0.5 * L) / dx, N));
	size_t const zi = size_t(periodicWrap((z_ + 0.5 * L) / dx, N));
	auto [E, F] = computeEr(xi, yi, zi, t);
	auto const Emin = 1.0e-10 * centralRadiationEnergyDensity;
	if (E < Emin) {
		E = Emin;
		F = 0_R;
	}
	F /= c;
	auto const F2 = F.dot(F);
	auto const E2 = sqr(E);
	Real theta = 0.9_R;
	//	auto const r = sqrt(sqr(x_-0.5) + sqr(y_-0.5) + sqr(z_-0.5));
	//	printf( "%e %e\n", r, sqrt(F2/E2));
	if (F2 > sqr(theta) * E2) {
		theta *= sqrt(E2 / F2);
		F *= theta;
	}
	ASSERT_NONNEGATIVE(E);
	G[spc_i] = G[rho_i] = centralDensity;
	G[egas_i] = centralInteralEnergyDensity;
	G[tau_i] = gasEnergy2Entropy(centralDensity, centralInteralEnergyDensity);
	R[0] = E;
	R[1] = F[0];
	R[2] = F[1];
	R[3] = F[2];
	G.insert(G.end(), R.begin(), R.end());
	return G;
}

std::vector<real> testGaussianPulse(Real x, Real y, Real z, Real dx) {
	return analyticGaussianPulse(x, y, z, 0_R);
}
