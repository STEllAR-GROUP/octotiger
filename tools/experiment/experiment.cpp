#include <hpx/hpx_init.hpp>

#include "hpxfft.hpp"

using complex = std::complex<double>;

// constexpr double c = 2.99792458e10;
constexpr int N = 16;
constexpr double L = 8.0;
constexpr double dx = L / N;
constexpr double V = L * L * L;

constexpr double c = 1.0;
constexpr double E0 = 1.0;
constexpr double t0 = 1.0;
constexpr double χ = 1.0;
constexpr double v = c * std::numbers::inv_sqrt3;
constexpr double α = c * χ * 0.5;
constexpr double α2 = α * α;
constexpr double v2 = v * v;
constexpr double D = c / (3.0 * χ);

double Eh(double r) {
	return E0 * std::exp(-0.25 * r * r / (D * t0));
}

double fft_k(int i) {
	int m = (i <= N / 2) ? i : i - N;
	return 2.0 * M_PI * double(m) / L;
}

auto computeEi(double k, double t) {
	double const Ei0 = E0 * std::pow(4.0 * M_PI * D * t0, 1.5);
	double const k2 = k * k;
	double const disc = α2 - v2 * k2;
	double const e = std::exp(-(D * t0 * k2 + α * t));
	complex E, dEdt;
	if (disc > 0.0) {
		double const λ = std::sqrt(disc);
		double const c = std::cosh(λ * t);
		double const s = std::sinh(λ * t);
		E = Ei0 * e * (c + (α / λ) * s);
		dEdt = Ei0 * e * (λ * s + α * c) - α * E;
	} else if (disc < 0.0) {
		double const μ = std::sqrt(-disc);
		double const c = std::cos(μ * t);
		double const s = std::sin(μ * t);
		E = Ei0 * e * (c + (α / μ) * s);
		dEdt = Ei0 * e * (α * c - μ * s) - α * E;
	} else {
		E = Ei0 * e * (1.0 + α * t);
		dEdt = α * (Ei0 * e - E);
	}
	return std::pair(E, dEdt);
}

auto computeEr(int xi, int yi, int zi, double t) {
	constexpr auto N3 = N * N * N;
	constexpr auto I = complex(0, 1);
	static IFFT ifft(std::vector<size_t>({N, N, N}));
	static double ct = INFINITY;
	static std::vector<complex> Er(N3);
	static std::array<std::vector<complex>, 3> Fr;
	for (auto &fr : Fr) {
		if (fr.size() == N3) continue;
		fr.resize(N3);
	}
	if (t != ct) {
		std::vector<complex> Ei(N3);
		std::array<std::vector<complex>, 3> Fi;
		for (auto &fi : Fi) {
			fi.resize(N3);
		}
		auto const norm = double(N3) / V;
		for (int ki = 0; ki < N; ki++) {
			for (int kj = 0; kj < N; kj++) {
				for (int kk = 0; kk < N; kk++) {
					auto const kx = fft_k(ki);
					auto const ky = fft_k(kj);
					auto const kz = fft_k(kk);
					auto const k = std::sqrt(kx * kx + ky * ky + kz * kz);
					auto const [E, dEdt] = computeEi(k, t);
					auto const index = kk + N * (kj + N * ki);
					Ei[index] = E * norm;
					auto const k2 = k * k;
					if (k2) {
						auto const ik2 = 1.0 / k2;
						Fi[0][index] = I * dEdt * kx * ik2 * norm;
						Fi[1][index] = I * dEdt * ky * ik2 * norm;
						Fi[2][index] = I * dEdt * kz * ik2 * norm;
					} else {
						Fi[0][index] = Fi[1][index] = Fi[2][index] = 0.0;
					}
				}
			}
		}
		Er = ifft(Ei).get();
		for (int d = 0; d < 3; d++) {
			Fr[d] = ifft(Fi[d]).get();
		}
		ct = t;
	}
	auto const index = zi + N * (yi + N * xi);
	auto const F = std::array<double, 3>({Fr[0][index].real(), Fr[1][index].real(), Fr[2][index].real()});
	auto const E = Er[index].real();
	return std::pair(E, F);
}

auto hpx_main() {
	double t = 1;
	for (int xi = 0; xi < N; xi++) {
		for (int yi = 0; yi < N; yi++) {
			for (int zi = 0; zi < N; zi++) {
				auto const [E, F] = computeEr(xi, yi, zi, t);
				printf("%i %i %i %e %e %e %e\n", xi, yi, zi, E, F[0], F[1], F[2]);
			}
		}
	}
	return hpx::finalize();
}

int main(int argc, char **argv) {
	return hpx::init(argc, argv);
}
