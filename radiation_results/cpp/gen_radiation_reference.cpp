// Copyright (c) 2026 AUTHORS. Distributed under the Boost Software License, Version 1.0.
// Serial FFTW3 only: no HPX, MPI, OpenMP or parallel FFT plans.
#include <stddef.h>
#include "octotiger/test_problems/radiation/reference.hpp"
#include <complex>
#include <fftw3.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace
{
struct fft_buffer {
	fftw_complex *data;
	explicit fft_buffer(size_t n) : data(fftw_alloc_complex(n))
	{
		if (!data) {
      throw std::bad_alloc();
    }
  }
	~fft_buffer() { fftw_free(data); }
	fft_buffer(fft_buffer const &) = delete;
};
struct fft_plan {
	fftw_plan plan;
	fft_plan(int n, fftw_complex *in, fftw_complex *out, int sign)
		: plan(fftw_plan_dft_3d(n, n, n, in, out, sign, FFTW_ESTIMATE))
	{
		if (!plan) {
      throw std::runtime_error("FFTW3 plan creation failed");
    }
  }
	~fft_plan() { fftw_destroy_plan(plan); }
	fft_plan(fft_plan const &) = delete;
};
double number(std::string const &text)
{
	size_t end;
	double const v = std::stod(text, &end);
	if (end != text.size() || !std::isfinite(v)) {
    throw std::runtime_error("Invalid numeric argument: " + text);
  }
  return v;
}
} // namespace
int main(int argc, char **argv)
{
	try {
		using namespace radiationTests;
		ReferenceData ref;
		ref.n = 32;
		std::string output = "gaussian_pulse.bin", model = "m1";
		for (int a = 1; a < argc; ++a) {
			std::string const key = argv[a];
			if (key == "--help") {
				std::cout << "gen_radiation_reference --output FILE --cells N --length L --c C "
							 "--chi CHI --width W --background E --amplitude A --time T --model m1|fld\n"
                             "m1: linearized moment/telegraph reference (default).\n"
                             "fld: periodic constant-D diffusion reference, D=c/(3*chi); chi>0.\n"
                             "The diffusion reference is exact for the represented heat equation,\n"
                             "and is a diffusion-limit reference for nonlinear flux-limited diffusion.\n";
				return 0;
			}
			if (++a == argc) {
        throw std::runtime_error("Missing value for " + key);
      }
      if (key == "--output") {
				output = argv[a];
				continue;
			}
            if (key == "--model") {
                model = argv[a];
                if (model != "m1" && model != "fld") {
                    throw std::runtime_error("--model must be m1 or fld");
                }
                continue;
            }
			double const v = number(argv[a]);
			if (key == "--cells") {
				if (v < 4 || v > 512 || std::floor(v) != v || int(v) % 2) {
          throw std::runtime_error("N must be even, 4 <= N <= 512");
        }
        ref.n = unsigned(v);
			} else if (key == "--length") {
        ref.p.length = v;
      } else if (key == "--c") {
        ref.p.c = v;
      } else if (key == "--chi") {
        ref.p.chi = v;
      } else if (key == "--width") {
        ref.p.width = v;
      } else if (key == "--background") {
        ref.p.background = v;
      } else if (key == "--amplitude") {
        ref.p.amplitude = v;
      } else if (key == "--time") {
        ref.p.time = v;
      } else {
        throw std::runtime_error("Unknown option: " + key);
      }
    }
		ref.p.validate();
        if (model == "fld" && !(ref.p.chi > 0 && ref.p.c / ref.p.chi / 3 > 0 &&
            std::isfinite(ref.p.c / ref.p.chi / 3))) {
            throw std::runtime_error("FLD diffusion reference requires chi>0 and finite D=c/(3*chi)");
        }
		int const n = ref.n;
		auto const N3 = ref.cells();
		auto const &p = ref.p;
		double const dx = p.length / n;
		fft_buffer spatial(N3), spectral(N3);
		fft_plan forward(n, spatial.data, spectral.data, FFTW_FORWARD);
		fft_plan inverse(n, spectral.data, spatial.data, FFTW_BACKWARD);
		// Exact cell averages of a periodized Gaussian. A forward FFT includes
		// the half-cell sampling phase automatically; no hand-written shift is needed.
		std::vector<double> gaussian(n);
		for (int i = 0; i < n; ++i) {
			double const x = -p.length / 2 + (i + .5) * dx;
			for (int image = -4; image <= 4; ++image) {
				double const a = (x - dx / 2 + image * p.length) / p.width;
				double const b = (x + dx / 2 + image * p.length) / p.width;
				gaussian[i] += .5 * std::sqrt(pi) * p.width / dx * (std::erf(b) - std::erf(a));
			}
		}
		for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
					auto const r = k + size_t(n) * (j + size_t(n) * i);
					spatial.data[r][0] = p.amplitude * gaussian[i] * gaussian[j] * gaussian[k];
					spatial.data[r][1] = 0;
				}
      }
    }
    fftw_execute(forward.plan);
		std::vector<std::complex<double>> initial(N3);
		for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
					auto const r = k + size_t(n) * (j + size_t(n) * i);
					// A Nyquist mode has no distinct conjugate on an even grid. Remove
					// those planes so odd spectral derivatives (physical F) stay real.
					if (i != n / 2 && j != n / 2 && k != n / 2) {
            initial[r] = {spectral.data[r][0], spectral.data[r][1]};
          }
        }
      }
    }
    ref.values.resize(8 * N3);
		double max_imag = 0;
		for (int snapshot = 0; snapshot < 2; ++snapshot) {
      for (int field = 0; field < 4; ++field) {
				double const t = snapshot == 0 ? 0 : p.time;
				for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
							auto const r = k + size_t(n) * (j + size_t(n) * i);
							auto freq = [&](int a) {
								return 2 * pi * (a < n / 2 ? a : a - n) / p.length;
							};
							Point const wave{freq(i), freq(j), freq(k)};
							double const k2 =
								wave[0] * wave[0] + wave[1] * wave[1] + wave[2] * wave[2];
                            std::complex<double> factor;
                            if (model == "fld") {
                                // The same represented initial energy evolves by the
                                // periodic heat equation. Flux is a derived gradient,
                                // including at t=0; it is not a telegraph moment.
                                const double diffusion = p.c / p.chi / 3;
                                const double decay = std::exp(-diffusion * k2 * t);
                                factor = field == 0 ? std::complex<double>(decay, 0) :
                                    std::complex<double>(0, -diffusion * wave[field - 1] * decay);
                            } else {
                                auto const mode = telegraphMode(k2, t, p.c, p.chi);
                                // Fhat = i k Ehat_dot/k^2 = -i (c^2/3) k B Ehat0.
                                factor = field == 0 ? std::complex<double>(mode[0], 0) :
                                    std::complex<double>(0, -p.c * p.c / 3 * wave[field - 1] * mode[1]);
                            }
							auto const value = initial[r] * factor;
							spectral.data[r][0] = value.real();
							spectral.data[r][1] = value.imag();
						}
          }
        }
        fftw_execute(inverse.plan);
				for (size_t r = 0; r < N3; ++r) {
					// FFTW's backward transform is unnormalized: divide exactly once.
					ref.values[(snapshot * 4 + field) * N3 + r] =
						spatial.data[r][0] / N3 + (field == 0 ? p.background : 0);
					max_imag = std::max(max_imag, std::abs(spatial.data[r][1] / N3));
				}
			}
    }
    if (max_imag > 1e-11 * p.amplitude * std::max(1., p.c)) {
      throw std::runtime_error("Reference lost Hermitian symmetry");
    }
    // Never hide an invalid reference by clipping energy or reduced flux.
		for (int s = 0; s < 2; ++s) {
      for (size_t r = 0; r < N3; ++r) {
				double const E = ref.values[s * 4 * N3 + r];
				double const F =
					std::hypot(ref.values[(s * 4 + 1) * N3 + r], ref.values[(s * 4 + 2) * N3 + r],
							   ref.values[(s * 4 + 3) * N3 + r]);
				if (!(E > 0 && F <= p.c * E)) {
          throw std::runtime_error("Reference requires positive E and |F|<=cE; increase "
                                  "background or opacity, or reduce amplitude");
        }
      }
    }
    ref.write(output);
		std::cout << "Wrote " << output << ": " << n << "^3 cells, physical E/F at t=0 and "
				  << std::setprecision(16) << p.time << "; " << (model == "fld" ?
                      "periodic constant-D diffusion-limit reference" : "linearized M1 telegraph reference") << "\n";
		return 0;
	} catch (std::exception const &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
