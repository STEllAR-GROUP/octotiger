/*
 * SphericalExpansion.hpp
 *
 *  Created on: Feb 3, 2016
 *      Author: dmarce1
 */

#ifndef SOLIDHARMONIC_HPP_
#define SOLIDHARMONIC_HPP_

#include <cassert>
#include <array>
#include <cmath>
#include <complex>

using real = double;

template<int N>
class factorial_table {
private:
	std::array<int, N + 1> f;
public:
	factorial_table() {
		f[0] = 1;
		for (int n = 1; n <= N; ++n) {
			f[n] = n * f[n - 1];
		}
	}
	inline const int& operator()(int n) const {
		return f[n];
	}
};

template<int N>
class spherical_expansion {
private:
	constexpr static int NN = N * N;
	std::array<std::complex<real>, NN> a;
	static real factorial(int n) {
		static const factorial_table<2 * (N - 1)> factor;
		return real(factor[n]);
	}
public:
	inline std::complex<real>& at(int l, int m) {
		assert(l >= 0);
		assert(l < N);
		assert(m <= l);
		assert(m >= 0);
		return a[(l * l + l) / 2 + m];
	}
	inline std::complex<real> operator()(int l, int m) const {
		assert(l >= 0);
		assert(l < N);
		assert(m <= +l);
		assert(m >= -l);
		if (m >= 0) {
			return a[(l * l + l) / 2 + m];
		} else {
			const auto b = a[(l * l + l) / 2 - m];
			const auto a = std::pow(-1, m);
			const auto c = a * b;
			return std::conj(c);
		}
	}
	inline void operator+=(const spherical_expansion<N>& other) {
		for (int l = 0; l < N; ++l) {
			for (int m = 0; m <= l; ++m) {
				this->at(l, m) += other(l, m);
			}
		}
	}
	template<class Archive>
	void serialize(Archive& arc, unsigned) {
		arc & a;
	}
};

template<int N>
class solidR: public spherical_expansion<N> {
public:
	solidR(real x, real y, real z) {
		static constexpr std::complex<real> ONE = std::complex < real > (real(1), real(0));
		//	static constexpr std::complex<real> ZERO = std::complex < real > (real(0), real(0));
		auto& R = *this;
		R.at(0, 0) = ONE;
		if (N > 0) {
			const real rho = std::sqrt(x * x + y * y);
			const real r = std::sqrt(rho * rho + z * z);
			const real cos0 = z / r;
			std::complex < real > eip;
			if (rho > real(0)) {
				eip = std::complex < real > (x / rho, y / rho);
			} else {
				eip = std::complex < real > (real(0), real(0));
			}
			if (N > 1) {
				R.at(1, 0) = ONE * z;
				R.at(1, 1) = -eip * (rho / real(2));
				for (int l = 2; l < N; ++l) {
					R.at(l, 0) = (real(2 * l - 1) * cos0 * R(l - 1, 0) - R(l - 2, 0) * r) * r / real(l * l);
					for (int m = 1; m <= l; ++m) {
						R.at(l, m) = -(r * r * R(l - 1, m - 1) - real(l - m + 1) * z * R(l, m - 1)) * eip
								/ (rho * real(l + m));
					}
				}
			}
		}
	}
};

template<int N>
class solidI: public spherical_expansion<N> {
public:
	solidI(real x, real y, real z) {
		auto& S = *this;
		static constexpr std::complex<real> ONE = std::complex < real > (real(1), real(0));
		static constexpr std::complex<real> ZERO = std::complex < real > (real(0), real(0));
		const real rho = std::sqrt(x * x + y * y);
		const real r = std::sqrt(rho * rho + z * z);
		if (r != 0.0) {
			const real r2inv = 1.0 / (r * r);
			S.at(0, 0) = ONE / r;
			if (N > 1) {
				const real cos0 = z / r;
				std::complex < real > eip;
				if (rho > real(0)) {
					eip = std::complex < real > (x / rho, y / rho);
				} else {
					eip = std::complex < real > (real(0), real(0));
				}
				const real r3inv = 1.0 / (r * r * r);
				S.at(1, 0) = ONE * z * r3inv;
				S.at(1, 1) = -eip * rho * r3inv;
				for (int l = 2; l < N; ++l) {
					S.at(l, 0) = real(2 * l - 1) * cos0 * S(l - 1, 0) / r;
					S.at(l, 0) -= S(l - 2, 0) * real(l * l - 2 * l + 1) / (r * r);
					for (int m = 1; m < l; ++m) {
						S.at(l, m) = (real(l - m) * z * S(l - 1, m) - real(l + m - 1) * eip * rho * S(l - 1, m - 1))
								* r2inv;
					}
					S.at(l, l) = (-real(2 * l - 1) * eip * rho * S(l - 1, l - 1)) * r2inv;
				}
			}
		} else {
			for (int l = 0; l < N; ++l) {
				for (int m = 0; m <= l; ++m) {
					S.at(l, m) = ZERO;
				}
			}
		}
	}
};

template<int N>
class local_expansion: public spherical_expansion<N> {
	static constexpr std::complex<real> ONE = std::complex < real > (real(1), real(0));
	static constexpr std::complex<real> ZERO = std::complex < real > (real(0), real(0));
public:
	local_expansion() :
			spherical_expansion<N>() {
		static constexpr std::complex<real> ZERO = std::complex < real > (real(0), real(0));
		for (int l = 0; l < N; ++l) {
			for (int m = 0; m <= l; ++m) {
				this->at(l, m) = ZERO;
			}
		}
	}
//	void L2P(real& phi, real& dphi_dx, real& dphi_dy, real& dphi_dz) {
//		phi = (*this)(0, 0);
//		dphi_dx = ((*this)(1, 1) + (*this)(1, -1)).real() / std::sqrt(2);
//		dphi_dy = ((*this)(1, 1) - (*this)(1, -1)).imag() / std::sqrt(2);
//		dphi_dz = (*this)(1, 0);
//	}
	void L2L(const local_expansion<N>& O, real dx, real dy, real dz) {
		solidR<N> Rlm(dx, dy, dz);
		for (int l0 = 0; l0 < N; ++l0) {
			for (int m0 = 0; m0 <= l0; ++m0) {
				for (int l1 = l0; l1 < N; ++l1) {
					for (int m1 = -l1; m1 <= l1; ++m1) {
						if (std::abs(m1 - m0) <= l1 - l0) {
							this->at(l0, m0) += O(l1, m1) * Rlm(l1 - l0, m1 - m0);
						}
					}
				}
			}
		}
	}
	template<class Archive>
	void serialize(Archive& arc, const unsigned v) {
		spherical_expansion<N>::serialize(arc, v);
	}
};

template<int N>
class multipole_expansion: public spherical_expansion<N> {
	static constexpr std::complex<real> ONE = std::complex < real > (real(1), real(0));
	static constexpr std::complex<real> ZERO = std::complex < real > (real(0), real(0));
public:
	multipole_expansion() :
			spherical_expansion<N>() {
		static constexpr std::complex<real> ZERO = std::complex < real > (real(0), real(0));
		for (int l = 0; l < N; ++l) {
			for (int m = 0; m <= l; ++m) {
				this->at(l, m) = ZERO;
			}
		}
	}
	void P2M(real m, real x, real y, real z) {
		solidR<N> Rlm(x, y, z);
		for (int l = 0; l < N; ++l) {
			for (int m = 0; m <= l; ++m) {
				this->at(l, m) += m * std::conj(Rlm(l, m));
			}
		}
	}
	void M2M(const multipole_expansion<N>& O, real dx, real dy, real dz) {
		solidR<N> Rlm(dx, dy, dz);
		for (int l0 = 0; l0 < N; ++l0) {
			for (int m0 = 0; m0 <= l0; ++m0) {
				for (int l1 = 0; l1 <= l0; ++l1) {
					for (int m1 = -l1; m1 <= +l1; ++m1) {
						if (std::abs(m0 - m1) <= l0 - l1) {
							this->at(l0, m0) += O(l1, m1) * std::conj(Rlm(l0 - l1, m0 - m1));
						}
					}
				}
			}
		}
	}
	local_expansion<N> M2L(real x, real y, real z, const multipole_expansion& MC) const {
		solidI<N + 1> Slm(x, y, z);
		local_expansion<N> L;
		const auto& M = *this;
		for (int l0 = 0; l0 < N; ++l0) {
			for (int m0 = 0; m0 <= l0; ++m0) {
				for (int l1 = 0; l0 + l1 < N && l1 < N - 1; ++l1) {
					for (int m1 = -l1; m1 <= +l1; ++m1) {
						if (std::abs(m0 - m1) <= l0 + l1) {
							L.at(l0, m0) += std::pow(-1, l1 + m1) * std::conj(Slm(l0 + l1, m0 - m1)) * M(l1, m1);
						}
					}
				}
			}
		}

		//const real mu = M(0, 0).real() / MC(0, 0).real();

		const int l0 = 1;
		const int l1 = N - 1;
		for (int m0 = 0; m0 <= l0; ++m0) {
			for (int m1 = -l1; m1 <= +l1; ++m1) {
				if (std::abs(m0 - m1) <= l0 + l1) {
					//		L.at(l0, m0) += std::pow(-1, l1 + m1) * std::conj(Slm(l0 + l1, m0 - m1))
					//				* (M(l1, m1) - mu * MC(l1, m1));
				}
			}
		}

		return L;
	}
	template<class Archive>
	void serialize(Archive& arc, const unsigned v) {
		spherical_expansion<N>::serialize(arc, v);
	}
};

#endif /* SOLIDHARMONIC_HPP_ */
