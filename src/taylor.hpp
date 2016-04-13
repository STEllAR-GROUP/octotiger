/*
 * taylor.hpp
 *
 *  Created on: Jun 9, 2015
 *      Author: dmarce1
 */

#ifndef TAYLOR_HPP_
#define TAYLOR_HPP_

#include "defs.hpp"
#include "simd.hpp"
#include <array>
#include <cmath>

#define MAX_ORDER 5

struct taylor_consts {
	static const real delta[3][3];
	static integer map2[3][3];
	static integer map3[3][3][3];
	static integer map4[3][3][3][3];
};

template<int N, class T = real>
class taylor {
private:
	static constexpr integer sizes[MAX_ORDER] = { 1, 4, 10, 20, 35 }; //
	static constexpr integer my_size = sizes[N - 1];
	static taylor_consts tc;
	std::array<T, my_size> data;
public:
	T& operator[](integer i) {
		return data[i];
	}
	const T& operator[](integer i) const {
		return data[i];
	}
	static constexpr decltype(my_size) size() {
		return my_size;
	}
	taylor() = default;
	~taylor() = default;
	taylor(const taylor<N, T>&) = default;
	taylor(taylor<N, T> && other) {
		data = std::move(other.data);
	}
	taylor<N, T>& operator=(const taylor<N, T>&) = default;
	taylor<N, T>& operator=(taylor<N, T> && other) {
		data = std::move(other.data);
		return *this;
	}

	taylor<N, T>& operator=(T d) {
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] = d;
		}
		return *this;
	}

	taylor<N, T>& operator*=(T d) {
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] *= d;
		}
		return *this;
	}

	taylor<N, T>& operator/=(T d) {
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] /= d;
		}
		return *this;
	}

	taylor<N, T>& operator+=(const taylor<N, T>& other) {
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] += other.data[i];
		}
		return *this;
	}

	taylor<N, T>& operator-=(const taylor<N, T>& other) {
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] -= other.data[i];
		}
		return *this;
	}

	taylor<N, T> operator+(const taylor<N, T>& other) const {
		taylor<N, T> r = *this;
		r += other;
		return r;
	}

	taylor<N, T> operator-(const taylor<N, T>& other) const {
		taylor<N, T> r = *this;
		r -= other;
		return r;
	}

	taylor<N, T> operator*(const T& d) const {
		taylor<N, T> r = *this;
		r *= d;
		return r;
	}

	taylor<N, T> operator/(const T& d) const {
		taylor<N, T> r = *this;
		r /= d;
		return r;
	}

	taylor<N, T> operator+() const {
		return *this;
	}

	taylor<N, T> operator-() const {
		taylor<N, T> r = *this;
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			r.data[i] = -r.data[i];
		}
		return r;
	}

	T operator()() const {
		return data[0];
	}

	T operator()(integer i) const {
		return data[1 + i];
	}

	T operator()(integer i, integer j) const {
		return data[tc.map2[i][j]];
	}

	T operator()(integer i, integer j, integer k) const {
		return data[tc.map3[i][j][k]];
	}

	T operator()(integer i, integer j, integer k, integer l) const {
		return data[tc.map4[i][j][k][l]];
	}

	T& operator()() {
		return data[0];
	}

	T& operator()(integer i) {
		return data[1 + i];
	}

	T& operator()(integer i, integer j) {
		return data[tc.map2[i][j]];
	}

	T& operator()(integer i, integer j, integer k) {
		return data[tc.map3[i][j][k]];
	}

	T& operator()(integer i, integer j, integer k, integer l) {
		return data[tc.map4[i][j][k][l]];
	}

	taylor<N, T>& operator>>=(const std::array<T, NDIM>& X) {
		const taylor<N, T>& A = *this;
		taylor<N, T> B = A;

		if (N > 1) {
			for (integer a = 0; a < NDIM; ++a) {
				B(a) += A() * X[a];
			}
			if (N > 2) {
				for (integer a = 0; a < NDIM; ++a) {
					for (integer b = a; b < NDIM; ++b) {
						B(a, b) += A(a) * X[b] + X[a] * A(b);
						B(a, b) += A() * X[a] * X[b];
					}
				}
				if (N > 3) {
					for (integer a = 0; a < NDIM; ++a) {
						for (integer b = a; b < NDIM; ++b) {
							for (integer c = b; c < NDIM; ++c) {
								B(a, b, c) += A(a, b) * X[c] + A(b, c) * X[a] + A(c, a) * X[b];
								B(a, b, c) += A(a) * X[b] * X[c] + A(b) * X[c] * X[a] + A(c) * X[a] * X[b];
								B(a, b, c) += A() * X[a] * X[b] * X[c];
							}
						}
					}
				}
			}
		}
		*this = B;
		return *this;
	}

	taylor<N, T> operator>>(const std::array<T, NDIM>& X) const {
		taylor<N, T> r = *this;
		r >>= X;
		return r;
	}

	taylor<N, T>& operator<<=(const std::array<T, NDIM>& X) {
		const taylor<N, T>& A = *this;
		taylor<N, T> B = A;

		if (N > 1) {
			for (integer a = 0; a != NDIM; a++) {
				B() += A(a) * X[a];
			}
			if (N > 2) {
				for (integer a = 0; a != NDIM; a++) {
					for (integer b = 0; b != NDIM; b++) {
						B() += A(a, b) * X[a] * X[b] * T(real(1) / real(2));
					}
				}
				for (integer a = 0; a != NDIM; a++) {
					for (integer b = 0; b != NDIM; b++) {
						B(a) += A(a, b) * X[b];
					}
				}
				if (N > 3) {
					for (integer a = 0; a != NDIM; a++) {
						for (integer b = 0; b != NDIM; b++) {
							for (integer c = 0; c != NDIM; c++) {
								B() += A(a, b, c) * X[a] * X[b] * X[c] * T(real(1) / real(6));
							}
						}
					}
					for (integer a = 0; a != NDIM; a++) {
						for (integer b = 0; b != NDIM; b++) {
							for (integer c = 0; c != NDIM; c++) {
								B(a) += A(a, b, c) * X[b] * X[c] * T(real(1) / real(2));
							}
						}
					}
					for (integer a = 0; a != NDIM; a++) {
						for (integer b = 0; b != NDIM; b++) {
							for (integer c = a; c != NDIM; c++) {
								B(a, c) += A(a, b, c) * X[b];
							}
						}
					}
				}
			}
		}
		*this = B;
		return *this;
	}

	taylor<N, T> operator<<(const std::array<T, NDIM>& X) const {
		taylor<N, T> r = *this;
		r <<= X;
		return r;
	}

	void set_basis(const std::array<T, NDIM>& X) {
		taylor<N, T>& A = *this;
		T r2 = X[0] * X[0] + X[1] * X[1] + X[2] * X[2];
		auto this_one = X[0];
		this_one = ONE;
		const T r2inv = this_one / r2;
		const T d0 = -sqrt(r2inv);
		A() = d0;

		if (N > 1) {
			const T d1 = -d0 * r2inv;
			for (integer a = 0; a != NDIM; a++) {
				A(a) = X[a] * d1;
			}
			if (N > 2) {
				const T d2 = -T(3) * d1 * r2inv;
				for (integer a = 0; a != NDIM; a++) {
					for (integer b = a; b != NDIM; b++) {
						A(a, b) = X[a] * X[b] * d2;
						if (a == b) {
							A(a, b) += d1;
						}
					}
				}
				if (N > 3) {
					const T d3 = -T(5) * d2 * r2inv;
					for (integer a = 0; a != NDIM; a++) {
						for (integer b = a; b != NDIM; b++) {
							for (integer c = b; c != NDIM && b != NDIM; c++) {
								A(a, b, c) = X[a] * X[b] * X[c] * d3;
								if (a == b) {
									A(a, b, c) += (X[c]) * d2;
								}
								if (b == c) {
									A(a, b, c) += (X[a]) * d2;
								}
								if (a == c) {
									A(a, b, c) += (X[b]) * d2;
								}
							}
						}
					}
					if (N > 4) {
						const T d4 = -T(7) * d3 * r2inv;
						for (integer a = 0; a != NDIM; a++) {
							for (integer b = a; b != NDIM; b++) {
								for (integer c = b; c != NDIM; c++) {
									for (integer d = c; d != NDIM && c != NDIM; ++d) {
										A(a, b, c, d) = X[a] * X[b] * X[c] * X[d] * d4;
										if (a == b) {
											A(a, b, c, d) += X[c] * X[d] * d3;
											if (c == d) {
												A(a, b, c, d) += d2;
											}
										}
										if (a == c) {
											A(a, b, c, d) += X[b] * X[d] * d3;
											if (b == d) {
												A(a, b, c, d) += d2;
											}
										}
										if (a == d) {
											A(a, b, c, d) += X[b] * X[c] * d3;
											if (b == c) {
												A(a, b, c, d) += d2;
											}
										}
										if (b == c) {
											A(a, b, c, d) += X[a] * X[d] * d3;
										}
										if (b == d) {
											A(a, b, c, d) += X[a] * X[c] * d3;
										}
										if (c == d) {
											A(a, b, c, d) += X[a] * X[b] * d3;
										}

									}
								}
							}
						}
					}
				}
			}
		}
	}

	T* ptr() {
		return data.data();
	}

	const T* ptr() const {
		return data.data();
	}

	template<class Arc>
	void serialize(Arc& arc, const unsigned) {
		arc & data;
	}

};

#include "space_vector.hpp"

template<int N, class T>
taylor_consts taylor<N, T>::tc;

#include "valarray.hpp"

typedef taylor<4, real> multipole;
typedef taylor<4, real> expansion;
typedef std::pair<std::vector<multipole>, std::vector<space_vector>> multipole_pass_type;
typedef std::pair<std::vector<expansion>, std::vector<expansion>> expansion_pass_type;

#endif /* TAYLOR_HPP_ */
