#pragma once

#include "Definitions.hpp"
#include "Math.hpp"
#include "Vector.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <iomanip>
#include <limits>

template <typename T, int N, int M = N>
struct Matrix;

template <typename T, int N, int M>
constexpr Matrix<T, N, M> inverse(Matrix<T, N, M>);

template <typename T, int N, int M>
struct Matrix {
	using LiteralType = std::array<std::array<T, M>, N>;
	static constexpr int rowCount() {
		return N;
	}
	static constexpr int colCount() {
		return M;
	}
	constexpr Matrix() = default;
	template <int L>
	constexpr Matrix(Vector<T, L> const &v) {
		if constexpr ((N == 1) && (L == M)) {
			data_[0] = v;
		} else if constexpr ((L == N) && (M == 1)) {
			*this = transpose(Matrix<T, 1, L>(v));
		} else {
			static_assert(false);
		}
	}
	constexpr Matrix(std::initializer_list<std::initializer_list<T>> const &list) {
		int i = 0;
		for (auto const &n : list) {
			int j = 0;
			for (auto const &nm : n) {
				data_[i][j++] = nm;
			}
			i++;
		}
	}
	constexpr Matrix(Vector<Vector<T, M>, N> const &array) :
		data_(array) {
	}
	constexpr Matrix(Matrix const &) = default;
	constexpr Matrix(Matrix &&) = default;
	constexpr Matrix &operator=(Matrix const &) = default;
	constexpr Matrix &operator=(Matrix &&) = default;
	constexpr T operator()(int n, int m) const {
		return data_[n][m];
	}
	constexpr T &operator()(int n, int m) {
		return data_[n][m];
	}
	constexpr Matrix &operator+=(Matrix const &B) {
		*this = *this + B;
		return *this;
	}
	constexpr Matrix &operator-=(Matrix const &B) {
		*this = *this - B;
		return *this;
	}
	constexpr Matrix &operator*=(Matrix const &B) {
		static_assert(N && (N == M));
		*this = *this * B;
		return *this;
	}
	constexpr Matrix &operator/=(Matrix const &B) {
		static_assert(N && (N == M));
		*this = *this / B;
		return *this;
	}
	constexpr Matrix &operator*=(T const &B) {
		*this = *this * B;
		return *this;
	}
	constexpr Matrix &operator/=(T const &B) {
		return (*this *= one / B);
	}
	template <int I, int J>
	constexpr T cofactor() const {
		static_assert(I > 1);
		static_assert(J > 1);
		static_assert(N && (N == M));
		return nonepow<I + J>() * minor<I, J>();
	}
	constexpr Matrix comatrix() const {
		static_assert(N == M);
		Matrix A;
		if constexpr (N == 1) {
			A(0, 0) = one;
		} else {
			auto const lambda = [&A, this]<int PQ>(auto const &self) {
				constexpr int P = PQ / M;
				constexpr int Q = PQ % M;
				A(P, Q) = cofactor<P, Q>();
				if constexpr (PQ + 1 < N * M) {
					self.template operator()<PQ + 1>(self);
				}
			};
			lambda.template operator()<0>(lambda);
		}
		return A;
	}
	template <int I, int J>
	constexpr T minor() const {
		static_assert(I > 1);
		static_assert(J > 1);
		static_assert(N && (N == M));
		auto const s = sub<I, J>();
		return det(s);
	}
	template <int I, int J>
	constexpr Matrix<T, N - 1, M - 1> sub() const {
		Matrix<T, N - 1, M - 1> A;
		auto const lambda = [&A, this]<int PQ>(auto const &self) {
			constexpr int P = PQ / (M - 1);
			constexpr int Q = PQ % (M - 1);
			if constexpr (P < I) {
				if constexpr (Q < J) {
					A(P, Q) = operator()(P, Q);
				} else {
					A(P, Q) = operator()(P, Q + 1);
				}
			} else {
				if constexpr (Q < J) {
					A(P, Q) = operator()(P + 1, Q);
				} else {
					A(P, Q) = operator()(P + 1, Q + 1);
				}
			}
			if constexpr (PQ + 1 < (N - 1) * (M - 1)) {
				self.template operator()<PQ + 1>(self);
			}
		};
		lambda.template operator()<0>(lambda);
		return A;
	}
	constexpr int density() const {
		int count = 0;
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < M; m++) {
				if (!isZero(data_[n][m])) {
					count++;
				}
			}
		}
		return count;
	}
	static constexpr Matrix identity() {
		static_assert(N && (N == M));
		;
		Matrix I;
		for (int n = 0; n < N; n++) {
			I.data_[n] = Vector<T, N>::unit(n);
		}
		return I;
	}
	friend constexpr Matrix operator+(Matrix const &A) {
		return A;
	}
	friend constexpr Matrix operator-(Matrix const &A) {
		return Matrix(-A.data_);
	}
	friend constexpr Matrix operator+(Matrix const &A, Matrix const &B) {
		return Matrix(A.data_ + B.data_);
	}
	friend constexpr Matrix operator-(Matrix A, Matrix const &B) {
		return Matrix(A.data_ - B.data_);
	}
	template <int L>
	constexpr Matrix<T, N, L> operator*(Matrix<T, M, L> B) const {
		Matrix<T, N, L> C;
		for (int n = 0; n < N; n++) {
			for (int l = 0; l < L; l++) {
				C(n, l) = zero;
				for (int m = 0; m < M; m++) {
					C(n, l) += (*this)(n, m) * B(m, l);
				}
			}
		}
		return C;
	}
	friend constexpr Vector<T, N> operator*(Matrix const &A, Vector<T, M> const &B) {
		Vector<T, N> C;
		for (int n = 0; n < N; n++) {
			C[n] = A.data_[n].dot(B);
		}
		return C;
	}
	friend constexpr Vector<T, N> operator*(Vector<T, N> const &A, Matrix const &B) {
		Vector<T, N> C;
		for (int n = 0; n < N; n++) {
			C[n] = B.data_[n].dot(A);
		}
		return transpose(C);
	}
	friend constexpr Matrix operator/(Matrix const &A, Matrix const &B) {
		return A * inverse(B);
	}
	friend constexpr Matrix operator*(Matrix A, T const &B) {
		for (int i = 0; i < N; i++) {
			A.data_[i] *= B;
		}
		return A;
	}
	friend constexpr Matrix operator*(T const &A, Matrix B) {
		return B * A;
	}
	friend constexpr Matrix operator/(Matrix A, T const &B) {
		return A * (one / B);
	}
	friend constexpr T det(Matrix const &A) {
		if constexpr ((N == 1) && (M == 1)) {
			return A(0, 0);
		} else {
			static_assert(N && (N == M));
			;
			auto const lambda = [A]<int I>(auto const &self) {
				if constexpr (I == M) {
					return zero;
				} else {
					if (A(0, I) == zero) {
						return self.template operator()<I + 1>(self);
					} else {
						return A(0, I) * A.cofactor<0, I>() + self.template operator()<I + 1>(self);
					}
				}
			};
			return lambda.template operator()<0>(lambda);
		}
	}
	friend constexpr Matrix norm(Matrix A) {
		using std::sqrt;
		T sum = zero;
		for (int n = 0; n < N; n++) {
			sum += A.data_[n].dot(A.data_[n]);
		}
		return sqrt(sum);
	}
	friend constexpr Matrix adjugate(Matrix const &A) {
		static_assert(N && (N == M));
		return transpose(A.comatrix());
	}
	friend constexpr auto pseudoinverse(Matrix const &A) {
		auto const trA = transpose(A);
		return trA * inverse(A * trA);
	}
	friend constexpr T trace(Matrix const &A) {
		T tr = zero;
		for (int n = 0; n < N; n++) {
			tr += A.data_[n][n];
		}
		return tr;
	}
	constexpr auto &rowSwp(int i, int j) {
		std::swap(data_[i], data_[j]);
		return *this;
	}
	constexpr auto &rowMul(int i, T const &value) {
		data_[i] *= value;
		return *this;
	}
	constexpr auto &rowSub(int i, int j) {
		data_[i] -= data_[j];
		return *this;
	}
	constexpr auto &rowMulSub(int i, int j, T c) {
		data_[i] -= c * data_[j];
		return *this;
	}
	constexpr auto &rowAdd(int i, int j) {
		data_[i] += data_[j];
		return *this;
	}
	friend std::ostream &operator<<(std::ostream &os, Matrix<T, N, M> const &A) {
		os << std::to_string(N) << "x" << std::to_string(M) << " matrix:" << std::endl;
		int constexpr minCellWidth = 4;
		int constexpr maxCellWidth = 32;
		int cellWidth = minCellWidth;
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < M; m++) {
				if (A(n, m) == T{}) {
					continue;
				}
				std::ostringstream ss;
				ss << A(n, m);
				int w = static_cast<int>(ss.str().size());
				if (w > cellWidth) {
					cellWidth = w;
				}
			}
		}
		bool const capped = (cellWidth > maxCellWidth);
		if (capped) {
			cellWidth = maxCellWidth;
		}
		auto printHorizontal = [&os, cellWidth]() {
			for (int m = 0; m < M; m++) {
				os << '+' << std::string(cellWidth, '-');
			}
			os << "+\n";
		};
		auto printCentered = [&os, cellWidth, capped](auto const &value) {
			if (value == T{}) {
				os << std::string(cellWidth, ' ');
				return;
			}
			std::ostringstream ss;
			ss << value;
			std::string s = ss.str();
			if (capped && static_cast<int>(s.size()) > cellWidth) {
				if (cellWidth >= 3) {
					s.resize(cellWidth - 3);
					s += "...";
				} else {
					s.resize(cellWidth);
				}
			}
			int const padding = cellWidth - static_cast<int>(s.size());
			int const left = padding / 2;
			int const right = padding - left;
			os << std::string(left, ' ') << s << std::string(right, ' ');
		};
		for (int n = 0; n < N; n++) {
			printHorizontal();
			for (int m = 0; m < M; m++) {
				os << '|';
				printCentered(A(n, m));
			}
			os << "|\n";
		}
		printHorizontal();
		return os;
	}
	constexpr Matrix(LiteralType const &lit) {
		for (int n = 0; n < N; n++) {
			data_[n] = lit[n];
		}
	}
	constexpr explicit operator LiteralType() const {
		return literal();
	}
	constexpr auto literal() const {
		LiteralType lit;
		for (int n = 0; n < N; n++) {
			lit[n] = data_[n];
		}
		return lit;
	}

	static constexpr bool isZero(T v) {
		if constexpr (!std::is_floating_point_v<T>) {
			return v == T(0);
		} else {
			constexpr auto eps = std::sqrt(std::numeric_limits<T>::epsilon());
			return abs(v) <= eps;
		}
	}
private:
	NUMERICAL_CONSTANTS(T);
	Vector<Vector<T, M>, N> data_;
};

template <typename T, int N, int M>
struct ConvertsToLiteral<Matrix<T, N, M>> {
	static constexpr bool value = true;
};

template <typename T, int N, int M>
constexpr Matrix<T, N, M> transpose(Matrix<T, M, N> const &A) {
	Matrix<T, N, M> B;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < M; m++) {
			B(n, m) = A(m, n);
		}
	}
	return B;
}

template <typename>
struct IsMatrix : public std::false_type {};

template <typename T, int N, int M>
struct IsMatrix<Matrix<T, N, M>> : public std::true_type {};

template <typename T>
concept MatrixType = IsMatrix<T>::value;

template <typename T, int N, int M>
constexpr Matrix<T, N, M> operator*(Vector<T, N> const &A, Vector<T, M> const &B) {
	Matrix<T, N, M> C;
	for (int n = 0; n < N; n++) {
		C.data_[n] = A[n] * B;
	}
	return C;
}

enum class GEType : int { full, half };

template <GEType geType, typename T, int N, int M, int... Ms>
constexpr auto gaussianElimination(Matrix<T, N, M> A, Matrix<T, N, Ms>... Bs) {
	NUMERICAL_CONSTANTS(T);
	static_assert(M >= N);
	using std::abs;
	using std::numeric_limits;
	using std::swap;
	auto const isZero = Matrix<T, N, M>::isZero;
	auto const swapRowsA = [&](int r0, int r1, int startCol) {
		for (int m = startCol; m < M; m++) {
			swap(A(r0, m), A(r1, m));
		}
	};
	auto const swapRowsB = [&](int r0, int r1) {
		(
			[&]<int K>(Matrix<T, N, K> &B) {
				for (int m = 0; m < K; m++) {
					swap(B(r0, m), B(r1, m));
				}
			}(Bs),
			...);
	};
	auto const scaleRowA = [&](int r, T s, int startCol) {
		for (int m = startCol; m < M; m++) {
			A(r, m) *= s;
		}
	};
	auto const scaleRowB = [&](int r, T s) {
		(
			[&]<int K>(Matrix<T, N, K> &B) {
				for (int m = 0; m < K; m++) {
					B(r, m) *= s;
				}
			}(Bs),
			...);
	};
	auto const rowMulSubA = [&](int dst, int src, T p, int startCol) {
		for (int m = startCol; m < M; m++) {
			A(dst, m) -= p * A(src, m);
		}
	};
	auto const rowMulSubB = [&](int dst, int src, T p) {
		(
			[&]<int K>(Matrix<T, N, K> &B) {
				for (int m = 0; m < K; m++) {
					B(dst, m) -= p * B(src, m);
				}
			}(Bs),
			...);
	};
	for (int n = 0; n < N; n++) {
		int pivotRow = -1;
		if constexpr (!std::is_floating_point_v<T>) {
			T best = T(0);
			for (int r = n; r < N; r++) {
				auto const v = abs(A(r, n));
				if (!isZero(v) && (pivotRow < 0 || v > best)) {
					best = v;
					pivotRow = r;
				}
			}
		} else {
			for (int r = n; r < N; r++) {
				if (!isZero(A(r, n))) {
					pivotRow = r;
					break;
				}
			}
		}
		if (pivotRow < 0) {
			continue;
		} else if (pivotRow != n) {
			swapRowsA(pivotRow, n, n);
			swapRowsB(pivotRow, n);
		}
		T const Ann = A(n, n);
		if (isZero(Ann)) {
			continue;
		}
		T const inv = one / Ann;
		A(n, n) = one;
		scaleRowA(n, inv, n + 1);
		scaleRowB(n, inv);
		for (int r = n + 1; r < N; r++) {
			T const p = A(r, n);
			if (isZero(p)) {
				continue;
			}
			A(r, n) = T(0);
			rowMulSubA(r, n, p, n + 1);
			rowMulSubB(r, n, p);
		}
	}
	if constexpr (geType == GEType::full) {
		for (int n = N - 1; n >= 0; n--) {
			for (int r = 0; r < n; r++) {
				T const p = A(r, n);
				if (isZero(p)) {
					A(r, n) = T(0);
					continue;
				}
				A(r, n) = T(0);
				rowMulSubA(r, n, p, n + 1);
				rowMulSubB(r, n, p);
			}
		}
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < N; m++) {
				if (isZero(A(n, m))) {
					A(n, m) = T(0);
				}
			}
		}
		return std::tuple(A, Bs...);
	} else {
		auto const rowIsZeroA = [&](int r) {
			for (int m = 0; m < M; m++) {
				if (!isZero(A(r, m))) {
					return false;
				}
			}
			return true;
		};
		int n, r;
		for (n = 0, r = N; n < r;) {
			if (rowIsZeroA(n)) {
				r--;
				swapRowsA(n, r, 0);
				swapRowsB(n, r);
			} else {
				n++;
			}
		}
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < N; m++) {
				if (isZero(A(n, m))) {
					A(n, m) = T(0);
				}
			}
		}
		return std::tuple(A, Bs..., r);
	}
}

template <typename T, int N, int M>
constexpr auto rankReduce(Matrix<T, N, M> A) {
	NUMERICAL_CONSTANTS(T);
	static_assert(M >= N);
	auto const swapRows = [&A](int r0, int r1, int startCol) {
		for (int m = startCol; m < M; m++) {
			std::swap(A(r0, m), A(r1, m));
		}
	};
	auto const scaleRow = [&A](int r, T s, int startCol) {
		for (int m = startCol; m < M; m++) {
			A(r, m) *= s;
		}
	};
	auto const rowMulSub = [&A](int dst, int src, T p, int startCol) {
		for (int m = startCol; m < M; m++) {
			A(dst, m) -= p * A(src, m);
		}
	};
	auto const rowIsZeroA = [&A](int r) {
		for (int m = 0; m < M; m++) {
			if (A(r, m) != zero) {
				return false;
			}
		}
		return true;
	};
	for (int n = 0; n < N; n++) {
		int pivotRow = -1;
		T best = T(0);
		for (int r = n; r < N; r++) {
			auto const v = abs(A(r, n));
			if ((v != zero) && (pivotRow < 0 || v > best)) {
				best = v;
				pivotRow = r;
			}
		}
		if (pivotRow < 0) {
			continue;
		} else if (pivotRow != n) {
			swapRows(pivotRow, n, n);
		}
		T const Ann = A(n, n);
		if (Ann == zero) {
			continue;
		}
		T const inv = one / Ann;
		A(n, n) = one;
		scaleRow(n, inv, n + 1);
		for (int r = n + 1; r < N; r++) {
			T const p = A(r, n);
			if (p == zero) {
				continue;
			}
			A(r, n) = zero;
			rowMulSub(r, n, p, n + 1);
		}
	}
	int n, r;
	for (n = 0, r = N; n < r;) {
		if (rowIsZeroA(n)) {
			r--;
			swapRows(n, r, 0);
		} else {
			n++;
		}
	}
	return A;
}

template <typename T, int N1, int N2, int M>
constexpr Matrix<T, N1 + N2, M> concatenateDown(Matrix<T, N1, M> const &H, Matrix<T, N2, M> const &L) {
	Matrix<T, N1 + N2, M> A;
	for (int i = 0; i < N1; i++) {
		for (int j = 0; j < M; j++) {
			A(i, j) = H(i, j);
		}
	}
	for (int i = 0; i < N2; i++) {
		for (int j = 0; j < M; j++) {
			A(i + N1, j) = L(i, j);
		}
	}
	return A;
}

template <typename T, int N, int M>
constexpr Matrix<T, N, M> inverse(Matrix<T, N, M> A) {
	static_assert(M == N);
	if constexpr (N <= 2) {
		return adjugate(A) / det(A);
	} else {
		return std::get<1>(gaussianElimination<GEType::full>(A, decltype(A)::identity()));
	}
}

template <typename T, int M, int N1, int N2>
constexpr Matrix<T, N1 + N2, M> concatenateUp(Matrix<T, N1, M> const &U, Matrix<T, N2, M> const &D) {
	return concatenateDown(D, U);
}

template <typename T, int N, int M1, int M2>
constexpr Matrix<T, N, M1 + M2> concatenateRight(Matrix<T, N, M1> const &L, Matrix<T, N, M2> const &R) {
	auto const trL = transpose(L);
	auto const trR = transpose(R);
	auto const C = concatenateDown(trL, trR);
	return transpose(C);
}

template <typename T, int N, int M1, int M2>
constexpr Matrix<T, N, M1 + M2> concatenateLeft(Matrix<T, N, M1> const &L, Matrix<T, N, M2> const &R) {
	return concatenateRight(R, L);
}

template <typename T, int N1, int M1, int N2, int M2>
constexpr auto kroneckerProduct(Matrix<T, N1, M1> const &A, Matrix<T, N2, M2> const &B) {
	Matrix<T, N1 * N2, M1 * M2> C;
	for (int n1 = 0; n1 < N1; n1++) {
		for (int m1 = 0; m1 < M1; m1++) {
			for (int n2 = 0; n2 < N2; n2++) {
				for (int m2 = 0; m2 < M2; m2++) {
					C(N2 * n1 + n2, M2 * m1 + m2) = A(n1, m1) * B(n2, m2);
				}
			}
		}
	}
	return C;
}
