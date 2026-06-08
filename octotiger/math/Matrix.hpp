/*
 * Matrix.hpp
 *
 *  Created on: Feb 4, 2026
 *      Author: dmarce1
 */

#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <cassert>
#include <iostream>
#include <ostream>

#include "Math.hpp"
#include "Vector.hpp"

template <typename T, Integer N, Integer M = N>
struct Matrix;

template <typename T>
struct IsMatrix : std::false_type
{
};

template <typename T, Integer N, Integer M>
struct IsMatrix<Matrix<T, N, M>> : std::true_type
{
};

template <typename T, Integer N, Integer M>
struct Matrix
{
    using LiteralType = std::array<std::array<T, M>, N>;
    constexpr LiteralType literal() const {
        LiteralType a;
        for (Integer n = 0; n < N; n++) {
            for (Integer m = 0; m < M; m++) {
                a[n][m] = data_[n][m];
            }
        }
        return a;
    }
    constexpr Matrix(LiteralType const& a) {
        for (Integer n = 0; n < N; n++) {
            for (Integer m = 0; m < M; m++) {
                data_[n][m] = a[n][m];
            }
        }
    }
    constexpr Matrix() = default;
    constexpr Matrix(Matrix const&) = default;
    explicit constexpr Matrix(Vector<T, N> const& v) {
        static_assert(N == M);
        for (auto n = 0_I; n < N; n++) {
            for (auto m = 0_I; m < N; m++) {
                (*this)[n][m] = (n == m) ? v[n] : T(0_R);
            }
        }
    }

    constexpr Matrix& operator=(Matrix const&) = default;
    constexpr Matrix(T const& init) {
        std::fill(data_.begin(), data_.end(), init);
    }
    constexpr Matrix(std::initializer_list<std::initializer_list<T>>&& iLists) {
        *this = std::move(iLists);
    }
    constexpr Matrix& operator=(std::initializer_list<std::initializer_list<T>>&& iLists) {
        data_.fill(T(0));
        std::copy(iLists.begin(), iLists.end(), data_.begin());
        return *this;
    }
    constexpr Matrix const& operator+() const {
        return *this;
    }
    constexpr auto const& operator[](Integer i) const {
        return data_[i];
    }
    constexpr auto& operator[](Integer i) {
        return data_[i];
    }
    constexpr Matrix operator-() const {
        Matrix result;
        for (Integer i = 0; i < N; i++) {
            result[i] = -data_[i];
        }
        return result;
    }
    constexpr Matrix operator+(Matrix const& other) const {
        Matrix result;
        for (Integer i = 0; i < N; i++) {
            result[i] = data_[i] + other.data_[i];
        }
        return result;
    }
    constexpr Matrix operator-(Matrix const& other) const {
        Matrix result;
        for (Integer i = 0; i < N; i++) {
            result[i] = data_[i] - other.data_[i];
        }
        return result;
    }
    constexpr Matrix& operator+=(Matrix const& other) {
        *this = *this + other;
        return *this;
    }
    constexpr Matrix& operator-=(Matrix const& other) {
        *this = *this - other;
        return *this;
    }
    Matrix& operator*=(Matrix const& other) {
        static_assert(N == M);
        *this = *this * other;
        return *this;
    }
    constexpr Matrix& operator*=(T const& scalar) {
        *this = *this * scalar;
        return *this;
    }
    constexpr Matrix& operator/=(T const& scalar) {
        *this = *this / scalar;
        return *this;
    }
    constexpr auto operator*(T const& scale) const {
        Matrix result;
        for (Integer i = 0; i < N; i++) {
            result[i] = data_[i] * scale;
        }
        return result;
    }
    constexpr auto operator/(T const& scalar) const {
        return (*this) * inv(scalar);
    }
    constexpr auto& front() {
        return data_.front();
    }
    constexpr auto& back() {
        return data_.back();
    }
    constexpr auto front() const {
        return data_.front();
    }
    constexpr auto back() const {
        return data_.back();
    }
    constexpr auto begin() const {
        return data_.cbegin();
    }
    constexpr auto end() const {
        return data_.cend();
    }
    constexpr auto begin() {
        return data_.begin();
    }
    constexpr auto end() {
        return data_.end();
    }
    constexpr void setCol(Integer m, Vector<T, M> const& col) {
        for (Integer n = 0; n < N; n++) {
            (*this)[n][m] = col[n];
        }
    }
    constexpr void setRow(Integer n, Vector<T, M> const& row) {
        (*this)[n] = row;
    }
    template <typename U, Integer L>
    constexpr auto operator*(Matrix<U, M, L> const& other) const {
        Matrix<T, N, L> result;
        for (Integer n = 0; n < N; n++) {
            for (Integer l = 0; l < L; l++) {
                result[n][l] = T(0_I);
                for (Integer m = 0; m < M; m++) {
                    result[n][l] += data_[n][m] * other[m][l];
                }
            }
        }
        return result;
    }
    friend constexpr Matrix<T, M, N> transpose(Matrix const& other) {
        Matrix<T, M, N> result;
        for (Integer m = 0; m < M; m++) {
            for (Integer n = 0; n < N; n++) {
                result[m][n] = other[n][m];
            }
        }
        return result;
    }
    friend constexpr auto operator*(T const& scalar, Matrix const& matrix) {
        return matrix * scalar;
    }
    static constexpr auto size() {
        return N * M;
    }
    static constexpr auto rowCount() {
        return N;
    }
    static constexpr auto colCount() {
        return M;
    }
    static constexpr auto identity() {
        Matrix<T, N, M> I{T(0_I)};
        constexpr Integer L = std::min(N, M);
        for (Integer n = 0; n < L; n++) {
            I.data_[n][n] = T(1_I);
        }
        return I;
    }
    static constexpr auto parity() {
        Matrix<Real, N> P{};
        for (auto m = 0_I, p = N - 1; m <= p; m++, p--) {
            P[m][m] -= 0.5_R;
            P[m][p] += 0.5_R;
            P[p][m] += 0.5_R;
            P[p][p] += 0.5_R;
        }
        return P;
    }

    static constexpr auto shuffle() {
        constexpr auto I = identity();
        Matrix<T, N> S;
        constexpr auto Ne = (N + 1) / 2;
        constexpr auto No = N / 2;
        for (Integer n = 0; n < Ne; n++) {
            S[2 * n + 0] = I[n];
        }
        for (Integer n = 0; n < No; n++) {
            S[2 * n + 1] = I[n + Ne];
        }
        return S;
    };

    static constexpr auto isZero(T const& value) {
        if constexpr (std::is_floating_point<T>::value) {
            constexpr T ε = std::sqrt(std::numeric_limits<T>::epsilon());
            return ((-ε < value) && (value < +ε));
        } else {
            return value == T(0_I);
        }
    }
    template <Integer N2, Integer M2>
    constexpr auto getBlock(Integer j, Integer k) const {
        Matrix<T, N2, M2> B;
        for (Integer n = 0; n < N2; n++) {
            for (Integer m = 0; m < M2; m++) {
                B[n][m] = (*this)[n + j][m + k];
            }
        }
        return B;
    }
    template <Integer N2, Integer M2>
    constexpr auto& setBlock(Integer j, Integer k, Matrix<T, N2, M2> const& B) {
        for (Integer n = 0; n < N2; n++) {
            for (Integer m = 0; m < M2; m++) {
                (*this)[n + j][m + k] = B[n][m];
            }
        }
        return *this;
    }
    constexpr auto sub(Integer j, Integer k) const {
        auto const& A = *this;
        Matrix<T, N - 1, M - 1> B;
        for (Integer n = 0; n < j; n++) {
            for (Integer m = 0; m < k; m++) {
                B[n][m] = A[n][m];
            }
            for (Integer m = k; m < M - 1; m++) {
                B[n][m] = A[n][m + 1];
            }
        }
        for (Integer n = j; n < N - 1; n++) {
            for (Integer m = 0; m < k; m++) {
                B[n][m] = A[n + 1][m];
            }
            for (Integer m = k; m < M - 1; m++) {
                B[n][m] = A[n + 1][m + 1];
            }
        }
        return B;
    }
    friend constexpr auto det(Matrix const& A) {
        static_assert(N == M);
        if constexpr (N == 1) {
            return A[0][0];
        } else {
            T d = T(0_I);
            for (Integer n = 0; n < N; n += 2) {
                if (A[0][n] != T(0_I))
                    d += A[0][n] * det(A.sub(0, n));
            }
            for (Integer n = 1; n < N; n += 2) {
                if (A[0][n] != T(0_I))
                    d -= A[0][n] * det(A.sub(0, n));
            }
            return d;
        }
    }
    friend constexpr auto inv(Matrix const& A) {
        static_assert(N == M);
        Matrix iA;
        if constexpr (N == 1) {
            iA[0][0] = inv(A[0][0]);
        } else {
            for (Integer n = 0; n < N; n += 2) {
                for (Integer m = 0; m < M; m += 2) {
                    iA[n][m] = +det(A.sub(n, m));
                }
                for (Integer m = 1; m < M; m += 2) {
                    iA[n][m] = -det(A.sub(n, m));
                }
            }
            for (Integer n = 1; n < N; n += 2) {
                for (Integer m = 0; m < M; m += 2) {
                    iA[n][m] = -det(A.sub(n, m));
                }
                for (Integer m = 1; m < M; m += 2) {
                    iA[n][m] = +det(A.sub(n, m));
                }
            }
            T d = A[0][0] * det(A.sub(0, 0));
            for (Integer n = 2; n < N; n += 2) {
                d += A[0][n] * det(A.sub(0, n));
            }
            for (Integer n = 1; n < N; n += 2) {
                d -= A[0][n] * det(A.sub(0, n));
            }
            T const id = inv(d);
            iA = id * transpose(iA);
        }
        return iA;
    }
    friend constexpr auto chop(Matrix A) {
        using std::abs;
        auto const ε = 1e-10;
        for (Integer n = 0; n < N; n++) {
            for (Integer m = 0; m < M; m++) {
                if (abs(A[n][m]) < ε)
                    A[n][m] = T(0);
            }
        }
        return A;
    }
    //	friend constexpr auto inv(Matrix A) {
    //		Matrix<T, M, N> iA;
    //		if constexpr (N == M) {
    //			iA = identity();
    //			for (Integer n = 0; n < N; n++) {
    //				if (isZero(A[n][n])) {
    //					for (Integer j = n; j < N; j++) {
    //						T const Aⱼₙ = A[j][n];
    //						if (!isZero(Aⱼₙ)) {
    //							std::swap(iA[j], iA[n]);
    //							std::swap(A[j], A[n]);
    //							break;
    //						}
    //						//	assert(j != N - 1);
    //					}
    //				}
    //				T const iAₙₙ = inv(A[n][n]);
    //				iA[n] *= iAₙₙ;
    //				A[n] *= iAₙₙ;
    //				for (Integer j = n + 1; j < N; j++) {
    //					T const Aⱼₙ = A[j][n];
    //					if (!isZero(Aⱼₙ)) {
    //						iA[j] -= Aⱼₙ * iA[n];
    //						A[j] -= Aⱼₙ * A[n];
    //					}
    //				}
    //			}
    //			for (Integer n = N - 1; n >= 0; n--) {
    //				for (Integer j = 0; j < n; j++) {
    //					T const Aⱼₙ = A[j][n];
    //					if (!isZero(Aⱼₙ)) {
    //						iA[j] -= Aⱼₙ * iA[n];
    //						A[j] -= Aⱼₙ * A[n];
    //					}
    //				}
    //			}
    //		} else {
    //			constexpr Integer L = N + M;
    //			constexpr auto Iₙₙ = Matrix<T, M>::identity();
    //			constexpr auto Oₘₘ = Matrix<T, M>(T(0));
    //			Matrix<T, L> K;
    //			auto const trA = transpose(A);
    //			K.setBlock(0, 0, Iₙₙ);
    //			K.setBlock(0, N, A);
    //			K.setBlock(N, 0, trA);
    //			K.setBlock(N, N, Oₘₘ);
    //			iA = inv(K).template getBlock<M, N>(N, 0);
    //		}
    //		return iA;
    //	}
    //	friend constexpr auto det(Matrix A) {
    //		static_assert(N == M);
    //		T det = T(1);
    //		for (Integer n = 0; n < N; n++) {
    //			T Aₙₙ = A[n][n];
    //			if (isZero(Aₙₙ)) {
    //				for (Integer j = n; j < N; j++) {
    //					if (j == N) {
    //						return T(0);
    //					}
    //					T const Aⱼₙ = A[j][n];
    //					if (!isZero(Aⱼₙ)) {
    //						std::swap(A[j], A[n]);
    //						Aₙₙ = A[n][n];
    //						det = -det;
    //						break;
    //					}
    //				}
    //			}
    //			A[n] *= inv(Aₙₙ);
    //			det *= Aₙₙ;
    //			for (Integer j = n + 1; j < N; j++) {
    //				T const Aⱼₙ = A[j][n];
    //				if (!isZero(Aⱼₙ)) {
    //					A[j] -= Aⱼₙ * A[n];
    //				}
    //			}
    //		}
    //		return det;
    //	}
    friend constexpr auto luDecomposition(Matrix LU) {
        static_assert(N == M);
        for (Integer n = 0; n < N; n++) {
            assert(!any(LU[n][n] == T(0)));
            auto const pivot = LU[n][n];
            auto const invPivot = inv(pivot);
            for (Integer k = n + 1; k < N; k++) {
                LU[n][k] *= invPivot;
            }
            for (Integer j = n + 1; j < N; j++) {
                auto const factor = LU[j][n];
                for (Integer k = n + 1; k < N; k++) {
                    LU[j][k] -= factor * LU[n][k];
                }
            }
        }
        return LU;
    }
    friend constexpr auto lupDecomposition(Matrix LU) {
        static_assert(N == M);
        Vector<Integer, N> P;
        Matrix L, U;
        std::iota(P.begin(), P.end(), 0);
        for (Integer n = 0; n < N; n++) {
            Integer j = n;
            while (LU[j][n] == T(0)) {
                j++;
            }
            std::swap(LU[j], LU[n]);
            std::swap(P[j], P[n]);
            auto const iLUnn = inv(LU[n][n]);
            for (Integer j = n + 1; j < N; j++) {
                LU[j][n] *= iLUnn;
                for (Integer k = n + 1; k < N; k++) {
                    LU[j][k] -= LU[j][n] * LU[n][k];
                }
            }
        }
        for (Integer n = 0; n < N; n++) {
            for (Integer j = n; j < N; j++) {
                U[n][j] = LU[n][j];
            }
            for (Integer j = 0; j < n; j++) {
                L[n][j] = LU[n][j];
            }
            L[n][n] = 1_R;
        }
        return std::tuple(L, U, P);
    }

private:
    std::array<Vector<T, M>, N> data_{};
};

template <typename T, typename U, Integer N1, Integer N2>
constexpr auto operator*(Matrix<T, N1, N2> const& M, Vector<U, N2> const& V) {
    using R = decltype(T{} * U{});
    Vector<R, N1> Y;
    for (Integer n = 0; n < N1; n++) {
        Y[n] = M[n].dot(V);
    }
    return Y;
}

template <typename T, typename U, Integer N1, Integer N2>
constexpr auto dyadicProduct(Vector<T, N1> const& u, Vector<U, N2> const& v) {
    using R = decltype(T{} * U{});
    Matrix<R, N1, N2> Y;
    for (Integer n = 0; n < N1; n++) {
        for (Integer m = 0; m < N2; m++) {
            Y[n][m] = u[n] * v[m];
        }
    }
    return Y;
}

template <typename T, typename U, Integer N1, Integer N2>
constexpr auto operator*(Vector<U, N2> const& V, Matrix<T, N2, N1> const& M) {
    return transpose(M) * V;
}

template <typename T, Integer N1, Integer M1, Integer N2, Integer M2>
constexpr auto kroneckerProduct(Matrix<T, N1, M1> const& A, Matrix<T, N2, M2> const& B) {
    constexpr Integer N = N1 * N2;
    constexpr Integer M = M1 * M2;
    Matrix<T, N, M> C;
    for (Integer n1 = 0; n1 < N1; n1++) {
        for (Integer m1 = 0; m1 < M1; m1++) {
            for (Integer n2 = 0; n2 < N2; n2++) {
                for (Integer m2 = 0; m2 < M2; m2++) {
                    C[n1 * N2 + n2][m1 * M2 + m2] = A[n1][m1] * B[n2][m2];
                }
            }
        }
    }
    return C;
}

template <typename T, Integer D>
constexpr auto rotationMatrix(Vector<T, D> const& n) {
    using std::abs;
    constexpr Integer ix = 0;
    constexpr Integer iy = 1;
    Matrix<T, D> R;
    if constexpr (D > 1) {
        constexpr auto ex = Vector<T, D>::unit(0);
        constexpr auto ey = Vector<T, D>::unit(1);
        R[ix] = n.normalize();
        R[iy] = select(abs(R[ix][ix]) < 0.5_R, ex, ey);
        for (Integer k = 2; k < D; k++) {
            R[k] = Vector<T, D>::unit(k);
            for (Integer i = 0; i < k; i++) {
                R[k] -= R[i][k] * R[i];
            }
            R[k] = R[k].normalize();
        }
    } else {
        R[ix][ix] = T(1_R);
    }
    auto const L = inv(R);
    return R;
}

// template <typename T, Integer N>
// struct DiagonalMatrix : public Vector<T, N> {
//	operator Matrix<T, N>() const {
//		Matrix<T, N> D{};
//		for (Integer n = 0; n < N; n++) {
//			D[n][n] = (*this)[n];
//		}
//		return D;
//	}
//	auto &operator*=(DiagonalMatrix<T, N> const &other) {
//		*this = *this * other;
//		return *this;
//	}
//	template <typename U>
//	auto operator*(DiagonalMatrix<U, N> const &other) const {
//		using R = decltype(U{} * T{});
//		DiagonalMatrix<R, N> result;
//		for (Integer n = 0; n < N; n++) {
//			result[n] = (this)[n] * other[n];
//		}
//		return result;
//	}
//	template <typename U, Integer M>
//	friend auto operator*(Matrix<U, M, N> const &A, DiagonalMatrix const &D) {
//		using R = decltype(U{} * T{});
//		Matrix<R, M, N> AD;
//		for (Integer n = 0; n < N; n++) {
//			T const d = D[n];
//			for (Integer m = 0; m < M; m++) {
//				AD[m][n] = A[m][n] * d;
//			}
//		}
//		return AD;
//	}
//	template <typename U, Integer M>
//	friend auto operator*(DiagonalMatrix const &D, Matrix<U, N, M> const &A) {
//		using R = decltype(T{} * U{});
//		Matrix<R, N, M> DA;
//		for (Integer n = 0; n < N; n++) {
//			T const d = D[n];
//			for (Integer m = 0; m < M; m++) {
//				DA[n][m] = d * A[n][m];
//			}
//		}
//		return DA;
//	}
//	friend auto inv(DiagonalMatrix const &D) {
//		using R = decltype(inv(T{1}));
//		DiagonalMatrix<R, N> iD;
//		for (Integer n = 0; n < N; n++) {
//			iD[n] = inv(D[n]);
//		}
//		return iD;
//	}
// };

template <typename T, Integer N, Integer M>
std::ostream& operator<<(std::ostream& os, Matrix<T, N, M> const& A) {
    std::array<std::array<std::string, M>, N> strs;
    size_t maxLen = 0;
    for (Integer n = 0; n < N; n++) {
        for (Integer m = 0; m < M; m++) {
            auto& str = strs[n][m];
            auto const& Anm = A[n][m];
            //			if (!MatrixType::isZero(Anm)) {
            std::stringstream ss;
            ss << Anm;
            str = ss.str();
            maxLen = std::max(maxLen, str.size());
            //			}
        }
    }
    size_t const fieldLen = maxLen + 2;
    auto const hlineOut = [fieldLen, &os](char const* left, char const* mid, char const* right) {
        os << left;
        for (size_t j = 0; j < M; ++j) {
            for (size_t k = 0; k < fieldLen; k++) {
                os << "─";
            }
            if (j + 1 < M) {
                os << mid;
            }
        }
        os << right << std::endl;
    };
    hlineOut("┌", "┬", "┐");
    for (Integer n = 0; n < N; n++) {
        os << "|";
        for (Integer m = 0; m < M; m++) {
            auto const& str = strs[n][m];
            auto const leftPad = (maxLen - str.size() + 1) / 2 + 1;
            auto const rightPad = (maxLen - str.size()) / 2 + 1;
            os << std::string(leftPad, ' ') + strs[n][m] + std::string(rightPad, ' ') << '|';
        }
        os << std::endl;
        if (n + 1 < N) {
            hlineOut("├", "┼", "┤");
        }
    }
    hlineOut("└", "┴", "┘");
    return os;
}

#endif /* MATRIX_HPP_ */
