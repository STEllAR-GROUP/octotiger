#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

template <typename T, int N>
struct Vector {
	static_assert(N >= 0);
	static constexpr int size() {
		return N;
	}
	constexpr T operator[](int i) const {
		return components[i];
	}
	constexpr T &operator[](int i) {
		return components[i];
	}
	constexpr Vector() = default;
	constexpr Vector(std::initializer_list<T> const &list) :
		components{} {
		if (list.size() > std::size_t(N)) {
			throw std::length_error("Too many elements in Vector initializer");
		}
		int i = 0;
		for (auto const &value : list) {
			components[i++] = value;
		}
	}
	constexpr Vector(std::array<T, N> const &array) :
		components(array) {
	}
	constexpr Vector(Vector const &) = default;
	constexpr Vector(Vector &&) = default;
	constexpr Vector &operator=(Vector const &) = default;
	constexpr Vector &operator=(Vector &&) = default;
	constexpr Vector &operator+=(Vector const &B) {
		forEachComponent([&](int i) { components[i] += B[i]; });
		return *this;
	}
	constexpr Vector &operator-=(Vector const &B) {
		forEachComponent([&](int i) { components[i] -= B[i]; });
		return *this;
	}
	constexpr Vector &operator*=(T const &B) {
		// B may refer to one of our components, as in v *= v[0].
		T const scalar = B;
		forEachComponent([&](int i) { components[i] *= scalar; });
		return *this;
	}
	constexpr Vector &operator/=(T const &B) {
		T const scalar = B;
		if constexpr (std::is_integral_v<T>) {
			forEachComponent([&](int i) { components[i] /= scalar; });
			return *this;
		} else {
			return (*this *= T(1) / scalar);
		}
	}
	template <int K>
	constexpr std::pair<Vector<T, K>, Vector<T, N - K>> split() const {
		static_assert(K >= 0 && K <= N);
		Vector<T, K> first;
		Vector<T, N - K> second;
		for (int i = 0; i < K; i++) {
			first[i] = components[i];
		}
		for (int i = K; i < N; i++) {
			second[i - K] = components[i];
		}
		return {first, second};
	}
	constexpr T dot(Vector const &A) const {
		auto const &B = *this;
		T sum = T(0);
		for (int k = 0; k < N; k++) {
			sum += A[k] * B[k];
		}
		return sum;
	}
	constexpr T max() const {
		static_assert(N > 0);
		using std::max;
		auto const &A = *this;
		T m = A[0];
		for (int k = 1; k < N; k++) {
			m = max(m, A[k]);
		}
		return m;
	}
	constexpr T min() const {
		static_assert(N > 0);
		using std::min;
		auto const &A = *this;
		T m = A[0];
		for (int k = 1; k < N; k++) {
			m = min(m, A[k]);
		}
		return m;
	}
	static constexpr Vector unit(int j) {
		Vector u;
		forEachComponent([&](int i) { u[i] = T(j == i); });
		return u;
	}
	friend constexpr Vector operator+(Vector const &A) {
		return A;
	}
	friend constexpr Vector operator-(Vector A) {
		forEachComponent([&](int i) { A[i] = -A[i]; });
		return A;
	}
	friend constexpr Vector operator+(Vector A, Vector const &B) {
		A += B;
		return A;
	}
	friend constexpr Vector operator-(Vector A, Vector const &B) {
		A -= B;
		return A;
	}
	friend constexpr Vector operator*(Vector A, T const &B) {
		A *= B;
		return A;
	}
	friend constexpr Vector operator*(T const &A, Vector B) {
		return B * A;
	}
	friend constexpr Vector operator/(Vector A, T const &B) {
		A /= B;
		return A;
	}
	friend constexpr T sqr(Vector const &A) {
		return A.dot(A);
	}
	friend constexpr T abs(Vector const &A) {
		using std::sqrt;
		return sqrt(sqr(A));
	}
	friend constexpr Vector norm(Vector A) {
		A /= abs(A);
		return A;
	}
	constexpr operator std::array<T, N>() const {
		return components;
	};

private:
	// Every caller writes component i and reads only that component or a
	// scalar snapshot. Self-aliasing therefore has no loop-carried dependence.
	// User-defined element types may have side effects, so keep their ordering.
	template <typename Function>
	static constexpr void forEachComponent(Function const &function) {
		if constexpr (std::is_arithmetic_v<T>) {
#if !defined(__CUDACC__) && !defined(__HIPCC__)
#if defined(__clang__)
#pragma clang loop vectorize(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
#endif
			for (int i = 0; i < N; i++) {
				function(i);
			}
		} else {
			for (int i = 0; i < N; i++) {
				function(i);
			}
		}
	}
	std::array<T, N> components;
};

template <typename T, int N, int M>
constexpr Vector<T, N + M> concatenate(Vector<T, N> const &A, Vector<T, M> const &B) {
	Vector<T, N + M> result;
	for (int i = 0; i < N; i++) {
		result[i] = A[i];
	}
	for (int i = 0; i < M; i++) {
		result[N + i] = B[i];
	}
	return result;
}

template <typename T, int N>
constexpr Vector<T, N + 1> concatenate(T const &A, Vector<T, N> const &B) {
	return concatenate(Vector<T, 1>{A}, B);
}

template <typename T, int N>
constexpr Vector<T, N + 1> concatenate(Vector<T, N> const &A, T const &B) {
	return concatenate(A, Vector<T, 1>{B});
}
