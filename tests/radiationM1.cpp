// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
// Standalone: g++ -std=c++20 -O2 -I. tests/radiationM1.cpp -o radiationM1Tests
#include "octotiger/radiation/m1.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

std::size_t checkCount = 0;

void require(bool condition, char const* message) {
	++checkCount;
	if (!condition) throw std::runtime_error(message);
}

// Use the tested scalar's precision and a caller-supplied physical scale;
// a fixed absolute tolerance would miss errors in weak radiation fields.
template <class Type>
void near(Type value, long double expected, long double scale, char const* message,
	long double epsilonCount = 256) {
	++checkCount;
	auto const tolerance = epsilonCount * std::numeric_limits<Type>::epsilon()
		* std::max(std::abs(expected), std::abs(scale));
	if (!std::isfinite(value) || std::abs(static_cast<long double>(value) - expected) > tolerance) {
		std::ostringstream out;
		out.precision(std::numeric_limits<long double>::max_digits10);
		out << message << ": got " << value << ", expected " << expected
			<< ", tolerance " << tolerance;
		throw std::runtime_error(out.str());
	}
}

template <class Type, int dimensionCount>
using Method = RadiationM1<Type, dimensionCount>;

template <class Type, int dimensionCount>
typename Method<Type, dimensionCount>::SpatialVector direction(int axis = -1) {
	typename Method<Type, dimensionCount>::SpatialVector result{};
	long double normSquared = 0;
	for (int d = 0; d < dimensionCount; ++d) {
		result[d] = Type(axis < 0 ? d + 1 : d == axis);
		normSquared += static_cast<long double>(result[d]) * result[d];
	}
	for (int d = 0; d < dimensionCount; ++d) result[d] /= Type(std::sqrt(normSquared));
	return result;
}

template <class Type, int dimensionCount>
void conversionsAndVectors() {
	using M1 = Method<Type, dimensionCount>;
	using State = typename M1::ConservedState;
	using Flux = typename M1::ConservedFlux;
	using Base = Vector<Type, dimensionCount + 1>;
	static_assert(std::is_base_of_v<Base, State>);
	static_assert(std::is_same_v<Base, Flux>);
	static_assert(!std::is_default_constructible_v<M1>);

	State const vacuum{};
	vacuum.checkState("vacuum");
	auto const vacuumClosure = vacuum.closure();
	near(vacuumClosure.chi, 1.L / 3, 1, "vacuum closure chi");
	near(vacuumClosure.isotropic, 0, 1, "vacuum isotropic pressure");
	near(vacuumClosure.directed, 0, 1, "vacuum directed pressure");

	for (int axis = -1; axis < dimensionCount; ++axis) {
		auto const unit = direction<Type, dimensionCount>(axis);
		for (Type const energy : {Type(1e-12), Type(1), Type(1e12)}) {
			for (Type const reducedFlux : {Type(0), Type(1e-6), Type(0.3), Type(0.95), Type(1)}) {
				State u{};
				u[0] = energy;
				for (int d = 0; d < dimensionCount; ++d)
					u[d + 1] = energy * reducedFlux * unit[d];
				u.checkState("closure test");
				require(M1::admissible(u), "valid state recognized as admissible");
				auto const closure = u.closure();
				long double const f2 = static_cast<long double>(reducedFlux) * reducedFlux;
				long double const s = std::sqrt(4 - 3 * f2);
				long double const chi = (3 + 4 * f2) / (5 + 2 * s);
				near(closure.chi, chi, 1, "Levermore closure chi");
				near(closure.isotropic,
					static_cast<long double>(energy) * (1 - f2) / (s + 1), energy,
					"Levermore isotropic coefficient");
				near(closure.directed,
					static_cast<long double>(energy) * 3 / (s + 2), energy,
					"Levermore directed coefficient");
				for (int d = 0; d < dimensionCount; ++d) {
					near(closure.reducedFlux[d],
						static_cast<long double>(reducedFlux) * unit[d], 1,
						"closure reduced flux");
					require(u.normalizedFlux()[d] == u[d + 1], "normalized flux accessor");
				}
				if constexpr (dimensionCount == 3) {
					long double pressureTrace = 0;
					for (int normal = 0; normal < dimensionCount; ++normal)
						pressureTrace += u.physicalFlux(normal, Type(1)).flux[normal + 1];
					near(Type(pressureTrace), energy, energy, "M1 pressure tensor trace");
				}
			}
		}
	}

	State a{};
	for (int f = 0; f <= dimensionCount; ++f) a[f] = Type(f + 2);
	State const original = a;
	a *= a[0];
	for (int f = 0; f <= dimensionCount; ++f) require(a[f] == 2 * original[f], "aliased scalar multiplication");
	a = original;
	a /= a[0];
	for (int f = 0; f <= dimensionCount; ++f) require(a[f] == original[f] / 2, "aliased scalar division");
	a = original;
	a += a;
	for (int f = 0; f <= dimensionCount; ++f) require(a[f] == 2 * original[f], "self addition");
	a -= a;
	for (int f = 0; f <= dimensionCount; ++f) require(a[f] == 0, "self subtraction");
	State const sum = original + Type(2) * original;
	State const difference = sum - original;
	Flux const flux = difference / Type(2);
	for (int f = 0; f <= dimensionCount; ++f) require(flux[f] == original[f], "inherited vector arithmetic");
	auto const [first, tail] = original.template split<1>();
	State const joined = concatenate(first, tail);
	State const scalarJoined = concatenate(first[0], tail);
	for (int f = 0; f <= dimensionCount; ++f) {
		require(joined[f] == original[f], "state split and concatenate");
		require(scalarJoined[f] == original[f], "scalar and vector concatenate");
	}
	std::array<Type, dimensionCount + 1> const array = original;
	State const fromArray(array);
	for (int f = 0; f <= dimensionCount; ++f) require(fromArray[f] == original[f], "array interoperation");
}

template <class Type, int dimensionCount>
void fluxesAndWaves(Type c) {
	using M1 = Method<Type, dimensionCount>;
	using State = typename M1::ConservedState;
	auto const unit = direction<Type, dimensionCount>();
	Type const energy = Type(2.5);
	for (Type const reducedFlux : {Type(0), Type(0.1), Type(0.7), Type(1)}) {
		State u{};
		u[0] = energy;
		for (int d = 0; d < dimensionCount; ++d) u[d + 1] = energy * reducedFlux * unit[d];
		long double fSquared = 0;
		std::array<long double, dimensionCount> f{};
		for (int d = 0; d < dimensionCount; ++d) {
			f[d] = static_cast<long double>(u[d + 1]) / energy;
			fSquared += f[d] * f[d];
		}
		fSquared = std::min(1.L, fSquared);
		// Levermore's E/f closure is independent of the implementation's H/beta algebra.
		auto const chi = (3 + 4 * fSquared) / (5 + 2 * std::sqrt(4 - 3 * fSquared));
		for (int normal = 0; normal < dimensionCount; ++normal) {
			for (Type const gridVelocity : {Type(0), Type(0.13) * c}) {
				auto const result = u.physicalFlux(normal, c, gridVelocity);
				auto const riemann = M1::hll(u, u, normal, c, gridVelocity);
				near(result.flux[0], static_cast<long double>(c) * u[normal + 1] - gridVelocity * static_cast<long double>(energy), c * static_cast<long double>(energy), "energy transport flux");
				for (int d = 0; d < dimensionCount; ++d) {
					auto const tensor = (d == normal ? (1 - chi) / 2 : 0)
						+ (fSquared > 0 ? (3 * chi - 1) / 2 * f[normal] * f[d] / fSquared : 0);
					auto const reference = static_cast<long double>(c) * energy * tensor
						- static_cast<long double>(gridVelocity) * u[d + 1];
					near(result.flux[d + 1], reference, static_cast<long double>(c) * energy, "Levermore pressure tensor");
				}
				for (int field = 0; field <= dimensionCount; ++field)
					near(riemann[field], result.flux[field], static_cast<long double>(c) * energy, "equal-state HLL consistency");
				require(result.minus <= result.plus, "ordered wave speeds");
				auto const allowance = 128 * std::numeric_limits<Type>::epsilon() * c;
				require(result.minus >= -c - gridVelocity - allowance && result.plus <= c - gridVelocity + allowance, "causal characteristic bounds");
				if (reducedFlux == 0) {
					near(result.minus, -static_cast<long double>(c) / std::sqrt(3.L) - gridVelocity, c, "isotropic lower wave speed");
					near(result.plus, static_cast<long double>(c) / std::sqrt(3.L) - gridVelocity, c, "isotropic upper wave speed");
				}
				if (reducedFlux == 1) {
					// Oblique unit vectors carry O(epsilon) norm error, and the
					// coalescing characteristic roots respond as sqrt(epsilon).
					auto const rootTolerance = dimensionCount == 1 ? 256.L
						: 8 / std::sqrt(static_cast<long double>(std::numeric_limits<Type>::epsilon()));
					near(result.minus, static_cast<long double>(c) * unit[normal] - gridVelocity, c, "streaming lower wave speed", rootTolerance);
					near(result.plus, static_cast<long double>(c) * unit[normal] - gridVelocity, c, "streaming upper wave speed", rootTolerance);
				}
			}
		}
	}
	State left{}, right{};
	left[0] = Type(2);
	right[0] = Type(1);
	for (int normal = 0; normal < dimensionCount; ++normal) {
		auto const flux = M1::hll(left, right, normal, c);
		near(flux[0], c / (2 * std::sqrt(3.L)), c, "isotropic HLL energy jump");
		for (int d = 0; d < dimensionCount; ++d)
			near(flux[d + 1], d == normal ? static_cast<long double>(c) / 2 : 0, c, "isotropic HLL pressure flux");
		for (int sign : {-1, 1}) {
			auto const gridVelocity = Type(sign * 2) * c;
			auto const upwind = (sign < 0 ? left : right).physicalFlux(normal, c, gridVelocity);
			auto const moving = M1::hll(left, right, normal, c, gridVelocity);
			for (int field = 0; field <= dimensionCount; ++field)
				require(moving[field] == upwind.flux[field], "supersonic HLL selects upstream state");
		}
	}
	near(M1::transportTimestep(Type(7), Type(5), Type(0.5), Type(2)), 0.5L / dimensionCount, 1, "dimension-dependent transport CFL");
}

template <class Type, int dimensionCount>
void randomizedConservedFluxPaths(Type c) {
	using M1 = Method<Type, dimensionCount>;
	using State = typename M1::ConservedState;
	std::mt19937 generator(2107 + dimensionCount);
	std::uniform_real_distribution<double> uniform(-1, 1);
	for (int sample = 0; sample < 128; ++sample) {
		std::array<State, 2> states;
		for (auto& u : states) {
			u[0] = Type(std::exp(8 * uniform(generator)));
			long double normSquared = 0;
			for (int d = 0; d < dimensionCount; ++d) {
				u[d + 1] = Type(uniform(generator));
				normSquared += static_cast<long double>(u[d + 1]) * u[d + 1];
			}
			Type const scale = u[0] * Type(0.95 * std::abs(uniform(generator)))
				/ Type(std::sqrt(normSquared));
			for (int d = 0; d < dimensionCount; ++d) u[d + 1] *= scale;
			u.checkState("random state");
		}
		if (sample % 7 == 0) states[0] = {};
		auto const energyScale = std::max(states[0][0], states[1][0]);
		for (int normal = 0; normal < dimensionCount; ++normal) {
			Type const gridVelocity = Type(0.27 * uniform(generator)) * c;
			auto const direct = states[0].physicalFlux(normal, c, gridVelocity);
			auto const directHll = M1::hll(states[0], states[1], normal, c, gridVelocity);
			auto const unitSpeedFlux = states[0].physicalFlux(normal, Type(1), gridVelocity / c);
			auto const unitSpeedHll = M1::hll(
				states[0], states[1], normal, Type(1), gridVelocity / c);
			for (int field = 0; field <= dimensionCount; ++field) {
				require(std::isfinite(directHll[field]), "finite random HLL flux");
				near(direct.flux[field] / c, unitSpeedFlux.flux[field], energyScale, "normalized physical-flux light-speed scaling");
				near(directHll[field] / c, unitSpeedHll[field], energyScale, "normalized HLL light-speed scaling");
			}
			auto const zeroStep = M1::limitFlux(
				states[0], states[1], directHll, normal, c, gridVelocity, Type(0));
			for (int field = 0; field <= dimensionCount; ++field)
				require(zeroStep[field] == directHll[field], "zero-step flux limiter identity");
		}
	}
}

template <class Type, int dimensionCount>
void reconstruction() {
	using M1 = Method<Type, dimensionCount>;
	using State = typename M1::ConservedState;
	State minus{}, center{}, plus{};
	minus[0] = 2;
	center[0] = 3;
	plus[0] = 4;
	minus[1] = Type(0.1);
	center[1] = Type(0.2);
	plus[1] = Type(0.3);
	auto const [left, right] = M1::reconstruct(minus, center, plus);
	near(left[0], 2.5, 3, "linear E reconstruction on lower face");
	near(right[0], 3.5, 3, "linear E reconstruction on upper face");
	near(left[1], 0.15, 1, "linear Q reconstruction on lower face");
	near(right[1], 0.25, 1, "linear Q reconstruction on upper face");
	near(M1::plmSlope(Type(1), Type(1)), 1, 1, "Athena PLM linear slope");
	near(M1::plmSlope(Type(1), Type(2)), 4.L / 3, 1, "Athena PLM harmonic slope");
	require(M1::plmSlope(Type(1), Type(-1)) == 0, "Athena PLM local extremum");

	std::mt19937 generator(1979 + dimensionCount);
	std::uniform_real_distribution<double> uniform(-1, 1);
	for (int sample = 0; sample < 512; ++sample) {
		std::array<State, 3> states;
		for (auto& u : states) {
			u[0] = Type(std::exp(4 * uniform(generator)));
			long double normSquared = 0;
			for (int d = 0; d < dimensionCount; ++d) {
				u[d + 1] = Type(uniform(generator));
				normSquared += static_cast<long double>(u[d + 1]) * u[d + 1];
			}
			auto const radius = sample % 2 ? Type(1) : Type(std::abs(uniform(generator)));
			for (int d = 0; d < dimensionCount; ++d)
				u[d + 1] *= u[0] * radius / Type(std::sqrt(normSquared));
		}
		auto const faces = M1::reconstruct(states[0], states[1], states[2]);
		for (auto const& u : {faces.first, faces.second}) {
			u.checkState("reconstructed face");
			require(u[0] >= 0, "reconstructed nonnegative energy");
			require(M1::admissible(u), "reconstructed state inside flux cone");
		}
		for (int field = 0; field <= dimensionCount; ++field) {
			near(faces.first[field] + faces.second[field], 2.L * states[1][field],
				states[1][0], "conserved face midpoint");
			auto const low = std::min({states[0][field], states[1][field], states[2][field]});
			auto const high = std::max({states[0][field], states[1][field], states[2][field]});
			require(faces.first[field] >= low && faces.first[field] <= high,
				"lower face component monotonicity");
			require(faces.second[field] >= low && faces.second[field] <= high,
				"upper face component monotonicity");
		}
	}
}

template <class Type, int dimensionCount>
void sourceConservation(Type c) {
	using M1 = Method<Type, dimensionCount>;
	(void)c;
	for (Type const ratio : {Type(1), Type(0.1), Type(0.001)}) {
		for (Type const eta : {Type(0), Type(1e-6), Type(0.1), Type(1), Type(1e6)}) {
			for (Type const radiation : {Type(0), Type(0.25), Type(50)}) {
				for (bool const marshak : {false, true}) {
					for (Type const theta : {Type(0.5), Type(0.51), Type(1)}) {
						Type const gas = Type(2);
						Type const alpha = Type(0.3);
						auto const result = M1::thermalExchange(gas, radiation, eta,
							std::log(alpha), marshak, ratio, theta);
						require(result.gas >= 0 && result.radiation >= 0,
							"thermal exchange positivity");
						near(result.gas + result.radiation / ratio,
							static_cast<long double>(gas) + radiation / ratio,
							gas + radiation / ratio, "RSLA thermal invariant", 1024);
					}
				}
			}
		}
	}
	near(M1::damp(Type(2), Type(0.3), Type(0.7), Type(1)),
		2.3L / 1.7L, 2, "backward-Euler damping");
	auto const stiff = M1::damp(Type(2), Type(0), Type(1e6), Type(0.51));
	near(stiff, 2.L / (1 + 1e6L), 2, "stiff theta fallback to backward Euler", 1024);
}

template <class Type, int dimensionCount>
void physicalSourceReference(Type c) {
	using M1 = Method<Type, dimensionCount>;
	typename M1::ConservedState u{};
	typename M1::SpatialVector velocity{};
	u[0] = Type(4);
	for (int d = 0; d < dimensionCount; ++d) {
		u[d + 1] = Type(0.1 * (d + 1));
		velocity[d] = Type(0.025 * (d + 1));
	}
	Type const ratio = Type(0.03);
	Type const chiA = Type(0.4);
	Type const chiT = Type(0.7);
	auto const source = M1::explicitSource(u, velocity, chiA, chiT, c, ratio, true);
	long double velocityFlux = 0;
	for (int d = 0; d < dimensionCount; ++d)
		velocityFlux += static_cast<long double>(velocity[d]) * u[d + 1];
	near(source[0], ratio * (2.L * chiA - chiT) * velocityFlux,
		u[0], "S&O explicit work source");
	for (int d = 0; d < dimensionCount; ++d) {
		near(source[d + 1], ratio * chiT * (4.L / 3) * u[0] * velocity[d],
			u[0], "S&O explicit momentum source");
	}
	auto const disabled = M1::explicitSource(u, velocity, chiA, chiT, c, ratio, false);
	for (int field = 0; field <= dimensionCount; ++field)
		require(disabled[field] == 0, "disabled velocity source is zero");
}

template <class Type, int dimensionCount>
void run() {
	conversionsAndVectors<Type, dimensionCount>();
	reconstruction<Type, dimensionCount>();
	for (Type const c : {Type(1), Type(7), Type(2.99792458e10)}) {
		fluxesAndWaves<Type, dimensionCount>(c);
		randomizedConservedFluxPaths<Type, dimensionCount>(c);
		sourceConservation<Type, dimensionCount>(c);
		physicalSourceReference<Type, dimensionCount>(c);
	}
}

} // namespace

int main() {
	try {
		run<float, 1>();
		run<float, 2>();
		run<float, 3>();
		run<double, 1>();
		run<double, 2>();
		run<double, 3>();
		run<long double, 1>();
		run<long double, 2>();
		run<long double, 3>();
		std::cout << "RadiationM1: " << checkCount << " checks passed\n";
		return 0;
	} catch (std::exception const& error) {
		std::cerr << "RadiationM1: " << error.what() << '\n';
		return 1;
	}
}
