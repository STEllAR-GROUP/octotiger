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
	using Primitive = typename M1::Primitives;
	using Flux = typename M1::ConservedFlux;
	using Base = Vector<Type, dimensionCount + 1>;
	static_assert(std::is_base_of_v<Base, State>);
	static_assert(std::is_base_of_v<Base, Primitive>);
	static_assert(std::is_base_of_v<Base, Flux>);
	static_assert(!std::is_default_constructible_v<M1>);
	// Single-state operations belong to the state objects. The method class
	// contains types, constants, and algorithms that combine multiple states.
	static_assert(!requires(State const& u) { M1::checkState(u); });
	static_assert(!requires(State const& u) { M1::close(u); });
	static_assert(!requires(State const& u, Type c) { M1::physicalFlux(u, 0, c); });
	static_assert(!requires(State const& u, Type c) { u.toPrimitives(c); });
	static_assert(!requires(Primitive const& q, Type c) { q.toConserved(c); });
	static_assert(!requires(State const& u, typename M1::SpatialVector const& momentum, Type x) {
		M1::couple(u, momentum, x, x, x, x, x, x, x, x);
	});

	{
		State const vacuum{};
		auto const zero = vacuum.toPrimitives().toConserved();
		for (int f = 0; f <= dimensionCount; ++f) require(zero[f] == 0, "vacuum roundtrip");
		for (int axis = -1; axis < dimensionCount; ++axis) {
			auto const unit = direction<Type, dimensionCount>(axis);
			for (Type const energy : {Type(1e-12), Type(1), Type(1e12)}) {
				for (Type const reducedFlux : {Type(0), Type(1e-6), Type(0.3), Type(0.95), Type(1)}) {
					State u{};
					u[0] = energy;
					for (int d = 0; d < dimensionCount; ++d) u[d + 1] = energy * reducedFlux * unit[d];
					u.checkState();
					auto const q = u.toPrimitives();
					q.checkState();
					auto const HOverE = (2 + std::sqrt(4 - 3 * static_cast<long double>(reducedFlux) * reducedFlux)) / 3;
					near(q[0], energy * HOverE, energy, "H from E and reduced flux");
					auto const conservedClosure = u.closure();
					auto const primitiveClosure = q.closure();
					for (auto const& closure : {conservedClosure, primitiveClosure}) {
						near(closure.H, energy * HOverE, energy, "member closure enthalpy");
						near(closure.pressure, energy * (HOverE - 1), energy, "member closure pressure H-E");
						long double betaSquared = 0;
						for (int d = 0; d < dimensionCount; ++d) {
							auto const beta = reducedFlux * static_cast<long double>(unit[d]) / HOverE;
							near(closure.beta[d], beta, 1, "member closure beta");
							betaSquared += beta * beta;
						}
						near(closure.beta2, betaSquared, 1, "member closure beta norm");
					}
					for (int d = 0; d < dimensionCount; ++d)
						near(q[d + 1], reducedFlux * static_cast<long double>(unit[d]) / HOverE, 1, "beta from reduced flux");
					auto const restored = q.toConserved();
					near(restored[0], u[0], energy, "conserved energy roundtrip");
					for (int d = 0; d < dimensionCount; ++d) {
						near(restored[d + 1], u[d + 1], energy, "normalized flux roundtrip");
						require(u.normalizedFlux()[d] == u[d + 1], "normalized flux accessor");
					}
				}
			}
		}
		Primitive q{};
		q[0] = Type(2.4);
		long double betaSquared = 0;
		for (int d = 0; d < dimensionCount; ++d) {
			q[d + 1] = Type((d % 2 ? -1 : 1) * 0.2 / (d + 1));
			betaSquared += static_cast<long double>(q[d + 1]) * q[d + 1];
		}
		auto const u = q.toConserved();
		// Independent inverse: E + P = H, P = H(1-beta^2)/4.
		near(u[0], q[0] * (3 + betaSquared) / 4, q[0], "E = H(3+beta^2)/4");
		for (int d = 0; d < dimensionCount; ++d)
			near(u[d + 1], static_cast<long double>(q[0]) * q[d + 1], q[0], "Q = H beta");
	}
	// Normalized conversion remains valid near the scalar range limit,
	// independently of the physical light speed used for subsequent transport.
	for (Type const beta : {Type(0), Type(0.2)}) {
		Primitive huge{};
		huge[0] = std::numeric_limits<Type>::max() * Type(0.6);
		huge[1] = beta;
		auto const u = huge.toConserved();
		near(u[0], static_cast<long double>(huge[0]) * ((3.L + static_cast<long double>(beta) * beta) / 4), huge[0], "representable large radiation energy");
		near(u[1], static_cast<long double>(huge[0]) * beta, huge[0], "representable large normalized flux");
		for (int d = 1; d < dimensionCount; ++d) require(u[d + 1] == 0, "large isotropic transverse flux");
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
	Primitive const scalarJoined = concatenate(first[0], tail);
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
	near(M1::transportTimestep(Type(7), Type(5), Type(0.5), Type(2)), 0.25L / dimensionCount, 1, "dimension-dependent transport CFL");
}

template <class Type, int dimensionCount>
void primitiveFluxPaths(Type c) {
	using M1 = Method<Type, dimensionCount>;
	using Primitive = typename M1::Primitives;
	std::mt19937 generator(2107 + dimensionCount);
	std::uniform_real_distribution<double> uniform(-1, 1);
	for (int sample = 0; sample < 128; ++sample) {
		std::array<Primitive, 2> q;
		for (auto& p : q) {
			p[0] = Type(std::exp(8 * uniform(generator)));
			Type normSquared = 0;
			for (int d = 0; d < dimensionCount; ++d) {
				p[d + 1] = Type(uniform(generator));
				normSquared += p[d + 1] * p[d + 1];
			}
			Type const scale = Type(0.95 * std::abs(uniform(generator))) / std::sqrt(normSquared);
			for (int d = 0; d < dimensionCount; ++d) p[d + 1] *= scale;
		}
		// Vacuum beta is immaterial, and must not change its characteristic
		// speeds or a neighboring non-vacuum state's HLL flux.
		if (sample % 7 == 0) q[0][0] = 0;
		auto const left = q[0].toConserved();
		auto const right = q[1].toConserved();
		auto const energyScale = std::max(q[0][0], q[1][0]);
		for (int normal = 0; normal < dimensionCount; ++normal) {
			Type const gridVelocity = Type(0.27 * uniform(generator)) * c;
			auto const direct = q[0].physicalFlux(normal, c, gridVelocity);
			auto const converted = left.physicalFlux(normal, c, gridVelocity);
			near(direct.minus, converted.minus, c, "primitive lower characteristic matches conserved path");
			near(direct.plus, converted.plus, c, "primitive upper characteristic matches conserved path");
			auto const directHll = M1::hll(q[0], q[1], normal, c, gridVelocity);
			auto const convertedHll = M1::hll(left, right, normal, c, gridVelocity);
			auto const unitSpeedFlux = q[0].physicalFlux(normal, Type(1), gridVelocity / c);
			auto const unitSpeedHll = M1::hll(q[0], q[1], normal, Type(1), gridVelocity / c);
			for (int field = 0; field <= dimensionCount; ++field) {
				auto const scale = static_cast<long double>(energyScale) * c;
				near(direct.flux[field], converted.flux[field], scale, "primitive physical flux matches conserved path");
				near(directHll[field], convertedHll[field], scale, "primitive HLL matches conserved path");
				// All normalized flux components scale with one power of c.
				near(direct.flux[field] / c, unitSpeedFlux.flux[field], energyScale, "normalized physical-flux light-speed scaling");
				near(directHll[field] / c, unitSpeedHll[field], energyScale, "normalized HLL light-speed scaling");
			}
		}
	}
}

template <class Type, int dimensionCount>
void reconstruction() {
	using M1 = Method<Type, dimensionCount>;
	using Primitive = typename M1::Primitives;
	Primitive minus{}, center{}, plus{};
	minus[0] = 2;
	center[0] = 3;
	plus[0] = 4;
	minus[1] = Type(0.1);
	center[1] = Type(0.2);
	plus[1] = Type(0.3);
	auto const [left, right] = M1::reconstruct(minus, center, plus);
	auto const qLeft = left.toPrimitives();
	auto const qRight = right.toPrimitives();
	near(qLeft[0], 2.5, 3, "linear H reconstruction on lower face");
	near(qRight[0], 3.5, 3, "linear H reconstruction on upper face");
	near(qLeft[1], 0.15, 1, "linear beta reconstruction on lower face");
	near(qRight[1], 0.25, 1, "linear beta reconstruction on upper face");
	near(M1::minmodTheta(Type(1), Type(1)), 1, 1, "minmod linear slope");
	require(M1::minmodTheta(Type(1), Type(-1)) == 0, "minmod local extremum");

	std::mt19937 generator(1979 + dimensionCount);
	std::uniform_real_distribution<double> uniform(-1, 1);
	for (int sample = 0; sample < 512; ++sample) {
		std::array<Primitive, 3> q;
		for (auto& p : q) {
			p[0] = Type(std::exp(4 * uniform(generator)));
			long double normSquared = 0;
			for (int d = 0; d < dimensionCount; ++d) {
				p[d + 1] = Type(uniform(generator));
				normSquared += static_cast<long double>(p[d + 1]) * p[d + 1];
			}
			auto const radius = sample % 2 ? Type(1) : Type(std::abs(uniform(generator)));
			for (int d = 0; d < dimensionCount; ++d) p[d + 1] *= radius / Type(std::sqrt(normSquared));
		}
		auto const faces = M1::reconstruct(q[0], q[1], q[2]);
		auto const primitiveFaces = M1::reconstructPrimitives(q[0], q[1], q[2]);
		auto const directLeft = primitiveFaces.first.toConserved();
		auto const directRight = primitiveFaces.second.toConserved();
		for (int field = 0; field <= dimensionCount; ++field) {
			require(directLeft[field] == faces.first[field], "primitive and conserved lower reconstruction paths");
			require(directRight[field] == faces.second[field], "primitive and conserved upper reconstruction paths");
		}
		auto const a = faces.first.toPrimitives();
		auto const b = faces.second.toPrimitives();
		for (auto const& u : {faces.first, faces.second}) {
			u.checkState();
			require(u[0] >= 0, "reconstructed nonnegative energy");
			long double reducedSquared = 0;
			for (int d = 0; d < dimensionCount; ++d) {
				auto const reduced = static_cast<long double>(u[d + 1]) / u[0];
				reducedSquared += reduced * reduced;
			}
			require(reducedSquared <= 1 + 128 * std::numeric_limits<Type>::epsilon(), "reconstructed state inside flux cone");
		}
		near(a[0] + b[0], 2.L * q[1][0], q[1][0], "H face midpoint");
		for (int d = 0; d < dimensionCount; ++d)
			near(a[d + 1] + b[d + 1], 2.L * q[1][d + 1], 1, "beta face midpoint");
	}
}

template <class Type, int dimensionCount>
void sourceConservation(Type c) {
	using M1 = Method<Type, dimensionCount>;
	for (Type const eta : {Type(0), Type(1e-6), Type(0.1), Type(1), Type(1e6)}) {
		for (Type const radiation : {Type(0), Type(0.25), Type(50)}) {
			for (bool const marshak : {false, true}) {
				Type const gas = Type(2);
				Type const alpha = Type(0.3);
				auto const result = M1::thermalExchange(gas, radiation, eta, std::log(alpha), marshak);
				require(result.gas >= 0 && result.radiation >= 0, "thermal exchange positivity");
				near(result.gas + result.radiation, static_cast<long double>(gas) + radiation, gas + radiation, "thermal energy conservation");
				auto const emission = marshak ? static_cast<long double>(result.gas)
					: static_cast<long double>(alpha) * std::pow(static_cast<long double>(result.gas), 4);
				auto const reference = (radiation + static_cast<long double>(eta) * emission) / (1.L + eta);
				near(result.radiation, reference, gas + radiation, "implicit thermal equation", 512);
			}
		}
	}
	Type const rho = Type(2);
	typename M1::SpatialVector momentum{};
	typename M1::ConservedState u{};
	u[0] = Type(4);
	Type kinetic = 0;
	for (int d = 0; d < dimensionCount; ++d) {
		momentum[d] = Type(0.025 * (d + 1));
		u[d + 1] = Type(0.1 * (d + 1));
		kinetic += momentum[d] * momentum[d] / (2 * rho);
	}
	Type const internalEnergy = Type(3);
	Type const gasEnergy = internalEnergy + kinetic;
	for (Type const dt : {Type(0), Type(1e-3), Type(0.1)}) {
		for (Type const chiA : {Type(0), Type(0.4)}) {
			auto const result = u.couple(momentum, gasEnergy, internalEnergy,
				rho, chiA, Type(0.7), dt, c, std::log(Type(0.04)));
			near(result.gasEnergy + result.radiation[0], static_cast<long double>(gasEnergy) + u[0], gasEnergy + u[0], "coupled total energy conservation");
			Type newKinetic = 0;
			for (int d = 0; d < dimensionCount; ++d) {
				near(result.momentum[d] + result.radiation[d + 1] / c,
					static_cast<long double>(momentum[d]) + static_cast<long double>(u[d + 1]) / c,
					1, "coupled total momentum conservation");
				newKinetic += result.momentum[d] * result.momentum[d] / (2 * rho);
			}
			near(result.internalEnergy + newKinetic, result.gasEnergy, gasEnergy, "consistent internal and total gas energy");
			if (dt == 0) {
				for (int field = 0; field <= dimensionCount; ++field)
					require(result.radiation[field] == u[field], "zero-time source identity");
			}
		}
	}
}

template <class Type, int dimensionCount>
void physicalSourceReference(Type c) {
	using M1 = Method<Type, dimensionCount>;
	typename M1::ConservedState u{};
	typename M1::SpatialVector momentum{};
	Type const rho = Type(2);
	Type const internalEnergy = Type(3);
	u[0] = Type(4);
	Type kinetic = 0;
	std::array<long double, dimensionCount> physicalFlux{}, velocity{}, direction{};
	long double reducedSquared = 0;
	for (int d = 0; d < dimensionCount; ++d) {
		u[d + 1] = Type(0.1 * (d + 1));
		momentum[d] = Type(0.05 * (d + 1));
		kinetic += momentum[d] * momentum[d] / (2 * rho);
		physicalFlux[d] = static_cast<long double>(c) * u[d + 1];
		velocity[d] = static_cast<long double>(momentum[d]) / rho;
		direction[d] = physicalFlux[d] / c / u[0];
		reducedSquared += direction[d] * direction[d];
	}
	Type const gasEnergy = internalEnergy + kinetic;
	// Independent reference in physical (E,F) units: construct P from
	// Levermore's E/f tensor, then apply physical flux damping and recoil.
	auto const chi = (3 + 4 * reducedSquared) / (5 + 2 * std::sqrt(4 - 3 * reducedSquared));
	Type const chiT = Type(0.7);
	for (Type const dt : {Type(0.025) / c, Type(0.25) / c, Type(0.1)}) {
		for (Type const chiA : {Type(0), Type(0.4)}) {
			auto const result = u.couple(momentum, gasEnergy, internalEnergy,
				rho, chiA, chiT, dt, c, Type(0), true);
			auto const etaT = static_cast<long double>(dt) * c * chiT;
			long double velocityFlux = 0, kineticChange = 0;
			for (int d = 0; d < dimensionCount; ++d) {
				long double advectiveFlux = u[0] * velocity[d];
				for (int k = 0; k < dimensionCount; ++k) {
					auto const pressure = u[0] * ((d == k ? (1 - chi) / 2 : 0)
						+ (3 * chi - 1) / 2 * direction[d] * direction[k] / reducedSquared);
					advectiveFlux += pressure * velocity[k];
				}
				auto const updatedFlux = (physicalFlux[d] + etaT * advectiveFlux) / (1 + etaT);
				auto const recoil = (physicalFlux[d] - updatedFlux) / c / c;
				kineticChange += recoil * (velocity[d] + recoil / (2 * rho));
				velocityFlux += velocity[d] * physicalFlux[d];
				near(result.radiation[d + 1], updatedFlux / c, updatedFlux / c, "physical-F damping reference converted to Q", 512);
				near(result.momentum[d], momentum[d] + recoil, momentum[d], "physical-F recoil reference", 512);
			}
			auto const work = dt * (2.L * chiA - chiT) * velocityFlux / c;
			auto const radiationStar = u[0] + work;
			auto const gasStar = internalEnergy - work - kineticChange;
			auto const etaA = static_cast<long double>(dt) * c * chiA;
			// Marshak equilibrium radiation energy Eeq=e makes the exchange a 2x2 linear
			// system. Solve it directly, without using the implementation's solver.
			auto const expectedGas = ((1 + etaA) * gasStar + etaA * radiationStar) / (1 + 2 * etaA);
			auto const expectedRadiation = gasStar + radiationStar - expectedGas;
			near(result.internalEnergy, expectedGas, internalEnergy, "physical-F source reference gas internal energy", 512);
			near(result.radiation[0], expectedRadiation, u[0], "physical-F source reference radiation energy", 512);
			near(result.gasEnergy, gasEnergy + (u[0] - expectedRadiation), gasEnergy, "physical-F source reference gas total energy", 512);
		}
	}
}

template <class Type, int dimensionCount>
void run() {
	conversionsAndVectors<Type, dimensionCount>();
	reconstruction<Type, dimensionCount>();
	for (Type const c : {Type(1), Type(7), Type(2.99792458e10)}) {
		fluxesAndWaves<Type, dimensionCount>(c);
		primitiveFluxPaths<Type, dimensionCount>(c);
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
