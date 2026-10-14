// Copyright (c) 2026 AUTHORS. Distributed under the Boost Software License, Version 1.0.
#include "octotiger/test_problems/radiation.hpp"
#include "octotiger/test_problems/radiation/reference.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/math/Debug.hpp"

radiationTests::Parameters radiationTestParameters() {
	FpeGuard fpeGuard{};
	radiationTests::Parameters p;
	p.length=2*opts().xscale; p.c=physcon().c;
	p.time=opts().stop_time * opts().radiation_reduced_c_factor;
	p.chi=opts().radTestChi; p.width=opts().radTestWidth;
	p.background=opts().radTestBackground; p.amplitude=opts().radTestAmplitude;
	p.luminosity=opts().radTestLuminosity;
	return p;
}
namespace {
radiationTests::ReferenceData const& reference() {
	// C++ static initialization is thread safe; each locality reads once.
	// The generator runs separately. No FFTs or mutable time cache in Octo-TIGER.
	static auto const data=radiationTests::ReferenceData::read(opts().radReference);
	return data;
}
std::vector<Real> sample(Real x,Real y,Real z,Real dx,Real t) {
	FpeGuard fpeGuard{};
	auto const p=radiationTestParameters();
	// Fixed-medium RSLA is the physical-c solution at the radiation clock εt.
	// Flux amplitudes and equilibrium coefficients retain physical c.
	t *= opts().radiation_reduced_c_factor;
	radiationTests::Point const pos{x,y,z};
	radiationTests::State r;
	switch (opts().problem) {
	case RADIATION_STREAMING_WAVE: r=radiationTests::streamingWave(pos,t,dx,p); break;
	case RADIATION_STREAMING_FRONT: r=radiationTests::streamingFront(pos,t,dx,p); break;
	case RADIATION_GAUSSIAN_PULSE: r=reference().average(pos,dx,t); break;
	case RADIATION_EQUILIBRIUM_SPHERE: r=radiationTests::sphereAverage(pos,dx,p); break;
	default: throw std::runtime_error("Invalid radiation regression problem");
	}
	// Uniform, stationary gas. These are transport tests in a prescribed medium.
	std::vector<Real> u(opts().n_fields+NRF,0);
	u[rho_i]=u[spc_i]=u[egas_i]=u[tau_i]=1;
	for (int f=0;f<4;++f) {
    u[opts().n_fields+f]=r[f];
  }
  return u;
}
}
void validateRadiationTest() {
	FpeGuard fpeGuard{};
	auto const p=radiationTestParameters(); p.validate();
	if (!opts().radiation || opts().hydro || opts().gravity || opts().omega!=0) {
    throw std::runtime_error("Radiation regression tests require radiation=on, hydro=off, gravity=off, omega=0");
  }
  bool const sphere=opts().problem==RADIATION_EQUILIBRIUM_SPHERE;
	if (opts().periodic==sphere || opts().rad_implicit!=radiationFixedMediumProblem()) {
    throw std::runtime_error("Radiation regression boundary/source flags disagree with the selected problem");
  }
  if (opts().problem==RADIATION_GAUSSIAN_PULSE) {
		auto const& r=reference();
		if (opts().max_level<0 || opts().max_level>20 || r.n!=std::uint64_t(INX)*(std::uint64_t(1)<<opts().max_level)) {
      throw std::runtime_error("Generate Gaussian reference at INX * 2^max_level cells per side");
    }
    auto same=[](double a,double b){
			FpeGuard fpeGuard{};
			return std::abs(a-b)<=1e-12*std::max(std::abs(a),std::abs(b));
		};
		if (!same(p.length,r.p.length) || !same(p.c,r.p.c) || !same(p.chi,r.p.chi) ||
			!same(p.width,r.p.width) || !same(p.background,r.p.background) ||
			!same(p.amplitude,r.p.amplitude) || !same(p.time,r.p.time)) {
      throw std::runtime_error("Gaussian reference parameters do not match this simulation");
    }
  }
}
std::vector<Real> radiationRegressionInit(Real x,Real y,Real z,Real dx) {
	FpeGuard fpeGuard{};
	return sample(x,y,z,dx,0);
}
std::vector<Real> radiationRegressionAnalytic(Real x,Real y,Real z,Real t) {
	FpeGuard fpeGuard{};
	// Uniform-grid tests compare cell averages at the configured finest level.
	Real const dx=2*opts().xscale/(INX*std::ldexp(1.,opts().max_level));
	return sample(x,y,z,dx,t);
}
std::vector<Real> radiationRegressionAnalyticCell(Real x,Real y,Real z,Real t,Real dx) {
	return sample(x,y,z,dx,t);
}

radiationHydroTests::Case radiationHydroCase() {
	switch (opts().problem) {
	case RADIATION_SUBCRITICAL_SHOCK: return radiationHydroTests::Case::subcriticalShock;
	case RADIATION_SUPERCRITICAL_SHOCK: return radiationHydroTests::Case::supercriticalShock;
	case RADIATION_BLAST_WAVE: return radiationHydroTests::Case::blastWave;
	default: throw std::invalid_argument("Not a radiation-hydrodynamics benchmark");
	}
}

void validateRadiationHydroTest() {
	const radiationHydroTests::Units units{opts().code_to_g, opts().code_to_cm, opts().code_to_s};
	units.validate();
	const auto p = radiationHydroTests::setup(radiationHydroCase());
	if (!opts().radiation || !opts().hydro || !opts().rad_implicit || opts().gravity ||
		opts().periodic || opts().omega != 0 || opts().eos != IDEAL) {
		throw std::invalid_argument("Radiation-hydro benchmarks require radiation=on, hydro=on, rad_implicit=on, gravity=off, periodic=off, omega=0, eos=IDEAL");
	}
	if (std::abs(opts().xscale * units.length / p.halfLength - 1) > 1e-12) {
		throw std::invalid_argument("Radiation-hydro benchmark xscale does not match its physical domain");
	}
	if (opts().atomic_mass[0] != 1 || opts().atomic_number[0] != 1 || opts().reflect_bc || opts().inflow_bc) {
		throw std::invalid_argument("Radiation-hydro benchmarks require atomic_mass=1, atomic_number=1 and their problem-specific boundaries (reflect_bc=off, inflow_bc=off)");
	}
}

std::vector<Real> radiationHydroInit(Real x, Real y, Real z, Real dx) {
	FpeGuard fpeGuard{};
	const auto state = radiationHydroTests::initialState(radiationHydroCase(), {x, y, z}, dx,
		{opts().code_to_g, opts().code_to_cm, opts().code_to_s});
	std::vector<Real> u(opts().n_fields + NRF, 0);
	u[rho_i] = u[spc_i] = state.rho;
	u[egas_i] = state.gasEnergy;
	u[tau_i] = std::pow(state.internalEnergy, 1 / radiationHydroTests::gamma);
	for (int d = 0; d < NDIM; ++d) {
		u[sx_i + d] = state.momentum[d];
	}
	u[lx_i] = y * u[sz_i] - z * u[sy_i];
	u[ly_i] = z * u[sx_i] - x * u[sz_i];
	u[lz_i] = x * u[sy_i] - y * u[sx_i];
	for (int f = 0; f < NRF; ++f) {
		u[opts().n_fields + f] = state.radiation[f];
	}
	return u;
}

void validateRadiationCoupledWave() {
	if (!opts().radiation || !opts().hydro || !opts().rad_implicit || opts().gravity ||
		!opts().periodic || opts().omega != 0 || opts().eos != IDEAL ||
		opts().radiation_reduced_c_factor != 1) {
		throw std::invalid_argument("RADIATION_COUPLED_WAVE requires radiation=on, hydro=on, rad_implicit=on, gravity=off, periodic=on, omega=0, eos=IDEAL and radiation_reduced_c_factor=1");
	}
	if (opts().n_species != 1 || opts().atomic_mass[0] != 1 || opts().atomic_number[0] != 1) {
		throw std::invalid_argument("RADIATION_COUPLED_WAVE requires one fully ionized hydrogen species (n_species=1, atomic_mass=1, atomic_number=1)");
	}
	if (!std::isfinite(opts().xscale) || !(opts().xscale > 0) ||
		!std::isfinite(opts().radTestChi) || opts().radTestChi < 0 ||
		!std::isfinite(opts().radTestAmplitude) || opts().radTestAmplitude < 0 ||
		!(opts().radTestAmplitude < Real(.25))) {
		throw std::invalid_argument("RADIATION_COUPLED_WAVE requires xscale>0, rad_test_chi>=0 and 0<=rad_test_amplitude<0.25");
	}
	const radiationHydroTests::Units units{opts().code_to_g, opts().code_to_cm, opts().code_to_s};
	units.validate();
}

std::vector<Real> radiationCoupledWaveInit(Real x, Real y, Real z, Real dx) {
	FpeGuard fpeGuard{};
	// A smooth, compressible, periodic material/radiation wave. At the
	// background temperature 10^7 K, e_gas=4*a_rad*T^4 gives equal gas and
	// radiation heat capacities. The velocity is at most 0.01c; the gas is
	// nonrelativistically hot. No analytic time evolution is assumed: temporal
	// convergence uses an independently refined solution on the same mesh.
	constexpr Real background_temperature = 1.e7;
	constexpr Real mean_molecular_weight = Real(.5);
	const Real gamma = grid::get_fgamma();
	const Real radiation_constant = 4*physcon().sigma/physcon().c;
	const Real background_radiation = radiation_constant*std::pow(background_temperature,4);
	const Real background_density = 4*background_radiation*(gamma-1)*mean_molecular_weight
		*physcon().mh/(physcon().kb*background_temperature);
	const Real wave_number = std::acos(Real(-1))/opts().xscale;
	const Real half_cell_phase = Real(.5)*wave_number*dx;
	const Real average_factor = half_cell_phase == 0 ? Real(1)
		: std::sin(half_cell_phase)/half_cell_phase;
	const Real cosine = average_factor*std::cos(wave_number*x);
	const Real sine = average_factor*std::sin(wave_number*x);
	const Real perturbation = 1+opts().radTestAmplitude*cosine;
	// Average only the sinusoidal mode before constructing the thermodynamics.
	// The nonlinear products below enforce equilibrium of the discrete cell
	// state; they are not exact cell averages of a continuum LTE profile.
	const Real rho = background_density*perturbation;
	const Real temperature = background_temperature*perturbation;
	const Real internal = rho*physcon().kb*temperature/
		(mean_molecular_weight*physcon().mh*(gamma-1));
	const Real beta = Real(.01)*sine;
	const Real velocity = beta*physcon().c;

	// Exact equilibrium of the implemented mixed-frame M1 source with equal
	// absorption and extinction. If b is M1's boost parameter, Q/E=4b/(3+b^2)
	// and beta=b/(1+b^2) make cQ=(E+P_xx)v. The energy source then vanishes
	// for E=a_rad*T^4/(1-beta*Q/E). Starting from these conservative cell
	// values avoids injecting an artificial stiff initial relaxation layer.
	const Real boost = 2*beta/(1+std::sqrt(1-4*beta*beta));
	const Real reduced_flux = 4*boost/(3+boost*boost);
	const Real radiation_energy = radiation_constant*std::pow(temperature,4)
		/(1-beta*reduced_flux);
	std::vector<Real> u(opts().n_fields+NRF,0);
	u[rho_i] = u[spc_i] = rho;
	u[sx_i] = rho*velocity;
	u[egas_i] = internal+Real(.5)*rho*velocity*velocity;
	u[tau_i] = std::pow(internal,1/gamma);
	u[ly_i] = z*u[sx_i];
	u[lz_i] = -y*u[sx_i];
	u[opts().n_fields] = radiation_energy;
	u[opts().n_fields+1] = physcon().c*reduced_flux*radiation_energy;
	return u;
}
