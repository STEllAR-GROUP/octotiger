// Copyright (c) 2026 AUTHORS. Distributed under the Boost Software License, Version 1.0.
#include "octotiger/test_problems/radiation.hpp"
#include "octotiger/test_problems/radiation/reference.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/math/Debug.hpp"

radiationTests::Parameters radiationTestParameters() {
	FpeGuard fpeGuard{};
	radiationTests::Parameters p;
	p.length=2*opts().xscale; p.c=physcon().c; p.time=opts().stop_time*opts().rad_c_ratio;
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
	radiationTests::Point const pos{x,y,z};
	t*=opts().rad_c_ratio; // RSLA rescales time, not the physical normalization of F.
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
	for (int f=0;f<4;++f) u[opts().n_fields+f]=r[f];
	return u;
}
}
void validateRadiationTest() {
	FpeGuard fpeGuard{};
	auto const p=radiationTestParameters(); p.validate();
	if (!opts().radiation || opts().hydro || opts().gravity || opts().omega!=0 || !opts().unigrid)
		throw std::runtime_error("Radiation regression tests require radiation=on, hydro=off, gravity=off, omega=0, unigrid=on");
	bool const sphere=opts().problem==RADIATION_EQUILIBRIUM_SPHERE;
	if (opts().periodic==sphere || opts().rad_implicit!=radiationFixedMediumProblem())
		throw std::runtime_error("Radiation regression boundary/source flags disagree with the selected problem");
	if (opts().problem==RADIATION_GAUSSIAN_PULSE) {
		auto const& r=reference();
		if (opts().max_level<0 || opts().max_level>20 || r.n!=std::uint64_t(INX)*(std::uint64_t(1)<<opts().max_level))
			throw std::runtime_error("Generate Gaussian reference at INX * 2^max_level cells per side");
		auto same=[](double a,double b){
			FpeGuard fpeGuard{};
			return std::abs(a-b)<=1e-12*std::max(std::abs(a),std::abs(b));
		};
		if (!same(p.length,r.p.length) || !same(p.c,r.p.c) || !same(p.chi,r.p.chi) ||
			!same(p.width,r.p.width) || !same(p.background,r.p.background) ||
			!same(p.amplitude,r.p.amplitude) || !same(p.time,r.p.time))
			throw std::runtime_error("Gaussian reference parameters do not match this simulation (reference time must be stop_time * rad_c_ratio)");
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
