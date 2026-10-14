// Radiation regression integration; analytic kernels and binary I/O have no HPX dependency.
#pragma once
#include "octotiger/options.hpp"
#include "octotiger/test_problems/radiation/profiles.hpp"
#include "radiation/hydro_benchmarks.hpp"

inline bool radiationRegressionProblem() {
	auto const p=opts().problem;
	return p==RADIATION_STREAMING_WAVE || p==RADIATION_STREAMING_FRONT ||
		p==RADIATION_GAUSSIAN_PULSE || p==RADIATION_EQUILIBRIUM_SPHERE;
}
inline bool radiationFixedMediumProblem() {
	return opts().problem==RADIATION_GAUSSIAN_PULSE || opts().problem==RADIATION_EQUILIBRIUM_SPHERE;
}
inline bool radiationShockProblem() {
	return opts().problem == RADIATION_SUBCRITICAL_SHOCK || opts().problem == RADIATION_SUPERCRITICAL_SHOCK;
}
inline bool radiationHydroProblem() {
	return radiationShockProblem() || opts().problem == RADIATION_BLAST_WAVE;
}
OCTOTIGER_EXPORT radiationTests::Parameters radiationTestParameters();
OCTOTIGER_EXPORT void validateRadiationTest();
OCTOTIGER_EXPORT std::vector<Real> radiationRegressionInit(Real x,Real y,Real z,Real dx);
OCTOTIGER_EXPORT std::vector<Real> radiationRegressionAnalytic(Real x,Real y,Real z,Real t);
OCTOTIGER_EXPORT std::vector<Real> radiationRegressionAnalyticCell(Real x,Real y,Real z,Real t,Real dx);
OCTOTIGER_EXPORT radiationHydroTests::Case radiationHydroCase();
OCTOTIGER_EXPORT void validateRadiationHydroTest();
OCTOTIGER_EXPORT std::vector<Real> radiationHydroInit(Real x, Real y, Real z, Real dx);
OCTOTIGER_EXPORT void validateRadiationCoupledWave();
OCTOTIGER_EXPORT std::vector<Real> radiationCoupledWaveInit(Real x, Real y, Real z, Real dx);
