#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "octotiger/scf_geometry.hpp"

namespace {

void require_close(double actual, double expected, double tolerance, char const *message) {
	if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
		throw std::runtime_error(message);
	}
}

void check_rotating_equilibrium() {
	double const com = 0.125;
	double const omega = 0.75;
	double const bernoulli = -2.0;
	std::array<octotiger::scf::SamplePosition, 4> const positions = {{{-0.25, 0.03125, 0.125},
		{0.375, -0.0625, -0.25}, {0.125, 0.125, 0.0}, {0.5, 0.0, 0.125}}};
	for (auto const &position : positions) {
		double const radius_squared = std::pow(position.x - com, 2) + position.y * position.y;
		// This manufactured rotating equilibrium has h = 1 - R^2/2 - z^2/3,
		// and a polytropic density rho = h^(3/2). The projected line coordinate
		// cannot determine its centrifugal potential for off-axis samples.
		double const gravity = bernoulli - 1.0 + 0.5 * (1.0 + omega * omega) * radius_squared
			+ position.z * position.z / 3.0;
		double const expected_enthalpy = 1.0 - 0.5 * radius_squared - position.z * position.z / 3.0;
		double const recovered_enthalpy = bernoulli - position.effective_potential(gravity, omega, com);
		require_close(std::pow(recovered_enthalpy, 1.5), std::pow(expected_enthalpy, 1.5), 2.0e-15,
			"SCF line sample does not preserve a rotating density fixed point");
	}
}

void check_half_cell_regression() {
	double const dx = 0.03125;
	double const omega = 0.968;
	double const gravity = -2.0;
	double const peak_enthalpy = 0.25;
	octotiger::scf::SamplePosition const peak = {-0.375, 0.5 * dx, 0.5 * dx};
	double const bernoulli = peak_enthalpy + peak.effective_potential(gravity, omega, 0.0);
	// Evaluate the cell density update independently, using the actual cell's
	// cylindrical radius. This is the consistency condition that SCF requires.
	double const update_potential = gravity - 0.5 * omega * omega * (peak.x * peak.x + peak.y * peak.y);
	double const recovered_density = std::pow(bernoulli - update_potential, 1.5);
	double const peak_density = std::pow(peak_enthalpy, 1.5);
	require_close(recovered_density / peak_density, 1.0, 3.0e-15,
		"Half-cell sample changes density at a purported SCF fixed point");

	double const projected_bernoulli = peak_enthalpy + gravity - 0.5 * omega * omega * peak.x * peak.x;
	double const omitted_enthalpy = projected_bernoulli - update_potential - peak_enthalpy;
	require_close(omitted_enthalpy, omega * omega * dx * dx / 8.0, 5.0e-16,
		"Regression fixture does not reproduce the omitted off-axis centrifugal term");
	if (std::pow((projected_bernoulli - update_potential) / peak_enthalpy, 1.5) - 1.0 < 1.0e-4) {
		throw std::runtime_error("Regression fixture is too weak to detect the original SCF density error");
	}
}

void check_rotation_center() {
	octotiger::scf::SamplePosition const position = {0.375, 0.125, 0.25};
	octotiger::scf::SamplePosition const translated = {2.375, 0.125, -0.5};
	double const potential = position.effective_potential(-1.5, 0.875, 0.125);
	require_close(translated.effective_potential(-1.5, 0.875, 2.125), potential, 0.0,
		"Centrifugal potential depends on the coordinate origin or the axial position");
	require_close(position.cylindrical_radius_squared(0.125), 0.078125, 0.0,
		"Centrifugal potential ignores the supplied rotation center");
}

}

int main() {
	check_rotating_equilibrium();
	check_half_cell_regression();
	check_rotation_center();
	std::puts("SCF sample geometry checks passed");
}
