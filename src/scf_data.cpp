/*
 * scf_data.cpp
 *
 *  Created on: Oct 20, 2016
 *      Author: dmarce1
 */

#include "octotiger/scf_data.hpp"

#include <cmath>
#include <limits>

scf_data_t::scf_data_t() {
	donor_phi_max = accretor_phi_max = l1_phi = -std::numeric_limits < real > ::max();
	accretor_mass = donor_mass = donor_central_enthalpy = accretor_central_enthalpy = ZERO;
	phiA = phiB = phiC = 0.0;
	virial_sum = virial_norm = 0.0;
	donor_central_density = accretor_central_density = 0.0;
	m = m_x = 0.0;
}


void scf_data_t::accumulate(const scf_data_t& other) {
	if (phiA > other.phiA) {
		phiA = other.phiA;
		xA = other.xA;
	}
	if (phiB > other.phiB) {
		phiB = other.phiB;
		xB = other.xB;
	}
	if (phiC > other.phiC) {
		phiC = other.phiC;
		xC = other.xC;
	}
	m += other.m;
	m_x += other.m_x;
	virial_sum += other.virial_sum;
	virial_norm += other.virial_norm;
	phiA = std::min(phiA, other.phiA);
	phiB = std::min(phiB, other.phiB);
	phiC = std::min(phiC, other.phiC);
	entC = std::max(entC, other.entC);
	donor_phi_max = std::max(donor_phi_max, other.donor_phi_max);
	accretor_phi_max = std::max(accretor_phi_max, other.accretor_phi_max);
	accretor_mass += other.accretor_mass;
	donor_mass += other.donor_mass;
	if (other.donor_central_enthalpy > donor_central_enthalpy) {
		donor_phi_min = other.donor_phi_min;
		donor_x = other.donor_x;
		donor_central_enthalpy = other.donor_central_enthalpy;
		donor_central_density = other.donor_central_density;
	}
	if (other.accretor_central_enthalpy > accretor_central_enthalpy) {
		accretor_phi_min = other.accretor_phi_min;
		accretor_x = other.accretor_x;
		accretor_central_enthalpy = other.accretor_central_enthalpy;
		accretor_central_density = other.accretor_central_density;
	}
	if (other.l1_phi > l1_phi) {
		l1_phi = other.l1_phi;
		l1_x = other.l1_x;
	}
}

