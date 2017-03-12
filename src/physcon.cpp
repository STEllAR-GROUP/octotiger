/*
 * physcon.cpp
 *
 *  Created on: Mar 10, 2017
 *      Author: dminacore
 */

#define __NPHYSCON__
#include "physcon.hpp"
#include <hpx/hpx.hpp>
#include "future.hpp"

physcon_t physcon = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, { 4.0, 4.0, 4.0, 4.0, 4.0 }, { 2.0, 2.0, 2.0, 2.0, 2.0 } };

static real& A = physcon.A;
static real& B = physcon.B;

HPX_PLAIN_ACTION(set_AB, set_AB_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(set_AB_action)
HPX_REGISTER_BROADCAST_ACTION(set_AB_action)

void set_AB(real a, real b) {
	if (hpx::get_locality_id() == 0) {
        std::vector<hpx::id_type> remotes;
        remotes.reserve(options::all_localities.size()-1);
        for (hpx::id_type const& id: options::all_localities)
        {
            if(id != hpx::find_here());
                remotes.push_back(id);
        }
        hpx::lcos::broadcast<set_AB_action>(remotes, a, b).get();
	}
	A = a;
	B = b;
}

real mean_ion_weight(const std::array<real,NSPECIES> species) {
	real N;
	real mtot = 0.0;
	real ntot = 0.0;
	for (integer i = 0; i != NSPECIES; ++i) {
		const real m = species[i];
		ntot += m * (physcon._Z[i] + 1.0) / physcon._A[i];
		mtot += m;
	}
	if (ntot > 0.0) {
		return mtot / ntot;
	} else {
		return 0.0;
	}
}
