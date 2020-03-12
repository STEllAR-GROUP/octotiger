/*
 * silo_out.cpp
 *
 *  Created on: Mar 12, 2020
 *      Author: dmarce1
 */


#include "./silo_convert.hpp"


void silo_output::set_vars(silo_vars_t vars) {
	int one = 1;
	DBWrite( db, "omega", &vars.omega, &one, 1, DB_DOUBLE);
	DBWrite( db, "n_species", &vars.n_species, &one, 1, DB_LONG_LONG);
	const int ns = vars.n_species;
	DBWrite( db, "atomic_mass", vars.atomic_mass.data(), &ns, 1, DB_DOUBLE);
	DBWrite( db, "atomic_number", vars.atomic_number.data(), &ns, 1, DB_DOUBLE);

}

