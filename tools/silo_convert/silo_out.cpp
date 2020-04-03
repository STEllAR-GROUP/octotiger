/*
 * silo_out.cpp
 *
 *  Created on: Mar 12, 2020
 *      Author: dmarce1
 */


#include "./silo_convert.hpp"


void silo_output::set_vars(silo_vars_t vars) {
	int one = 1;
	const int ns = vars.n_species;
	const int nnode = vars.node_count;

	DBWrite(db, "n_species", &vars.n_species, &one, 1, DB_LONG_LONG);
	DBWrite(db, "node_count", &vars.node_count, &one, 1, DB_LONG_LONG);
	DBWrite(db, "leaf_count", &vars.leaf_count, &one, 1, DB_LONG_LONG);
	DBWrite(db, "node_list", vars.node_list.data(), &nnode, 1, DB_LONG_LONG);
	DBWrite(db, "node_positions", vars.node_positions.data(), &nnode, 1, DB_LONG_LONG);
	DBWrite(db, "omega", &vars.omega, &one, 1, DB_DOUBLE);
	DBWrite(db, "atomic_number", vars.atomic_number.data(), &ns, 1, DB_DOUBLE);
	DBWrite(db, "atomic_mass", vars.atomic_mass.data(), &ns, 1, DB_DOUBLE);
	DBWrite(db, "X", vars.X.data(), &ns, 1, DB_DOUBLE);
	DBWrite(db, "Z", vars.Z.data(), &ns, 1, DB_DOUBLE);
	DBWrite(db, "version", &vars.version, &one, 1, DB_LONG_LONG);
	DBWrite(db, "code_to_g", &vars.code_to_g, &one, 1, DB_DOUBLE);
	DBWrite(db, "code_to_s", &vars.code_to_s, &one, 1, DB_DOUBLE);
	DBWrite(db, "code_to_cm", &vars.code_to_cm, &one, 1, DB_DOUBLE);
	DBWrite(db, "eos", &vars.eos, &one, 1, DB_LONG_LONG);
	DBWrite(db, "gravity", &vars.gravity, &one, 1, DB_LONG_LONG);
	DBWrite(db, "hydro", &vars.hydro, &one, 1, DB_LONG_LONG);
	DBWrite(db, "radiation", &vars.radiation, &one, 1, DB_LONG_LONG);
	DBWrite(db, "output_frequency", &vars.output_frequency, &one, 1, DB_DOUBLE);
	DBWrite(db, "problem", &vars.problem, &one, 1, DB_LONG_LONG);
	DBWrite(db, "refinement_floor", &vars.refinement_floor, &one, 1, DB_DOUBLE);
	DBWrite(db, "cgs_time", &vars.cgs_time, &one, 1, DB_DOUBLE);
	DBWrite(db, "rotational_time", &vars.rotational_time, &one, 1, DB_DOUBLE);
	DBWrite(db, "xscale", &vars.xscale, &one, 1, DB_DOUBLE);
	DBWrite(db, "cycle", &vars.cycle, &one, 1, DB_LONG_LONG);
	DBWrite(db, "hostname", vars.hostname, &HOST_NAME_LEN, 1, DB_CHAR);
	DBWrite(db, "timestamp", &vars.timestamp, &one, 1, DB_LONG_LONG);
	DBWrite(db, "epoch", &vars.epoch, &one, 1, DB_LONG_LONG);
	DBWrite(db, "locality_count", &vars.locality_count, &one, 1, DB_LONG_LONG);
	DBWrite(db, "thread_count", &vars.thread_count, &one, 1, DB_LONG_LONG);
	DBWrite(db, "step_count", &vars.step_count, &one, 1, DB_LONG_LONG);
	DBWrite(db, "time_elapsed", &vars.time_elapsed, &one, 1, DB_LONG_LONG);
	DBWrite(db, "steps_elapsed", &vars.steps_elapsed, &one, 1, DB_LONG_LONG);


}


void silo_output::set_mesh_count(int n) {
	mesh_count = n;
}



