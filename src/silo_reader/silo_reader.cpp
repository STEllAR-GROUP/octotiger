#include <silo.h>
#include <memory>
#include <string>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <iostream>
#include "./libeos/libeos.hpp"

using real = double;
using integer = long long int;
integer hydro, gravity, radiation, n_species;

template<class T>
T read_var(DBfile* db, const std::string& name) {
	T var;
	if (DBReadVar(db, name.c_str(), &var) != 0) {
		std::cout << "Unable to read variable " << name << "\n";
		abort();
	}
	return var;
}
;

#define READ_INT(n) \
	n = read_var<integer>(db, #n );

#define READ_REAL(n) \
	n = read_var<real>(db, #n );

int main(int argc, char* argv[]) {
	if (argc < 2) {
		printf("missing command line arguments\n");
		abort();
	}
	std::string silo_file(argv[1]);

	DBfile* db = DBOpen(silo_file.c_str(), DB_PDB, DB_READ);
	if (db == NULL) {
		printf("Unable to open %s\n", silo_file.c_str());
		abort();
	}
	DBmultimesh* multimesh = DBGetMultimesh(db, "mesh");
	if (multimesh == NULL) {
		printf("mesh not found\n");
		abort();
	}

	real cm, g, s, K;
	integer eos;
	READ_INT(hydro);
	READ_INT(radiation);
	READ_INT(gravity);
	READ_INT(n_species);
	READ_INT(eos);
	READ_REAL(cm);
	READ_REAL(g);
	READ_REAL(s);
	READ_REAL(K);

	if (eos < 0 || eos > 1) {
		printf("eos out of range\n");
		abort();
	}

	eos::set_eos_type(eos::eos_type(eos));
	eos::set_units(cm, g, s, K);

	std::vector<std::string> field_names;
	std::vector<std::string> hydro_names;
	if (hydro) {
		field_names.push_back("rho");
		field_names.push_back("tau");
		field_names.push_back("egas");
		field_names.push_back("sx");
		field_names.push_back("sy");
		field_names.push_back("sz");
		field_names.push_back("zx");
		field_names.push_back("zy");
		field_names.push_back("zz");
		for (int s = 0; s < n_species; s++) {
			field_names.push_back("spc_" + std::to_string(s + 1));
		}
	}
	hydro_names = field_names;
	if (gravity) {
		field_names.push_back("phi");
		field_names.push_back("gx");
		field_names.push_back("gy");
		field_names.push_back("gz");
	}
	if (radiation) {
		field_names.push_back("er");
		field_names.push_back("fx");
		field_names.push_back("fy");
		field_names.push_back("fz");
	}

	std::unordered_map<std::string, std::shared_ptr<real>> grid_sum;

	for (const auto& name : field_names) {
		grid_sum[name] = std::make_shared<real>(0);
	}

	for (int i = 0; i < multimesh->nblocks; i++) {
		std::unordered_map<std::string, DBquadvar*> vars;
		std::unordered_map<std::string, real> outflows;
		std::string meshname = multimesh->meshnames[i];
		auto* mesh = DBGetQuadmesh(db, meshname.c_str());
		if (mesh == NULL) {
			printf("Could not read mesh %s\n", meshname.c_str());
		}
		for (const auto& field : field_names) {
			const std::string var_name = field + std::string("_") + meshname;
			const std::string out_name = field + std::string("_outflow_") + meshname;
			vars[field] = DBGetQuadvar(db, var_name.c_str());
			if (vars[field] == NULL) {
				printf("Could not read variable %s\n", var_name.c_str());
				abort();
			}
			real o;
			if (DBReadVar(db, out_name.c_str(), &o) == 0) {
				outflows[field] = o;
			} else {
				printf("Could not read outflow variable %s\n", out_name.c_str());
				abort();
			}

			int lll = 0;
			real* coords[3];
			for (int d = 0; d < 3; d++) {
				coords[d] = static_cast<real*>(mesh->coords[d]);
			}

			real virial = 0.0;
			real virial_norm = 0.0;

			const real dx = coords[0][1] - coords[0][0];
			const real dv = dx * dx * dx;
			for (int k = 0; k < mesh->dims[2]; k++) {
				for (int j = 0; j < mesh->dims[1]; j++) {
					for (int i = 0; i < mesh->dims[0]; i++) {
						const real x = 0.5 * (coords[0][i + 1] + coords[0][i]);
						const real y = 0.5 * (coords[1][j + 1] + coords[1][j]);
						const real z = 0.5 * (coords[2][k + 1] + coords[2][k]);
#define VAR(f) ((double*) vars[f]->vals[0])
						for (const auto& name : hydro_names) {
							*grid_sum[name] += VAR(name)[lll] * dv;
						}
						if (hydro) {
							const real rho = VAR("rho")[lll];
							const real sx = VAR("sx")[lll];
							const real sy = VAR("sy")[lll];
							const real sz = VAR("sz")[lll];
							const real egas = VAR("egas")[lll];
							const real tau = VAR("tau")[lll];
							real etot = egas;
							*grid_sum["zx"] += rho * (y * sz - z * sy);
							*grid_sum["zy"] -= rho * (x * sz - z * sx);
							*grid_sum["zz"] += rho * (y * sz - z * sy);
							const real ek = 0.5 * (sx * sx + sy * sy + sz * sz) / 2.0;
							const real ein = egas - ek;
							const real p = eos::pressure_de(rho, egas, tau, ek);
							if (gravity) {
								real& phi = VAR("phi")[lll];
								etot += 0.5 * phi * rho;
								const real v1 = 3.0 * p;
								const real v2 = 3.0 * ek;
								const real v3 = 0.5 * rho * phi;
								virial += (v1 + v2 + v3) * dv;
								virial_norm += (std::abs(v1) + std::abs(v2) + std::abs(v3)) * dv;
							}

						}
						lll++;
					}
				}
			}

			DBFreeQuadmesh(mesh);
			for (auto v : vars) {
				DBFreeQuadvar(v.second);
			}
		}
	}

	DBFreeMultimesh(multimesh);
	DBClose(db);


	/* show results */
	for( const auto& nm : field_names ) {

	}



}
