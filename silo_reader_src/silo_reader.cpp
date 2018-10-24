#include <silo.h>

#include <string>
#include <unordered_map>
#include <vector>

using real = double;
using integer = long long int;

struct subgrid_t {
	DBquadmesh* mesh;
	std::unordered_map<std::string, DBquadvar*> vars;
	std::unordered_map<std::string, real> outflow;
};

integer hydro, gravity, radiation, n_species;

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

	if (DBReadVar(db, "hydro", &hydro) != 0) {
		printf("Unable to read variable: hydro\n");
		abort();
	}
	if (DBReadVar(db, "hydro", &gravity) != 0) {
		printf("Unable to read variable: gravity\n");
		abort();
	}
	if (DBReadVar(db, "radiation", &radiation) != 0) {
		printf("Unable to read variable: radiation\n");
		abort();
	}
	if (DBReadVar(db, "n_species", &n_species) != 0) {
		printf("Unable to read variable: n_species\n");
		abort();
	}
	printf( "hydro:     %i\n", int(hydro));
	printf( "gravity:   %i\n", int(gravity));
	printf( "radiation: %i\n", int(radiation));
	printf( "n species: %i\n", int(n_species));

	std::vector<std::string> field_names;
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

	std::vector<subgrid_t> grids(multimesh->nblocks);
	for (int i = 0; i < grids.size(); i++) {
		std::string meshname = multimesh->meshnames[i];
		grids[i].mesh = DBGetQuadmesh(db, meshname.c_str());
		if (grids[i].mesh == NULL) {
			printf("Could not read mesh %s\n", meshname.c_str());
		}
		for (const auto& field : field_names) {
			const std::string var_name = field + std::string("_") + meshname;
			const std::string out_name = field + std::string("_outflow_") + meshname;
			grids[i].vars[field] = DBGetQuadvar(db, var_name.c_str());
			if (grids[i].vars[field] == NULL) {
				printf("Could not read variable %s\n", var_name.c_str());
				abort();
			}
			real o;
			if (DBReadVar(db, out_name.c_str(), &o) == 0) {
				grids[i].outflow[field] = o;
			} else {
				printf("Could not read outflow variable %s\n", out_name.c_str());
				abort();
			}
		}
	}

	DBFreeMultimesh(multimesh);
	for (int i = 0; i < grids.size(); i++) {
		DBFreeQuadmesh(grids[i].mesh);
	}
	DBClose(db);

}
