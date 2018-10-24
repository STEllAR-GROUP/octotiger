#include <silo.h>

#include <string>
#include <unordered_map>
#include <vector>

struct subgrid_t {
	DBquadmesh* mesh;
	std::unordered_map<std::string, DBquadvar*> vars;
};

int main(int argc, char* argv[]) {
	if (argc < 2) {
		printf("missing command line arguments\n");
		abort();
	}
	std::string silo_file(argv[1]);

	DBfile* db = DBOpen(silo_file.c_str(), DB_PDB, DB_READ);

	DBmultimesh* multimesh = DBGetMultimesh(db, "mesh");
	if (multimesh == NULL) {
		printf("mesh not found\n");
		abort();
	}

	std::vector<subgrid_t> grids(multimesh->nblocks);
	for (int i = 0; i < grids.size(); i++) {
		grids[i].mesh = DBGetQuadmesh(db, multimesh->meshnames[i]);
	}

	DBFreeMultimesh(multimesh);
	for (int i = 0; i < grids.size(); i++) {
		DBFreeQuadmesh(grids[i].mesh);
	}
	DBClose(db);

}
