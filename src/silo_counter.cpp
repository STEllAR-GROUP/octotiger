
#include <silo.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
	if (argc != 3) {
		printf("silo_counter <prefix> <num_files>\n");
		return -1;
	}
	int maxblockindex;
	int maxblocks = 0;
	int count = atoi(argv[2]);
	char* prefix = argv[1];
	for (int i = 0; i < count; i++) {
		char* name;
		asprintf(&name, "%s.%i.silo", prefix, i);
		auto db = DBOpenReal(name, DB_HDF5, DB_READ);
		free(name);
		if (db) {
			auto mesh = DBGetMultimesh(db, "quadmesh");
			if( maxblocks < mesh->nblocks ){
				maxblockindex =i ;
				maxblocks = mesh->nblocks;
			}
			printf("%i has %i blocks current max is %i\n", i, int(mesh->nblocks), maxblocks);
			DBFreeMultimesh(mesh);
			DBClose(db);
		} else {
			printf( "%i is missing\n", i);
		}
	}
	printf( "max blocks = %i at %i\n", maxblocks, maxblockindex);
	return 0;
}
