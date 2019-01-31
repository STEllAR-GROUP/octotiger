#include <array>
#include <cassert>

#define NDIM 3

void to_id(unsigned long long& id, std::array<int, NDIM> x, int lev) {
	id = 1;
	for (int l = 0; l < lev; l++) {
		for (int d = 0; d < NDIM; d++) {
			id <<= 1;
			id |= ((x[d] >> l) & 1);
		}
	}
}

void from_id(unsigned long long id, std::array<int, NDIM>& x, int& lev) {
	for (int d = 0; d < NDIM; d++) {
		x[d] = 0;
	}
	for (lev = 0; id != 1; lev++) {
		printf( "%i %llo\n", lev, id);
		for (int d = NDIM - 1; d >= 0; d--) {
			x[d] <<= 1;
			x[d] |= (id & 1);
			id >>= 1;
		}
	}
}

int main(int argc, char* argv[]) {
	if (argc != 3) {
		printf("Usage: full_octree <string subgrid id> <number of additional levels>\n");
	}

	const char* ptr = argv[1];
	int dlev = std::atoi(argv[2]);
	assert(*ptr == '1');
	unsigned long long id = 1;
	ptr++;
	while (*ptr != '\0') {
		id <<= NDIM;
		id |= (*ptr - '0');
		ptr++;
	}
	std::array<int, NDIM> x;
	std::array<int, NDIM> y;
	int lev;
	from_id(id, x, lev);
	printf( "base_level is %i\n", lev);
	for (int i = 0; i < (1 << dlev); i++) {
		for (int j = 0; j < (1 << dlev); j++) {
			for (int k = 0; k < (1 << dlev); k++) {
				y[0] = (x[0] << dlev) + i;
				y[1] = (x[1] << dlev) + j;
				y[2] = (x[2] << dlev) + k;
				unsigned long long new_id;
				to_id(new_id,y,lev+dlev);
				printf( "the (%i,%i,%i) cell in subgrid %s has full octree id %o  \n", i, j, k, argv[1], new_id);
			}
		}
	}

}
