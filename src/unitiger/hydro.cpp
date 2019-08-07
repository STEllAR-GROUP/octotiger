//============================================================================
// Name        : hydro.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#ifdef NOHPX
#include "../../octotiger/octotiger/unitiger/unitiger.hpp"
#include "../../octotiger/octotiger/unitiger/hydro.hpp"
#else
#include "../../octotiger/unitiger/unitiger.hpp"
#include "../../octotiger/unitiger/hydro.hpp"
#endif

#include <functional>

namespace hydro {



void output_cell2d(FILE *fp, const std::array<safe_real, 9> &C, int joff, int ioff) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i < 2)
				fprintf(fp, "%i %i %e %i %i %e \n", i + ioff, j + joff, double(C[3 * i + j]), 1, 0, double(C[3 * (i + 1) + j] - C[3 * i + j]));
			if (j < 2)
				fprintf(fp, "%i %i %e %i %i %e \n", i + ioff, j + joff, double(C[3 * i + j]), 0, 1, double(C[3 * i + j + 1] - C[3 * i + j]));
		}
	}
}


}
