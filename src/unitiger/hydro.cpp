//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/unitiger/unitiger.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct.hpp"
#include "octotiger/unitiger/hydro_impl/flux.hpp"
#include "octotiger/unitiger/hydro_impl/boundaries.hpp"
#include "octotiger/unitiger/hydro_impl/advance.hpp"
#include "octotiger/unitiger/hydro_impl/output.hpp"


#include <functional>


namespace hydro {



void output_cell2d(FILE *fp, const oct::array<safe_real, 9> &C, int joff, int ioff) {
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
