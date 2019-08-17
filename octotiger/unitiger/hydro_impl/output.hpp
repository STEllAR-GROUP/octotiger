#include "../unitiger.hpp"


template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::output(const hydro::state_type &U, const hydro::x_type &X, int num, safe_real t) {

	const auto dx = X[0][1] - X[0][0];
	FILE *fp = fopen("sums.dat", "at");
	auto sums = get_field_sums(U, dx);
	fprintf(fp, "%e ", (double) t);
	for (int f = 0; f < nf_; f++) {
		fprintf(fp, "%e ", (double) sums[f]);
	}
	fprintf(fp, "\n");
	fclose(fp);

	fp = fopen("mags.dat", "at");
	sums = get_field_mags(U, dx);
	fprintf(fp, "%e ", (double) t);
	for (int f = 0; f < nf_; f++) {
		fprintf(fp, "%e ", (double) sums[f]);
	}
	fprintf(fp, "\n");
	fclose(fp);

	std::string filename = "Y." + std::to_string(num);
	if (NDIM == 1) {
		filename += ".txt";
		FILE *fp = fopen(filename.c_str(), "wt");
		for (int i = 0; i < geo::H_NX; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				fprintf(fp, "%13.6e ", double(X[dim][i]));
			}
			for (int f = 0; f < nf_; f++) {
				fprintf(fp, "%13.6e ", double(U[f][i]));
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	} else {

		filename += ".silo";

		auto db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Uni-tiger", DB_PDB);

		auto opts = DBMakeOptlist(1);
		float ft = t;
		DBAddOption(opts, DBOPT_TIME, &ft);
		DBAddOption(opts, DBOPT_DTIME, &t);

		const char *coord_names[] = { "x", "y", "z" };
		safe_real coords[NDIM][geo::H_NX + 1];
		for (int i = 0; i < geo::H_NX + 1; i++) {
			const auto x = safe_real(i - geo::H_BW) / INX - safe_real(0.5);
			for (int dim = 0; dim < NDIM; dim++) {
				coords[dim][i] = x;
			}
		}
		void *coords_[] = { coords, coords + 1, coords + 2 };
		int dims1[] = { geo::H_NX + 1, geo::H_NX + 1, geo::H_NX + 1 };
		int dims2[] = { geo::H_NX, geo::H_NX, geo::H_NX };
		const auto &field_names = NDIM == 2 ? field_names2 : field_names3;
		DBPutQuadmesh(db, "quadmesh", coord_names, coords_, dims1, NDIM, DB_DOUBLE, DB_COLLINEAR, opts);
		for (int f = 0; f < nf_; f++) {
			DBPutQuadvar1(db, field_names[f], "quadmesh", U[f].data(), dims2, NDIM, NULL, 0, DB_DOUBLE, DB_ZONECENT, opts);
		}
		DBFreeOptlist(opts);
		DBClose(db);
	}

}
