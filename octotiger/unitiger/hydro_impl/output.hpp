//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "../unitiger.hpp"


template<int NDIM, int INX, class PHYS>
void hydro_computer<NDIM, INX, PHYS>::output(const hydro::state_type &U, const hydro::x_type &X, int num, safe_real t) {

	const auto dx = X[0][1] - X[0][0];
	FILE *fp = fopen("sums.dat", "at");
	auto sums = get_field_sums(U, dx);
	fprintf(fp, "%e ", (double) t);
	for (int f = 0; f < nf_; f++) {
		fprintf(fp, "%e ", (double) sums[f]);
	}
	fprintf(fp, "\n");
	fclose(fp);

//	fp = fopen("mags.dat", "at");
//	sums = get_field_mags(U, dx);
//	fprintf(fp, "%e ", (double) t);
//	for (int f = 0; f < nf_; f++) {
//		fprintf(fp, "%e ", (double) sums[f]);
//	}
//	fprintf(fp, "\n");
//	fclose(fp);

	std::string filename = "Y." + std::to_string(num);
	if (NDIM == 1) {
		filename += ".txt";
		FILE *fp = fopen(filename.c_str(), "wt");
		for (int i = 0; i < geo::H_NX; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				fprintf(fp, "%17.10e ", double(X[dim][i]));
			}
			for (int f = 0; f < nf_; f++) {
				fprintf(fp, "%17.10e ", double(U[f][i]));
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	} else {

		filename += ".silo";

		auto db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Uni-tiger", DB_HDF5);

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
		const auto &field_names = NDIM == 2 ? PHYS::field_names2 : PHYS::field_names3;
		DBPutQuadmesh(db, "quadmesh", coord_names, coords_, dims1, NDIM, DB_DOUBLE, DB_COLLINEAR, opts);
		for (int f = 0; f < nf_; f++) {
			DBPutQuadvar1(db, field_names[f], "quadmesh", U[f].data(), dims2, NDIM, NULL, 0, DB_DOUBLE, DB_ZONECENT, opts);
		}
		DBFreeOptlist(opts);
		DBClose(db);
	}

}
template<int NDIM, int INX, class PHYS>
void hydro_computer<NDIM, INX, PHYS>::outputQ(const hydro::recon_type<NDIM> &Q, int num, std::string test_type)
{       
        std::string filename;
        
        if (num > 0)
                filename = test_type + "_Q_test_" + std::to_string(num) + ".data";
        else    
                filename = test_type + "_Q_test_final.data";
        FILE *fp = fopen(filename.c_str(), "wb");
        for (int f = 0; f < nf_; f++) {
                for (int i = 0; i < geo::H_NX; i++) {
                                fwrite(Q[f][i].data(), sizeof(double), geo::NDIR, fp);
                        }
                }
        fclose(fp);
}
template<int NDIM, int INX,class PHYS>
void hydro_computer<NDIM, INX, PHYS>::outputU(const hydro::state_type &U, int num, std::string test_type){
        std::string filename;

        if (num > 0)
                filename = test_type + "_U_test_" + std::to_string(num) + ".data";
        else
                filename = test_type + "_U_test_final.data";
        FILE *fp = fopen(filename.c_str(), "wb");
        for (int f = 0; f < nf_; f++) {
                fwrite(U[f].data(), sizeof(double), geo::H_NX, fp);
        }
        fclose(fp);
}
template<int NDIM, int INX, class PHYS>
void hydro_computer<NDIM, INX, PHYS>::outputF(const hydro::flux_type &Fl, int num, std::string test_type){
        std::string filename;

        if (num > 0)
                filename = test_type + "_F_test_" + std::to_string(num) + ".data";
        else
                filename = test_type + "_F_test_final.data";
        FILE *fp = fopen(filename.c_str(), "wb");
        for (int i=0; i < NDIM; i++)
        {
                for (int f = 0; f < nf_; f++) {
                fwrite(Fl[i][f].data(), sizeof(double), geo::H_N3, fp);
                }
        }
        fclose(fp);
}
template<int NDIM, int INX,class PHYS>
int hydro_computer<NDIM, INX, PHYS>::compareQ(const hydro::recon_type<NDIM> &Q, int num, std::string test_type)
{
        double dline[geo::NDIR];
        std::string filename;

        if (num > 0)
                filename = test_type + "_Q_test_" + std::to_string(num) + ".data";
        else
                filename = test_type + "_Q_test_final.data";
        FILE *fp = fopen(filename.c_str(), "rb");
        if (fp != nullptr)
        {
                for (int f = 0; f < nf_; f++) {
                        for (int i = 0; i < geo::H_NX; i++) {
                                fread(&dline, sizeof(double), geo::NDIR, fp);
                                for (int j = 0; j < geo::NDIR; j++)
                                {
                                        if (std::abs(Q[f][i][j] - dline[j])/(1e-12+dline[j]+Q[f][i][j]) > 1e-12)
                                        {
                                                printf("differnt Q values in: (%d of %d), (%d of %d), (%d of %d). The values are (old, new): %f, %f\n",
                                                                f, nf_, i, geo::H_NX, j, geo::NDIR, dline[j], double(Q[f][i][j]));
                                                fclose(fp);
                                                return 0;
                                        }
                                }
                        }
                }
                fclose(fp);
        }
        else
                return -1;
        return 1;
}
template<int NDIM, int INX, class PHYS>
int hydro_computer<NDIM, INX, PHYS>::compareF(const hydro::flux_type &Fl, int num, std::string test_type)
{
        double dline[geo::H_N3];
        std::string filename;

        if (num > 0)
                filename = test_type + "_F_test_" + std::to_string(num) + ".data";
        else
                filename = test_type + "_F_test_final.data";
        FILE *fp = fopen(filename.c_str(), "rb");
        if (fp != nullptr)
        {
                for (int i = 0; i < NDIM; i++) {
                        for (int f = 0; f < nf_; f++) {
                                fread(&dline, sizeof(double), geo::H_N3, fp);
                                for (int j = 0; j < geo::H_N3; j++)
                                {
                                        if (std::abs(Fl[i][f][j] - dline[j])/(1e-12+dline[j]+Fl[i][f][j]) > 1e-12)
                                        {
                                                 printf("differnt F values in: (%d of %d), (%d of %d), (%d of %d). The values are (old, new): %f, %f\n",
                                                                i, NDIM, f, nf_, j, geo::H_N3, dline[j], double(Fl[i][f][j]));
                                                fclose(fp);
                                                return 0;
                                        }
                                }
                        }
                }
                fclose(fp);
        }
        else
                return -1;
        return 1;
}
template<int NDIM, int INX, class PHYS>
int hydro_computer<NDIM, INX, PHYS>::compareU(const hydro::state_type &U, int num, std::string test_type){
        hydro::state_type V;
        double dline[geo::H_NX];
        std::string filename;

        if (num > 0)
                filename = test_type + "_U_test_" + std::to_string(num) + ".data";
        else
                filename = test_type + "_U_test_final.data";
        FILE *fp = fopen(filename.c_str(), "rb");
        if (fp != nullptr)
        {
                for (int f = 0; f < nf_; f++) {
                        fread(&dline, sizeof(double), geo::H_NX, fp);
                        for (int i = 0; i < geo::H_NX; i++)
                        {
                                if (std::abs(U[f][i] - dline[i])/(1e-12+U[f][i]+dline[i]) > 1e-12)
                                {
                                        printf("differnt U values in: (%d of %d), (%d of %d). The values are (old, new): %f, %f\n",
                                                                f, nf_, i, geo::H_NX, dline[i], double(U[f][i]));
                                        fclose(fp);
                                        return 0;
                                }
                        }
                }
                fclose(fp);
        }
        else
                return -1;
        return 1;
}

