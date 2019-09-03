//  Copyright (c) 2019
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <vector>
#include <cstring>
#include <silo.h>
#include <cmath>
#include <unordered_map>

#define NOSTAR 0
#define COMMON_ENVELOPE 1
#define STAR1 2
#define STAR2 3


template<class T = double>
using array_type = std::vector<T>;

using map_type = std::unordered_map<std::string, array_type<>>;

map_type var_map_;
array_type<> x_, y_, z_, dx_;
array_type<bool> in_plane_;
array_type<bool> in_loc_;
array_type<double> loc_x_;
double rho_mid_;
array_type<int> in_star_;

static DBfile* db_;

std::array<double, 3> find_com() {
	std::array<double, 3> com = { { 0, 0, 0 } };
	double mtot = 0.0;
	const auto& rho = var_map_["rho"];
	for (int i = 0; i < rho.size(); i++) {
		const double V = std::pow(dx_[i], 3);
		const double M = V * rho[i];
		const double x = x_[i];
		const double y = y_[i];
		const double z = z_[i];
		mtot += M;
		com[0] += x * M;
		com[1] += y * M;
		com[2] += z * M;
	}
	com[0] /= mtot;
	com[1] /= mtot;
	com[2] /= mtot;
	return com;
}

double find_omega(std::array<double, 3> com) {
	const auto& rho = var_map_["rho"];
	const auto& sx = var_map_["sx"];
	const auto& sy = var_map_["sy"];
	double I = 0.0;
	double l = 0.0;
	for (int i = 0; i < rho.size(); i++) {
		if (rho[i] > 1.) {
			const double V = std::pow(dx_[i], 3);
			const double x = x_[i] - com[0];
			const double y = y_[i] - com[1];
			const double r2 = x * x + y * y;
			const double this_omega = (x * sy[i] - y * sx[i]) / r2 / rho[i];
			I += rho[i] * V * r2;
			l += (x * sy[i] - y * sx[i]) * V;
		}

	}
	printf("%e %e\n", l, I);
	return l / I;
}

void tag_loc(std::array<double, 2> loc, std::array<double, 3> a) {
	const auto& rho = var_map_["rho"];
	double p[3], n[3];
	n[0] = loc[0];
	n[1] = loc[1];
	n[2] = 0;
	for (int i = 0; i < rho.size(); i++) {
		p[0] = x_[i];
		p[1] = y_[i];
		p[2] = z_[i];
		double amp[3];
		double ampdotn;
		for (int d = 0; d < 3; d++) {
			amp[d] = a[d] - p[d];
		}
		ampdotn = 0.0;
		for (int d = 0; d < 3; d++) {
			ampdotn += amp[d] * n[d];
		}
		double dist = 0.0;
		double loc_x = 0.0;
		for (int d = 0; d < 3; d++) {
			dist += std::pow(amp[d] - ampdotn * n[d], 2);
			loc_x += n[d] * p[d];
		}
		dist = std::sqrt(dist);
		loc_x_.push_back(loc_x);
		if (dist < dx_[i]) {
			in_loc_.push_back(true);
		} else {
			in_loc_.push_back(false);
		}
	}
}

double find_eigenvector(std::array<double, 2>& e) {
	std::array<std::array<double, 2>, 2> q = { { { 0, 0 }, { 0, 0 } } };

	const auto& rho = var_map_["rho"];

	for (int i = 0; i < rho.size(); i++) {
		const double V = std::pow(dx_[i], 3);
		const double M = V * rho[i];
		const double x = x_[i];
		const double y = y_[i];
		q[0][0] += x * x * M;
		q[0][1] += x * y * M;
		q[1][0] += y * x * M;
		q[1][1] += y * y * M;
	}

	std::array<double, 2> b0, b1;
	double A, bdif;
	int iter = 0;
	b0[0] = 1.0;
	b0[1] = 1.0;
	do {
		iter++;
		b1[0] = b1[1] = 0.0;
		for (int i = 0; i < 2; i++) {
			for (int m = 0; m < 2; m++) {
				b1[i] += q[i][m] * b0[m];
			}
		}
		A = sqrt(b1[0] * b1[0] + b1[1] * b1[1]);
		bdif = 0.0;
		for (int i = 0; i < 2; i++) {
			b1[i] = b1[i] / A;
			bdif += pow(b0[i] - b1[i], 2);
		}
		for (int i = 0; i < 2; i++) {
			b0[i] = b1[i];
		}
	} while (fabs(bdif) > 1.0e-14);
	double lambda = 0.0;
	double e2 = 0.0;
	e = b0;
	for (int m = 0; m < 2; m++) {
		lambda += e[m] * (q[m][0] * e[0] + q[m][1] * e[1]);
		e2 += e[m] * e[m];
	}
	return lambda / e2;
}

double sum_all(const std::string var_name) {
	const auto var = var_map_[var_name];
	double sum = 0.0;
	for (int i = 0; i < var.size(); i++) {
		const auto V = dx_[i] * dx_[i] * dx_[i];
		sum += var[i] * V;
	}
	return sum;
}

std::pair<double, int> max_all(const std::string var_name, bool plane_only = false) {
	const auto var = var_map_[var_name];
	double max = -std::numeric_limits<double>::max();
	int max_i = 0;
	for (int i = 0; i < var.size(); i++) {
		if (!plane_only || in_plane_[i]) {
			if (max < var[i]) {
				max = var[i];
				max_i = i;
			}
		}
	}
	return std::pair<double, int>(max, max_i);
}

std::pair<double, int> min_all(const std::string var_name, bool plane_only = false) {
	const auto var = var_map_[var_name];
	double min = +std::numeric_limits<double>::max();
	int min_i = 0;
	for (int i = 0; i < var.size(); i++) {
		if (!plane_only || in_plane_[i]) {
			if (min > var[i]) {
				min = var[i];
				min_i = i;
			}
		}
	}
	return std::pair<double, int>(min, min_i);
}

std::string strip_nonnumeric(std::string&& s) {
	s.erase(std::remove_if(s.begin(), s.end(), [](char c) {return c < '0' || c > '9';}), s.end());
	return std::move(s);
}

int main(int argc, char* argv[]) {

	if (argc != 2) {
		printf("Usage: binary_params <silo_file>\n");
		return -1;
	}

	/* Open SILO */

	printf("Opening SILO\n");
	std::string filename = argv[1];
	db_ = DBOpenReal(filename.c_str(), DB_HDF5, DB_READ );

	if (db_ == nullptr) {
		printf("Unable to open %s\n", filename.c_str());
		return -1;
	}

	long long n_species;
	long long version;
	double omega;
	double code_to_s;
	double cgs_time;
	DBReadVar(db_, "cgs_time", (void*) &cgs_time);
	DBReadVar(db_, "version", (void*) &version);
	DBReadVar(db_, "n_species", (void*) &n_species);
	DBReadVar(db_, "code_to_s", (void*) &code_to_s);
	DBReadVar(db_, "omega", (void*) &omega);
	printf("Omega = %e\n", omega);
	printf("SILO version: %i\n", static_cast<int>(version));
	printf("N species   : %i\n", static_cast<int>(n_species));

	printf("Reading table of contents\n");
	DBmultimesh* mmesh = DBGetMultimesh(db_, "quadmesh");
	std::vector < std::string > dir_names;
	for (int i = 0; i < mmesh->nblocks; i++) {
		const std::string dir = strip_nonnumeric(mmesh->meshnames[i]);
		dir_names.push_back(dir);
	}
	DBFreeMultimesh(mmesh);

	for (int i = 0; i < dir_names.size(); i++) {
		const std::string dir = dir_names[i];
		if (dir == "Decomposition") {
			continue;
		}
//		printf("%i of %i - %s\n", i, dir_names.size(), dir.c_str());
		DBSetDir(db_, dir.c_str());

		int sz;

		const DBtoc* this_toc = DBGetToc(db_);
		for (int j = 0; j < this_toc->nvar; j++) {
			const std::string qvar = this_toc->qvar_names[j];
			DBquadvar* var = DBGetQuadvar(db_, qvar.c_str());
			sz = var->nels;
			auto& data = var_map_[qvar];
			data.resize(data.size() + sz);
			double* dest = &(data[data.size() - sz]);
			if (version == 100 && (qvar == "sx" || qvar == "sy" || qvar == "sz")) {
				for (int k = 0; k < sz; k++) {
					(((double**) var->vals)[0])[k] *= code_to_s;
				}
			}
			std::memcpy(dest, ((double**) var->vals)[0], sizeof(double) * sz);
			DBFreeQuadvar(var);
		}

		DBquadmesh* mesh = DBGetQuadmesh(db_, "quadmesh");
		const double* xc = static_cast<double*>(mesh->coords[0]);
		const double* yc = static_cast<double*>(mesh->coords[1]);
		const double* zc = static_cast<double*>(mesh->coords[2]);
		const double dx = xc[1] - xc[0];
		for (int l = 0; l < mesh->dims[2] - 1; l++) {
			for (int k = 0; k < mesh->dims[1] - 1; k++) {
				for (int j = 0; j < mesh->dims[0] - 1; j++) {
					x_.push_back(xc[j] + 0.5 * dx);
					y_.push_back(yc[k] + 0.5 * dx);
					z_.push_back(zc[l] + 0.5 * dx);
					dx_.push_back(dx);
					const bool in_plane = std::abs(zc[l] - 0.5 * dx) < dx;
					in_plane_.push_back(in_plane);
				}
			}
		}

		DBFreeQuadmesh(mesh);

		DBSetDir(db_, "..");

	}
	auto total_size = var_map_["rho_1"].size();
	auto& rho = var_map_["rho"];
	rho.resize(total_size, 0.0);

	for (int i = 0; i < n_species; i++) {
		std::string name = std::string("rho_") + char('1' + i);
		auto& this_rho = var_map_[name];
		for (int j = 0; j < total_size; j++) {
			rho[j] += this_rho[j];
		}
	}

	printf("Mass sum = %e\n\n\n\n", sum_all("rho"));

	auto rho_max = max_all("rho").first;
	auto rho_min = min_all("rho").first;
	rho_mid_ = std::sqrt(rho_max * rho_min);

	printf( "rho_max = %e\n", rho_max);
	printf( "rho_mid = %e\n", rho_mid_);
	printf( "rho_min = %e\n", rho_min);

	auto c1i = max_all("rho_1").second;
	auto c2i = max_all("rho_3").second;

	printf("c1 at %e %e %e\n", x_[c1i], y_[c1i], z_[c1i]);
	printf("c2 at %e %e %e\n", x_[c2i], y_[c2i], z_[c2i]);

	printf("Mass sum = %e\n", sum_all("rho_1") / 2e33);
	printf("Mass sum = %e\n", sum_all("rho_2") / 2e33);
	printf("Mass sum = %e\n", sum_all("rho_3") / 2e33);
	printf("Mass sum = %e\n", sum_all("rho_4") / 2e33);
	printf("Mass sum = %e\n", sum_all("rho_5") / 2e33);

	auto com = find_com();

	printf("Center of Mass = %e %e %e\n", com[0], com[1], com[2]);

	std::array<double, 2> loc;
	double q = find_eigenvector(loc);
	printf("LOC = %e  %e\n", loc[0], loc[1]);
	printf("q = %e\n", q);

	tag_loc(loc, com);
	{
		FILE* fp = fopen("loc.txt", "wt");
		auto& rho = var_map_["rho"];
		for (int i = 0; i < rho.size(); i++) {
			if (in_loc_[i]) {
				fprintf(fp, "%e %e\n", loc_x_[i], rho[i]);
			}
		}
		fclose(fp);
	}

	omega = find_omega(com);
	double period = 2.0 * M_PI / omega / 60. / 60. / 24.;
	printf("Omega = %e\n", omega);
	printf("Period = %e days\n", period);


	double l1 = -std::numeric_limits<double>::max();
	double l2 = -std::numeric_limits<double>::max();
	double l3 = -std::numeric_limits<double>::max();
	double l1_loc, l2_loc, l3_loc;

	double c1_loc = x_[c1i] * loc[0] + y_[c1i] * loc[1];
	double c2_loc = x_[c2i] * loc[0] + y_[c2i] * loc[1];

	{
		const auto& phi = var_map_["phi"];
		for( int i = 0; i < phi.size(); i++ ) {
			double this_loc = x_[i] * loc[0] + y_[i] * loc[1];
			if( in_loc_[i] ) {
				double phi_eff = -omega * (x_[i] * x_[i] + y_[i] * y_[i]) + phi[i];
				double* lptr;
				double* loc_ptr;
				if( (c1_loc < c2_loc && this_loc < c1_loc) || (c1_loc > c2_loc && this_loc > c1_loc) ) {
					lptr = &l3;
					loc_ptr = &l3_loc;
				}  else if( (this_loc - c1_loc) * (this_loc - c2_loc) <= 0.0 ) {
					lptr = &l1;
					loc_ptr = &l1_loc;
				} else {
					lptr = &l2;
					loc_ptr = &l2_loc;
				}
				if( phi_eff > *lptr ) {
					*lptr = phi_eff;
					*loc_ptr = this_loc;
				}
			}
		}
	}

	printf( "L1 = %e @ %e\n", l1, l1_loc );
	printf( "L2 = %e @ %e\n", l2, l2_loc );
	printf( "L3 = %e @ %e\n", l3, l3_loc );

	const auto& sx = var_map_["sx"];
	const auto& sy = var_map_["sy"];
	const auto& tau = var_map_["tau"];

	double M[4] = {0,0,0,0};
	double spin[4] = {0,0,0,0};
	double lorb = 0.0;
	double heat = 0.0;
	FILE* fp = fopen( "roche.txt", "wt");
	{
		const auto& phi = var_map_["phi"];
		for( int i = 0; i < phi.size(); i++ ) {
			double R2 = x_[i] * x_[i] + y_[i] * y_[i];
			double phi_eff = phi[i] - 0.5 * R2 * omega;
			double g1, g2;
			std::array<double, 3> d1, d2;
			d1[0] = x_[c1i] - x_[i];
			d1[1] = y_[c1i] - y_[i];
			d1[2] = z_[c1i] - z_[i];
			d2[0] = x_[c2i] - x_[i];
			d2[1] = y_[c2i] - y_[i];
			d2[2] = z_[c2i] - z_[i];
			g1 = g2 = 0.0;
			g1 += d1[0] * x_[c1i];
			g1 += d1[1] * y_[c1i];
			g1 += d1[2] * z_[c1i];
			g2 += d2[0] * x_[c2i];
			g2 += d2[1] * y_[c2i];
			g2 += d2[2] * z_[c2i];
			int v;
			std::array<double,3> pivot;
	//		if( phi_eff < l1) {
				if( g1 >= 0 && g2 >=0 ) {
					if( g1 > g2) {
						v = STAR1;
					} else {
						v = STAR2;
					}
				} else if( g1 >= 0 ) {
					v = STAR1;
				} else if( g2 >= 0 ) {
					v = STAR2;
				} else {
					v = NOSTAR;
				}
		/*	} else  {
				if( g1 >= 0 || g2 >=0 ) {
					v = COMMON_ENVELOPE;
				} else {
					v = NOSTAR;
				}
			}*/
			if( v == STAR1 ) {
				pivot[0] = x_[c1i];
				pivot[1] = y_[c1i];
				pivot[2] = z_[c1i];
			} else if( v == STAR2 ) {
				pivot[0] = x_[c2i];
				pivot[1] = y_[c2i];
				pivot[2] = z_[c2i];
			} else {
				pivot = com;
			}
			double arm[2];
			arm[0] = x_[i] - pivot[0];
			arm[1] = y_[i] - pivot[1];
			const double vol = dx_[i] * dx_[i] * dx_[i];
			M[v] += rho[i] * vol;
			spin[v] += (sx[i] * arm[1] - sy[i] * arm[0])*vol;
			arm[0] = x_[i] - com[0];
			arm[1] = y_[i] - com[1];
			if( v == STAR1 || v == STAR2 ) {
				lorb += (sx[i] * arm[1] - sy[i] * arm[0])*vol;
			}
			heat += std::pow(tau[i],5./3.) * vol;
			in_star_.push_back(v);
			fprintf( fp, "%e %e %e %i\n", x_[i], y_[i], z_[i], v);
		}

	}
	lorb -= spin[3] + spin[2];
	fclose(fp);
#define sqr(a) ((a)*(a))

	for( int f = 0; f < 4; f++) {
		M[f] /= 1.99e+33;
	}
	double sep = std::sqrt(sqr(x_[c1i] - x_[c2i]) + sqr(y_[c1i] - y_[c2i]) + sqr(z_[c1i] - z_[c2i]));
	fp = fopen( "binary.txt", "at");
	fprintf( fp, "%e %e %e %e %e %e %e %e %e\n", cgs_time, M[0], M[2], M[3], sep, spin[2], spin[3], lorb, heat );
	fclose(fp);
	/* Close SILO */

	DBClose(db_);

}

/*
 using space_vector = std::array<double,3>;
 using std::sqrt;

 template<class T>
 inline T sqr(T s) {
 return s * s;
 }

 std::string strip_nonnumeric(std::string&& s) {
 s.erase(std::remove_if(s.begin(), s.end(), [](char c) {return c < '0' || c > '9';}), s.end());
 return std::move(s);
 }

 double find_eigenvector(const std::array<std::array<double, 3>, 3>& q, std::array<double, 3>& e) {
 std::array<double, 3> b0, b1;
 double A, bdif;
 int iter = 0;
 b0[2] = 0.0;
 b0[0] = 1.0;
 b0[1] = 1.0;
 do {
 iter++;
 b1[0] = b1[1] = b1[2] = 0.0;
 for (int i = 0; i < 3; i++) {
 for (int m = 0; m < 3; m++) {
 b1[i] += q[i][m] * b0[m];
 }
 }
 A = sqrt(sqr(b1[0]) + sqr(b1[1]) + sqr(b1[2]));
 bdif = 0.0;
 for (int i = 0; i < 3; i++) {
 b1[i] = b1[i] / A;
 bdif += pow(b0[i] - b1[i], 2);
 }
 for (int i = 0; i < 3; i++) {
 b0[i] = b1[i];
 }
 } while (fabs(bdif) > 1.0e-14);
 double lambda = 0.0;
 double e2 = 0.0;
 e = b0;
 for (int m = 0; m < 3; m++) {
 lambda += e[m] * (q[m][0]*e[0]+q[m][1]*e[1]+q[m][2]*e[2]);
 e2 += e[m] * e[m];
 }
 return lambda / e2;
 }


 std::array<double, 3> center_of_mass(const std::vector<cell_t>& cells) {
 std::array<double, 3> com;
 double mass = 0.0;
 for (int d = 0; d < 3; d++) {
 com[d] = 0.0;
 }
 for (const auto& c : cells) {
 const auto this_vol = c.dx * c.dx * c.dx;
 double rho = 0.0;
 for (int i = 0; i < cell_t::n_species; i++) {
 rho += c.rho[i];
 }
 const auto this_mass = this_vol * rho;
 mass += this_mass;
 for (int d = 0; d < 3; d++) {
 com[d] += this_mass * c.x[d];
 }
 }
 for (int d = 0; d < 3; d++) {
 com[d] /= mass;
 }
 return com;
 }

 double total_mass(std::vector<cell_t>& cells) {
 double mass = 0.0;
 for (auto& c : cells) {
 const auto this_vol = c.dx * c.dx * c.dx;
 double rho = 0.0;
 for (int i = 0; i < cell_t::n_species; i++) {
 rho += c.rho[i];
 }
 c.rho_tot = rho;
 const auto this_mass = this_vol * rho;
 mass += this_mass;
 }
 return mass;
 }

 std::array<std::array<double, 3>, 3> quadrupole_moment(const std::vector<cell_t>& cells, const std::array<double, 3>& com) {
 std::array<std::array<double, 3>, 3> q = { { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } } };
 for (const auto& c : cells) {
 for (int n = 0; n < 3; n++) {
 const auto this_vol = c.dx * c.dx * c.dx;
 double rho = 0.0;
 for (int i = 0; i < cell_t::n_species; i++) {
 rho += c.rho[i];
 }
 const auto this_mass = this_vol * rho;
 double x = c.x[0] - com[0];
 double y = c.x[1] - com[1];
 double z = c.x[2] - com[2];
 double r2 = x * x + y * y;
 q[0][0] += 3 * this_mass * x * x;
 q[0][1] += 3 * this_mass * x * y;
 //			q[0][2] += 3 * this_mass * x * z;
 q[1][0] += 3 * this_mass * y * x;
 q[1][1] += 3 * this_mass * y * y;
 //			q[1][2] += 3 * this_mass * y * z;
 //			q[2][0] += 3 * this_mass * z * x;
 //			q[2][1] += 3 * this_mass * z * y;
 //			q[2][2] += 3 * this_mass * z * z;
 q[0][0] -= r2 * this_mass;
 q[1][1] -= r2 * this_mass;
 //			q[2][2] -= r2 * this_mass;
 }
 }
 return q;
 }

 int main(int argc, char* argv[]) {

 std::set<std::string> mesh_names;
 std::set<std::string> var_names;
 std::vector<cell_t> cells;
 var_names.insert("phi");
 var_names.insert("sx");
 var_names.insert("sy");
 var_names.insert("sz");

 if (argc != 2) {
 printf("Usage: binary_params <silo_file>\n");
 return -1;
 }

 std::string filename = argv[1];
 auto handle = DBOpenReal(filename.c_str(), DB_HDF5, DB_READ);

 if (handle == nullptr) {
 printf("Unable to open %s\n", filename.c_str());
 return -1;
 }

 DBReadVar(handle, "n_species", &cell_t::n_species);
 for (int i = 0; i < cell_t::n_species; i++) {
 var_names.insert("rho_" + std::to_string(i + 1));
 }
 auto mesh = DBGetMultimesh(handle, "quadmesh");
 for (int i = 0; i < mesh->nblocks; i++) {
 auto name = strip_nonnumeric(std::string(mesh->meshnames[i]));
 mesh_names.insert(name);
 }

 bool first_call = true;
 for (const auto& vn : var_names) {
 printf("Reading %s\n", vn.c_str());
 int p = 0;
 for (auto const& mn : mesh_names) {
 std::string mesh_loc = "/" + mn + "/quadmesh";
 std::string var_loc = "/" + mn + "/" + vn;

 auto quadmesh = DBGetQuadmesh(handle, mesh_loc.c_str());

 double* X = (double*) quadmesh->coords[0];
 const auto dx = X[1] - X[0];
 const auto dv = dx * dx * dx;

 //		printf("Reading %s\n", var_loc.c_str());
 auto var = DBGetQuadvar(handle, var_loc.c_str());
 if (var == nullptr) {
 printf("Unable to read %s\n", var_loc.c_str());
 return -1;
 }
 int i = 0;
 for (int l = 0; l < var->dims[2]; l++) {
 for (int k = 0; k < var->dims[1]; k++) {
 for (int j = 0; j < var->dims[0]; j++) {
 if (first_call) {
 cell_t c;
 c.dx = dx;
 c.x[0] = ((double*) quadmesh->coords[0])[j] + 0.5 * dx;
 c.x[1] = ((double*) quadmesh->coords[1])[k] + 0.5 * dx;
 c.x[2] = ((double*) quadmesh->coords[2])[l] + 0.5 * dx;
 for (int d = 0; d < 3; d++) {
 //			printf("%e ", c.x[d]);
 }
 cells.push_back(c);
 }
 auto& c = cells[p];
 const auto& val = ((double*) var->vals[0])[i];
 //		printf("%e\n", val);
 if (vn == std::string("phi")) {
 c.phi = val;
 } else {
 if (vn.size() == 2 && vn[0] == 's') {
 int index = vn[1] - 'x';
 c.s[index] = val;
 } else if (std::strncmp("rho", vn.c_str(), 3) == 0) {
 int index = vn[4] - '1';
 c.rho[index] = val;
 } else {
 printf("Error on line %i\n", __LINE__);
 return -1;
 }
 }
 p++;
 i++;
 }
 }
 }

 DBFreeQuadvar(var);
 DBFreeQuadmesh(quadmesh);
 }
 first_call = false;
 }
 DBClose(handle);

 auto M = total_mass(cells);
 printf("Total Mass: %e\n", M);

 auto com = center_of_mass(cells);
 printf("Center of Mass: %e %e %e\n", com[0], com[1], com[2]);

 auto q = quadrupole_moment(cells, com);
 printf("Quadrupole Moment: %12e %12e\n", q[0][0], q[0][1]);
 printf("                   %12e %12e\n", q[1][0], q[1][1]);

 double lambda;
 std::array<double, 3> loc;
 lambda = find_eigenvector(q, loc);

 printf("Line of Centers:   %12e %12e %12e\n", loc[0], loc[1], loc[2]);

 double rho_max = 0.0;
 space_vector c1, c2;
 for( const auto& c : cells) {
 double dx2_max = 0.0;
 for( int d = 0; d < 3; d++) {
 dx2_max = std::max(dx2_max,sqr(c.x[d] - loc[d]));
 }
 auto rho = c.rho_tot;
 if( dx2_max <= c.dx* c.dx ) {
 if( rho > rho_max) {
 rho_max = rho;
 c1 = c.x;
 }
 }
 }


 double d1 = sqrt(c1[0]*c1[0]+c1[1]*c1[1]+c1[2]*c1[2]);
 double d2 = lambda / d1 / M;

 double a = d1 + d2;
 printf( "%e %e %e\n", d1, d2, a);

 for( int d= 0; d < 3; d++) {
 c2[d] = c1[d] + loc[d]*(d1+d2);
 }
 printf( "First  star at %e %e %e with rho_max = %e\n", c1[0], c1[1], c1[2], rho_max);
 printf( "Second star at %e %e %e\n", c2[0], c2[1], c2[2]);


 return 0;
 }
 */
