/*
 * binary_params.cpp
 *
 *  Created on: Mar 8, 2019
 *      Author: dmarce1
 */

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <vector>
#include <cstring>
#include <silo.h>
#include <cmath>
#include <unordered_map>

template<class T = double>
using array_type = std::vector<T>;

using map_type = std::unordered_map<std::string, array_type<>>;

map_type var_map_;
array_type<> x_, y_, z_, dx_;
array_type<bool> in_plane_;

static DBfile* db_;

double sum_all(const std::string var_name) {
	const auto var = var_map_[var_name];
	double sum = 0.0;
	for (int i = 0; i < var.size(); i++) {
		const auto V = dx_[i] * dx_[i] * dx_[i];
		sum += var[i] * V;
	}
	return sum;
}

double max_all(const std::string var_name, bool plane_only = false) {
	const auto var = var_map_[var_name];
	double max = -std::numeric_limits<double>::max();
	for (int i = 0; i < var.size(); i++) {
		if (!plane_only || in_plane[i]) {
			max = std::max(max, var[i]);
		}
	}
	return max;
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
	db_ = DBOpenReal(filename.c_str(), DB_HDF5, DB_READ);

	if (db_ == nullptr) {
		printf("Unable to open %s\n", filename.c_str());
		return -1;
	}

	int n_species;

	DBReadVar(db_, "n_species", (void*) &n_species);

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

	printf("Max rho = %e\n", max_all("rho"));

	printf("Mass sum = %e\n", sum_all("rho_1") / 2e33);
	printf("Mass sum = %e\n", sum_all("rho_2") / 2e33);
	printf("Mass sum = %e\n", sum_all("rho_3") / 2e33);
	printf("Mass sum = %e\n", sum_all("rho_4") / 2e33);
	printf("Mass sum = %e\n", sum_all("rho_5") / 2e33);

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
