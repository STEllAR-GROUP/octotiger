/*
 * compare.cpp
 *
 *  Created on: Jan 7, 2019
 *      Author: dmarce1
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <cstring>
#include <string>
#include <vector>

#include <silo.h>

#define SILO_DRIVER DB_HDF5

std::string strip_nonnumeric(std::string&& s) {
	s.erase(std::remove_if(s.begin(), s.end(), [](char c) {return c < '0' || c > '9';}), s.end());
	return std::move(s);
}

struct silo_file {
	DBfile* handle;
	std::set<std::string> var_names;
	std::set<std::string> mesh_names;

	silo_file(const std::string name) {
		handle = DBOpenReal(name.c_str(), SILO_DRIVER, DB_READ);

		//      std::cout << "Variables in " << name << " are:\n";
		auto toc = DBGetToc(handle);
		for (int i = 0; i < toc->nmultivar; i++) {
			auto name = std::string(toc->multivar_names[i]);
			var_names.insert(name);
			// std::cout << "   " << name << "\n";
		}

//         std::cout << "Meshes in " << name << " are:\n";
		auto mesh = DBGetMultimesh(handle, "quadmesh");
		for (int i = 0; i < mesh->nblocks; i++) {
			auto name = strip_nonnumeric(std::string(mesh->meshnames[i]));
			mesh_names.insert(name);
			//       std::cout << "   " << name << "\n";
		}

		DBFreeMultimesh(mesh);
	}

	void compare(const std::string name) {
		auto other = DBOpenReal(name.c_str(), SILO_DRIVER, DB_READ);
		std::string copy = std::string("cp ") + name.c_str() + std::string(" diff.silo\n");
		std::cout << copy;
		std::system(copy.c_str());
		auto diff = DBOpenReal("diff.silo", SILO_DRIVER, DB_APPEND);

		for (const auto& vn : var_names) {
			double vtot = 0.0;
			double l1 = 0.0;
			double l2 = 0.0;
			double linf = 0.0;
			for (auto const& mn : mesh_names) {
				std::string mesh_loc = "/" + mn + "/quadmesh";
				std::string var_loc = "/" + mn + "/" + vn;

				auto quadmesh = DBGetQuadmesh(handle, mesh_loc.c_str());

				double* X = (double*) quadmesh->coords[0];
				const auto dx = X[1] - X[0];
				const auto dv = dx * dx * dx;

				auto this_var = DBGetQuadvar(handle, var_loc.c_str());
				auto other_var = DBGetQuadvar(other, var_loc.c_str());

				const auto n = this_var->dims[0] * this_var->dims[1] * this_var->dims[2];
				for (int i = 0; i < n; i++) {
					double& this_val = ((double*) this_var->vals[0])[i];
					double other_val = ((double*) other_var->vals[0])[i];
					auto d = std::abs(this_val - other_val);
					l1 += d * dv;
					l2 += d * d * dv;
					linf = std::max(linf, d);
					vtot += dv;
					this_val -= other_val;
				}
				auto optlist = DBMakeOptlist(100);
				int one = 1;
				DBAddOption(optlist, DBOPT_HIDE_FROM_GUI, &one);
				std::string diff_loc = var_loc + std::string("_diff");
				DBPutQuadvar1(diff, diff_loc.c_str(), this_var->meshname, (double*) this_var->vals[0], this_var->dims,
						this_var->ndims, nullptr, 0, this_var->datatype, this_var->centering, optlist);

				DBFreeOptlist(optlist);

				DBFreeQuadvar(this_var);
				DBFreeQuadvar(other_var);
				DBFreeQuadmesh(quadmesh);

			}
			auto mv = DBGetMultivar(diff, vn.c_str());
			const auto name = vn + std::string("_diff");
			std::vector<char*> names;
			for (int i = 0; i < mv->nvars; i++) {
				const auto name = std::string(mv->varnames[i]) + std::string("_diff");
				char* ptr = new char[name.size() + 1];
				std::strcpy(ptr, name.c_str());
				names.push_back(ptr);
			}
			DBPutMultivar(diff, name.c_str(), mv->nvars, names.data(), mv->vartypes, nullptr);
			for (auto & n : names) {
				delete[] n;
			}
			DBFreeMultivar(mv);
			l1 = l1 / vtot;
			l2 = std::sqrt(l2 / vtot);
			printf("variable: %s\n", vn.c_str());
			printf("     L1 : %e\n", l1);
			printf("     L2 : %e\n", l2);
			printf("     Linf : %e\n", linf);
		}

		DBClose(other);
		DBClose(diff);
	}

	~silo_file() {
		DBClose(handle);
	}
};

int main(int argc, char* argv[]) {
	if (argc != 3) {
		std::cout << "Usage -> compare <file1> <file2>\n";
		return -1;
	} else {
		silo_file file1(argv[1]);
		file1.compare(argv[2]);
	}
	return 0;
}
