#include "node_registry.hpp"

#include <mutex>

namespace node_registry {

table_type table_;
hpx::lcos::local::mutex mtx_;

void add(const node_location& loc, node_ptr id) {
	std::lock_guard<hpx::lcos::local::mutex> lock(mtx_);
	table_.insert(std::make_pair(loc, id));
}

void delete_(const node_location& loc) {
	std::lock_guard<hpx::lcos::local::mutex> lock(mtx_);
	table_.erase(loc);
}

iterator_type begin() {
	return table_.begin();
}

iterator_type end() {
	return table_.end();
}

}

#include <silo.h>

void output_all(std::string fname, int cycle);

HPX_PLAIN_ACTION(output_all, output_all_action);

void output_all(std::string fname, int cycle) {
	std::vector<hpx::future<void>> futs;
	std::string this_fname = fname + std::to_string(cycle) + std::to_string(hpx::get_locality_id()) + std::string(".silo");
	DBfile *db = DBCreateReal(this_fname.c_str(), DB_CLOBBER, DB_LOCAL, "Octo-tiger", DB_PDB);
	for (auto i = node_registry::begin(); i != node_registry::end(); i++) {
		futs.push_back(hpx::async([](DBfile* db, node_location loc, node_server* this_ptr) {
			std::array<std::array<real, INX+1>, NDIM> X;
			const real dx = TWO / real(1 << loc.level()) / real(INX);
			for( int d = 0; d < NDIM; d++) {
				const real x0 = loc.x_location(d);
				for( int i = 0; i <= INX; i++) {
					X[d][i] = x0 + real(i) * dx;
				}
			}
			const std::string mesh_name = std::string( "mesh_") + loc.to_str();
			const char* mname = mesh_name.c_str();
			const char* coord_names[] = {"x", "y", "z"};
			const real* coords[] = {X[0].data(),X[1].data(),X[2].data()};
			constexpr int dims[] = {INX,INX,INX};
			constexpr int data_type = DB_DOUBLE;
			constexpr int ndim = NDIM;
			constexpr int coord_type = DB_COLLINEAR;

			DBPutQuadmesh(db, mname, coord_names, coords, dims, ndim, data_type, coord_type, NULL);

			return;
		}, db, i->first, i->second));
	}
	for (auto& f : futs) {
		f.get();
	}
	DBClose(db);
}
