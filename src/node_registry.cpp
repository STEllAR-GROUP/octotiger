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
