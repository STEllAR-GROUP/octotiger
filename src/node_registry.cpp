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

namespace load_registry {

static auto localities = hpx::find_all_localities();

table_type table_;
hpx::lcos::local::mutex mtx_;

void put(node_location::node_id id, const hpx::id_type& component);
hpx::id_type get(node_location::node_id id);
}
HPX_PLAIN_ACTION(load_registry::put,put_action);
HPX_PLAIN_ACTION(load_registry::get,get_action);
namespace load_registry {

void put(node_location::node_id id, const hpx::id_type& component) {
	const int locality = id % localities.size();
	if (locality != hpx::get_locality_id()) {
		put_action f;
		f(localities[locality], id, component);
	} else {
		std::lock_guard<hpx::lcos::local::mutex> lock(mtx_);
		table_[id] = component;
	}
}

hpx::id_type get(node_location::node_id id) {
	hpx::id_type rc;
	const int locality = id % localities.size();
	if (locality != hpx::get_locality_id()) {
		get_action f;
		rc = f(localities[locality], id);
	} else {
		std::lock_guard<hpx::lcos::local::mutex> lock(mtx_);
		auto i = table_.find(id);
		if (i != table_.end()) {
			rc = i->second;
		} else {
			auto f = hpx::new_ < node_server > (hpx::find_here());
			auto component = f.get();
			table_.insert(std::make_pair(id, component));
			rc = std::move(component);

		}
	}
	return rc;
}

std::size_t hash::operator()(const node_location::node_id id) const {
	return id / localities.size();
}

}
