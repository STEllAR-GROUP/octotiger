#include "node_registry.hpp"

#include <mutex>

namespace node_registry {

table_type table_;
hpx::lcos::local::mutex mtx_;

node_ptr get(const node_location& loc ) {
	return table_[loc];
}


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

table_type table_;
hpx::lcos::local::mutex mtx_;

void put(node_location::node_id id, const hpx::id_type& component);
hpx::id_type get(node_location::node_id id);
}

HPX_PLAIN_ACTION(load_registry::put,put_action);
HPX_PLAIN_ACTION(load_registry::get,get_action);
HPX_PLAIN_ACTION(load_registry::destroy,destroy_action);

namespace load_registry {

void destroy() {
	static auto localities = hpx::find_all_localities();
	if (hpx::get_locality_id() == 0) {
		for (int i = 1; i < localities.size(); i++) {
			hpx::apply<destroy_action>(localities[i]);
		}
	}
	table_.clear();
}

void put(node_location::node_id id, const hpx::id_type& component) {
	static auto localities = hpx::find_all_localities();
	const int locality = (id % node_location::node_id(localities.size()));
	if (locality != hpx::get_locality_id()) {
		put_action f;
		f(localities[locality], id, component);
	} else {
		std::lock_guard<hpx::lcos::local::mutex> lock(mtx_);
		auto iter = table_.find(id);
		assert(iter == table_.end());
		table_.insert(std::make_pair(id,component));
	}
}

hpx::id_type get(node_location::node_id id) {
	static auto localities = hpx::find_all_localities();
	hpx::id_type rc;
	const int locality = id % node_location::node_id(localities.size());
	if (locality != hpx::get_locality_id()) {
		get_action f;
		rc = f(localities[locality], id);
	} else {
		std::unique_lock<hpx::lcos::local::mutex> lock(mtx_);
		auto i = table_.find(id);
		if (i != table_.end()) {
			rc = i->second;
		} else {
			lock.unlock();
			node_location full_loc;
			full_loc.from_id(id);
			auto f = hpx::new_ < node_server > (hpx::find_here(), full_loc);
			auto component = f.get();
			lock.lock();
			table_.insert(std::make_pair(id, component));
			lock.unlock();
			rc = std::move(component);
			if( full_loc.level() != 0 ) {
				auto this_parent = get(full_loc.get_parent().to_id());
				node_client c(rc);
				assert( this_parent != hpx::invalid_id);
				node_client p(this_parent);
				auto f1 = c.set_parent(this_parent);
				p.notify_parent(full_loc,rc).get();
				f1.get();
			}
		}
	}
	return rc;
}

std::size_t hash::operator()(const node_location::node_id id) const {
	static auto localities = hpx::find_all_localities();
	return id / localities.size();
}

}
