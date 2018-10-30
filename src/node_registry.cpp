#include "node_registry.hpp"
#include "options.hpp"

static const auto& localities = options::all_localities;

namespace node_registry {

static table_type table_;
static hpx::lcos::local::spinlock mtx_;

node_ptr get(const node_location& loc) {
	std::lock_guard<hpx::lcos::local::spinlock> lock(mtx_);
	const auto i = table_.find(loc);
	if (i == table_.end()) {
		printf("Error in node_registry::get %s\n", loc.to_str().c_str());
		abort();
	}
	return i->second;
}

void add(const node_location& loc, node_ptr id) {
	std::lock_guard<hpx::lcos::local::spinlock> lock(mtx_);
	if (table_.find(loc) != table_.end()) {
		printf("Error in node_registry::add %s\n", loc.to_str().c_str());
		abort();
	}
	table_.insert(std::make_pair(loc, id));
}

void delete_(const node_location& loc) {
	std::lock_guard<hpx::lcos::local::spinlock> lock(mtx_);
	if (table_.find(loc) == table_.end()) {
		printf("Error in node_registry::delete_ %s\n", loc.to_str().c_str());
		abort();
	}
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

static table_type table_;
static hpx::lcos::local::spinlock mtx_;
}

HPX_PLAIN_ACTION(load_registry::make_at, make_at_action);
HPX_PLAIN_ACTION(load_registry::put, put_action);
HPX_PLAIN_ACTION(load_registry::get, get_action);
HPX_PLAIN_ACTION(load_registry::destroy, destroy_action);

namespace load_registry {

void destroy() {
	if (hpx::get_locality_id() == 0) {
		for (int i = 1; i < localities.size(); i++) {
			hpx::apply<destroy_action>(localities[i]);
		}
	}
	table_.clear();
}

void put(node_location::node_id id, const hpx::id_type& component) {
	const int locality = (id % node_location::node_id(localities.size()));
	if (locality != hpx::get_locality_id()) {
		put_action f;
		f(localities[locality], id, component);
	} else {
		std::lock_guard<hpx::lcos::local::spinlock> lock(mtx_);
		auto iter = table_.find(id);
		assert(iter == table_.end());
		table_.insert(std::make_pair(id, hpx::make_ready_future(component)));
	}
}

hpx::id_type get(node_location::node_id id) {
	hpx::id_type rc;
	const int locality = id % node_location::node_id(localities.size());
	if (locality != hpx::get_locality_id()) {
		get_action f;
		rc = f(localities[locality], id);
	} else {
		std::unique_lock<hpx::lcos::local::spinlock> lock(mtx_);
		auto i = table_.find(id);
		if (i != table_.end()) {
			rc = i->second.get();
		} else {
			node_location full_loc;
			full_loc.from_id(id);
			hpx::shared_future<hpx::id_type> f(hpx::new_<node_server>(hpx::find_here(), full_loc));

			table_.insert(std::make_pair(id, f));
			lock.unlock();
			rc = f.get();
			if (full_loc.level() != 0) {
				auto this_parent = get(full_loc.get_parent().to_id());
				node_client c(rc);
				assert(this_parent != hpx::invalid_id);
				node_client p(this_parent);
				p.notify_parent(full_loc, rc).get();
			}
		}
	}
	return rc;
}

hpx::id_type make_at(node_location::node_id id, hpx::id_type locality) {
	hpx::id_type rc;
	if (locality != hpx::find_here()) {
		get_action f;
		rc = f(locality, id);
	} else {
		std::unique_lock<hpx::lcos::local::spinlock> lock(mtx_);

		node_location full_loc;
		full_loc.from_id(id);
		hpx::shared_future<hpx::id_type> f(hpx::new_<node_server>(hpx::find_here(), full_loc));

		table_.insert(std::make_pair(id, f));
		lock.unlock();
		rc = f.get();
		if (full_loc.level() != 0) {
			auto this_parent = get(full_loc.get_parent().to_id());
			node_client c(rc);
			assert(this_parent != hpx::invalid_id);
			node_client p(this_parent);
			p.notify_parent(full_loc, rc).get();
		}
	}
	return rc;
}

std::size_t hash::operator()(const node_location::node_id id) const {
	return id / localities.size();
}

}
