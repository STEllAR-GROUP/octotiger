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
	table_[loc] = id;
}

void delete_(const node_location& loc) {
	std::lock_guard<hpx::lcos::local::spinlock> lock(mtx_);
	if (table_.find(loc) != table_.end()) {
		table_.erase(loc);
	}
}

iterator_type begin() {
	return table_.begin();
}

iterator_type end() {
	return table_.end();
}

void clear_();

}

HPX_PLAIN_ACTION(node_registry::clear_, node_registry_clear_action);

namespace node_registry {
void clear_() {
	std::vector<hpx::future<void>> futs;
	if (hpx::get_locality_id() == 0) {
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async<node_registry_clear_action>(localities[i]));
		}
		table_.clear();
		hpx::wait_all(std::move(futs));
	} else {
		table_.clear();
	}
}

void clear() {
	clear_();
}

}

namespace load_registry {

static table_type table_;
static hpx::lcos::local::spinlock mtx_;
}
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
		table_.insert(std::make_pair(id, std::make_shared<hpx::shared_future<hpx::id_type>>(hpx::make_ready_future(component))));
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
			auto tmp = i->second;
			lock.unlock();
			rc = tmp->get();
		} else {
			node_location full_loc;
			full_loc.from_id(id);
			// Done this way to avoid swapping threads while lock is held
			auto prms = std::make_shared<hpx::lcos::local::promise<hpx::id_type>>();
			auto entry = std::make_pair(id, std::make_shared<hpx::shared_future<hpx::id_type>>(prms->get_future()));
			auto f = entry.second;
			table_.insert(std::move(entry));
			lock.unlock();
			hpx::apply([prms,full_loc]() {
				auto tmp = hpx::new_<node_server>(hpx::find_here(), full_loc);
				prms->set_value(tmp.get());
			});
			rc = f->get();
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
std::size_t hash::operator()(const node_location::node_id id) const {
	return id;
}

}
