//  Copyright (c) 2019
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/node_registry.hpp"
#include "octotiger/options.hpp"

#include <cstdio>
#include <mutex>
#include <vector>

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

const size_t size() {
  return table_.size();
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
