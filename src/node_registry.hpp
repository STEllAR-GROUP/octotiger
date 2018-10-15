#ifndef ___NODE_REGISTRY_H
#define ___NODE_REGISTRY_H

#include "defs.hpp"
#include "node_server.hpp"

#include <unordered_map>

namespace node_registry {

struct hash {
	std::size_t operator()(const node_location& loc) const {
		return loc.hash();
	}
};


using node_ptr = node_server*;
using table_type = std::unordered_map<node_location,node_ptr,hash>;
using iterator_type = std::unordered_map<node_location,node_ptr,hash>::iterator;

void add(const node_location&, node_ptr);

void delete_(const node_location&);

iterator_type begin();

iterator_type end();



}

#endif
