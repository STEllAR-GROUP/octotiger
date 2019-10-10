//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef NODE_LOCATION_HPP_
#define NODE_LOCATION_HPP_

#include "octotiger/defs.hpp"
#include "octotiger/geometry.hpp"

#include <hpx/include/components.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#include <array>

class node_client;

using range_type = std::array<std::pair<int, int>, NDIM>;

range_type intersection(const range_type& r1, const range_type& r2);

class node_location {
private:
	std::array<integer, NDIM> xloc;
	integer lev;
public:
	using node_id = std::uint64_t;
	node_id to_id() const;
	void from_id(const node_id&);
	node_location();
	node_location(const node_location& other);
	node_location(node_location::node_id id);
	node_location& operator=(const node_location& other);
	integer level() const;
	node_location get_child(integer x, integer y, integer z) const;
	node_location get_child(integer c) const;
	node_location get_parent() const;
	node_location get_sibling(integer face) const;
	geo::side get_child_side(const geo::dimension&) const;
	geo::octant get_child_index() const;
	integer operator[](integer i) const {
		return xloc[i];
	}
	std::size_t hash() const;
	bool operator==(const node_location& other) const;
	bool operator!=(const node_location& other) const;
	bool operator<(const node_location& other) const;
	bool operator >=(const node_location& other) const;
	bool operator >(const node_location& other) const;
	bool operator <=(const node_location& other) const;
	std::size_t unique_id() const;
	hpx::future<void> register_client(const node_client& client) const;
//	hpx::future<hpx::id_type> get_id() const;
//	hpx::future<node_client> get_client() const;
	bool is_physical_boundary(integer) const;
	real x_location(integer d) const;
	std::string to_str() const;
	template<class Archive>
	void serialize(Archive& arc, unsigned) {
		arc & lev;
		arc & xloc;
	}
	std::size_t load(FILE* fp);
	std::size_t save(FILE* fp) const;
	std::vector<node_location> get_neighbors() const;
	bool has_neighbor(const geo::direction dir) const;
	node_location get_neighbor(const geo::direction dir) const;
	bool is_child_of(const node_location& other) const;
	bool neighbors_with( const node_location& ) const;
	range_type abs_range() const;
};

namespace hpx { namespace traits
{
    template <>
    struct is_bitwise_serializable<node_location>
      : std::true_type
    {};
}}

#endif /* NODE_LOCATION_HPP_ */
