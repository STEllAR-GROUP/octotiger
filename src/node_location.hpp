/*
 * node_location.hpp
 *
 *  Created on: Jun 11, 2015
 *      Author: dmarce1
 */

#ifndef NODE_LOCATION_HPP_
#define NODE_LOCATION_HPP_

#include "defs.hpp"
#include "geometry.hpp"

#include <hpx/include/components.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

class node_client;

class node_location {
private:
	std::array<integer, NDIM> xloc;
	integer lev;
public:
	node_location();
	node_location(const node_location& other);
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
};

namespace hpx { namespace traits
{
    template <>
    struct is_bitwise_serializable<node_location>
      : std::true_type
    {};
}}

#endif /* NODE_LOCATION_HPP_ */
