/*
 * node_location.cpp
 *
 *  Created on: Jun 11, 2015
 *      Author: dmarce1
 */

#ifndef NODE_LOCATION_CPP_
#define NODE_LOCATION_CPP_

/*
 * node_location.hpp
 *
 *  Created on: Jun 11, 2015
 *      Author: dmarce1
 */

#include "octotiger/node_location.hpp"
#include "octotiger/node_client.hpp"

#include <cstdio>

range_type intersection(const range_type& r1, const range_type& r2) {
	range_type ri;
	for (int d = 0; d < NDIM; d++) {
		ri[d].first = std::max(r1[d].first, r2[d].first);
		ri[d].second = std::min(r1[d].second, r2[d].second);
		if (ri[d].first > ri[d].second) {
			for (int d1 = 0; d1 < NDIM; d1++) {
				ri[d1].first = ri[d1].second = -1;
				break;
			}
		}
	}
	return ri;
}


range_type node_location::abs_range() const {
	range_type range;
	integer shift = opts().max_level - level();
	for( int d = 0; d < NDIM; d++) {
		range[d].first = (INX*xloc[d]) << shift;
		range[d].second = (INX*(xloc[d]+1)) << shift;
	}
	return std::move(range);
}



bool node_location::neighbors_with( const node_location& n ) const {
	node_location n1, n2;
	if( level() > n.level()) {
		n1 = *this;
		n2 = n;
	} else {
		n2 = *this;
		n1 = n;
	}
	integer max1[NDIM], min1[NDIM];
	integer max2[NDIM], min2[NDIM];
	for( int d = 0; d < NDIM; d++) {
		min1[d] = n1.xloc[d];
		max1[d] = n1.xloc[d];
	}
	integer span = 1;
	while( n1.level() > n2.level()) {
		n2.lev++;
		for( int d = 0; d < NDIM; d++ ) {
			n2.xloc[d] *= 2;
		}
		span *= 2;
	}
	for (int d = 0; d < NDIM; d++) {
		min2[d] = n2.xloc[d];
		max2[d] = n2.xloc[d] + span - 1;
	}
	bool rc = false;
	for (int d = 0; d < NDIM; d++) {
		if (std::abs(max2[d] - min1[d]) == 1) {
			rc = true;
		} else if (std::abs(min2[d] - max1[d]) == 1) {
			rc = true;
		}
		if (rc) {
			break;
		}
	}
	return rc;
}


node_location::node_id node_location::to_id() const {
	node_id id = 1;
	for (int l = 0; l < lev; l++) {
		for (int d = 0; d < NDIM; d++) {
			id <<= 1;
			id |= ((xloc[d] >> l) & 1);
		}
	}
#ifdef DEBUG
	node_location test;
	test.from_id(id);
	assert(test == *this);
#endif
	return id;
}

node_location::node_location(node_location::node_id id) {
	this->from_id(id);
}

void node_location::from_id(const node_id& id_) {
	node_id id = id_;
	for (int d = 0; d < NDIM; d++) {
		xloc[d] = 0;
	}
	for (lev = 0; id != 1; lev++) {
		for (int d = NDIM - 1; d >= 0; d--) {
			xloc[d] <<= 1;
			xloc[d] |= (id & 1);
			id >>= 1;
		}
	}
}

std::size_t node_location::hash() const {
	return std::size_t(this->to_id());
}

std::vector<node_location> node_location::get_neighbors() const {
	std::vector<node_location> locs;
	locs.reserve(NDIM * NDIM * NDIM - 1);
	const integer lb = 0;
	const integer ub = (1 << lev) - 1;
	for (integer i = -1; i <= +1; ++i) {
		for (integer j = -1; j <= +1; ++j) {
			for (integer k = -1; k <= +1; ++k) {
				if (i != 0 || j != 0 || k != 0) {
					node_location this_loc(*this);
					this_loc.xloc[XDIM] += i;
					this_loc.xloc[YDIM] += j;
					this_loc.xloc[ZDIM] += k;
					bool in = true;
					for (integer d = 0; d != NDIM; ++d) {
						if (this_loc.xloc[d] < lb || this_loc.xloc[d] > ub) {
							in = false;
							break;
						}
					}
					if (in) {
						locs.push_back(std::move(this_loc));
					}
				}
			}
		}
	}
	return locs;
}

std::size_t node_location::load(FILE* fp) {
	std::size_t cnt = 0;
	cnt += fread(&lev, sizeof(integer), 1, fp) * sizeof(integer);
	cnt += fread(xloc.data(), sizeof(integer), NDIM, fp) * sizeof(integer);
	return cnt;
}

std::size_t node_location::save(FILE* fp) const {
	std::size_t cnt = 0;
	cnt += fwrite(&lev, sizeof(integer), 1, fp) * sizeof(integer);
	cnt += fwrite(xloc.data(), sizeof(integer), NDIM, fp) * sizeof(integer);
	return cnt;
}

geo::side node_location::get_child_side(const geo::dimension& d) const {
	return (xloc[d] & 1) ? geo::PLUS : geo::MINUS;
}

geo::octant node_location::get_child_index() const {
	return geo::octant(std::array<geo::side, NDIM>( { { get_child_side(XDIM), get_child_side(YDIM), get_child_side(ZDIM) } }));
}

bool node_location::is_child_of(const node_location& other) const {
	bool rc;
	if (other.lev < lev) {
		rc = true;
		for (integer d = 0; d != NDIM; ++d) {
			if ((xloc[d] >> (lev - other.lev)) != other.xloc[d]) {
				rc = false;
				break;
			}
		}
	} else {
		rc = false;
	}
	return rc;
}

real node_location::x_location(integer d) const {
	const real dx = TWO / real(1 << lev);
	return real(xloc[d]) * dx - 1.0;
}

node_location::node_location() {
	lev = 0;
	xloc = { {0,0,0}};
}

integer node_location::level() const {
	return lev;
}

node_location::node_location(const node_location& other) {
	*this = other;
}

node_location& node_location::operator=(const node_location& other) {
	lev = other.lev;
	xloc = other.xloc;
	return *this;
}

node_location node_location::get_child(integer x, integer y, integer z) const {
	node_location child;
	child.lev = lev + 1;
	child.xloc[XDIM] = 2 * xloc[XDIM] + x;
	child.xloc[YDIM] = 2 * xloc[YDIM] + y;
	child.xloc[ZDIM] = 2 * xloc[ZDIM] + z;
	return child;
}

node_location node_location::get_child(integer c) const {
	return get_child((c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1);
}

std::string node_location::to_str() const {
	char buffer[100];    // 21 bytes for int (max) + some leeway
	sprintf(buffer, "%i_%i_%i_%i", int(lev), int(xloc[XDIM]), int(xloc[YDIM]), int(xloc[ZDIM]));
	return std::string(buffer);
}

node_location node_location::get_parent() const {
	assert(lev >= 1);
	node_location parent;
	parent.lev = lev - 1;
	for (integer d = 0; d != NDIM; ++d) {
		parent.xloc[d] = xloc[d] / 2;
	}
	return parent;
}

node_location node_location::get_sibling(integer face) const {
	node_location sibling(*this);
	const integer dim = face / 2;
	const integer dir = face % 2;
	if (dir == 0) {
		sibling.xloc[dim]--;
		assert(sibling.xloc[dim] >= 0);
	} else {
		sibling.xloc[dim]++;
		assert(sibling.xloc[dim] < (1 << lev));
	}
	return sibling;
}

bool node_location::operator==(const node_location& other) const {
	bool rc = true;
	if (lev != other.lev) {
		rc = false;
	} else {
		for (integer d = 0; d != NDIM; ++d) {
			if (xloc[d] != other.xloc[d]) {
				rc = false;
				break;
			}
		}
	}
	return rc;
}

bool node_location::operator!=(const node_location& other) const {
	return !(*this == other);
}

bool node_location::operator<(const node_location& other) const {
	bool rc = false;
	if (lev < other.lev) {
		rc = true;
	} else if (lev == other.lev) {
		for (integer d = 0; d != NDIM; ++d) {
			if (xloc[d] < other.xloc[d]) {
				rc = true;
				break;
			} else if (xloc[d] > other.xloc[d]) {
				break;
			}
		}
	}
	return rc;
}

bool node_location::operator >=(const node_location& other) const {
	return !(*this == other);
}

bool node_location::operator >(const node_location& other) const {
	return !(*this == other) && !(*this < other);
}

bool node_location::operator <=(const node_location& other) const {
	return (*this == other) || (*this < other);
}

std::size_t node_location::unique_id() const {
	std::size_t id = 1;
	std::array<std::size_t, NDIM> x;
	for (integer d = 0; d != NDIM; ++d) {
		x[d] = std::size_t(xloc[d]);
	}
	for (integer l = 0; l != lev; ++l) {
		for (integer d = 0; d != NDIM; ++d) {
			id <<= std::size_t(1);
			id |= x[d] & 1;
			x[d] >>= std::size_t(1);
		}
	}
	return id;
}
/*
 hpx::future<node_client> node_location::get_client() const {
 return hpx::async([](node_location loc) -> node_client {
 auto f = hpx::find_id_from_basename("node_location", loc.unique_id());
 return node_client(std::move(f));
 }, *this);
 }*/
/*
 hpx::future<hpx::id_type> node_location::get_id() const {
 return hpx::find_id_from_basename("node_location", unique_id());
 }
 */

node_location node_location::get_neighbor(const geo::direction dir) const {
	node_location nloc;
	nloc = *this;
	for (auto d : geo::dimension::full_set()) {
		nloc.xloc[d] += dir[d];
	}
	return nloc;
}

bool node_location::has_neighbor(const geo::direction dir) const {
	bool rc = true;
	;
	for (auto d : geo::dimension::full_set()) {
		if (dir[d] == -1) {
			if (xloc[d] == 0) {
				rc = false;
				break;
			}
		} else if (dir[d] == +1) {
			if (xloc[d] == ((1 << level()) - 1)) {
				rc = false;
				break;
			}
		}
	}
	return rc;
}

bool node_location::is_physical_boundary(integer face) const {
	bool rc = false;
	const integer dim = face / 2;
	const integer dir = face % 2;
	if (dir == 0) {
		if (xloc[dim] == integer(0)) {
			rc = true;
		}
	} else {
		if (xloc[dim] == (integer(1) << lev) - integer(1)) {
			rc = true;
		}
	}
	return rc;
}

#endif /* NODE_LOCATION_CPP_ */
