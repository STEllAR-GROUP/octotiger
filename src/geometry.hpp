/*
 * geometry.hpp
 *
 *  Created on: Oct 13, 2015
 *      Author: dmarce1
 */

#ifndef GEOMETRY_HPP_
#define GEOMETRY_HPP_

#include "defs.hpp"
#include <cstdlib>
#include <cassert>

namespace geo {

constexpr integer INNER = 0;
constexpr integer OUTER = 1;
constexpr integer MINUS = 0;
constexpr integer PLUS = 1;
constexpr integer FXM = 0;
constexpr integer FXP = 1;
constexpr integer FYM = 2;
constexpr integer FYP = 3;
constexpr integer FZM = 4;
constexpr integer FZP = 5;
constexpr integer XDIM = 0;
constexpr integer YDIM = 1;
constexpr integer ZDIM = 2;

class geo_type;
class side;
class dimension;
class direction;
class face;
class octant;
class quadrant;

class geo_type {
protected:
	integer i;
public:
	geo_type() = default;
	constexpr geo_type(integer);
	operator integer() const;
	template<class Arc>
	void serialize(Arc& arc, const unsigned);

	friend class side;
	friend class dimension;
	friend class face;
	friend class octant;
	friend class quadrant;
};

class side: public geo_type {
	static constexpr integer _count = 2;
public:
	side() = default;
	constexpr side(integer);
	side flip() const;
	static constexpr integer count();
	static constexpr std::array<side, _count> full_set();
};

class dimension: public geo_type {
	static constexpr integer _count = 3;
public:
	dimension() = default;
	constexpr dimension(integer);
	static constexpr integer count();
	static constexpr std::array<dimension, _count> full_set();
};

class face: public geo_type {
	static constexpr integer _count = 6;
public:
	face() = default;
	constexpr face(integer);
	face(const dimension&, const side&);
	dimension get_dimension() const;
	side get_side() const;
	face flip() const;
	direction to_direction() const;
	static constexpr integer count();
	static constexpr std::array<face, _count> full_set();
	static std::array<face, _count / NDIM> dimension_subset(const dimension&);
};

class direction: public geo_type {
	static constexpr integer _count = 26;
public:
	direction() = default;
	constexpr direction(integer j) :
			geo_type(j) {
	}
	integer operator[](const dimension& dim) const {
		switch (dim) {
		case XDIM:
			return (i % NDIM) - 1;
		case YDIM:
			return ((i / NDIM) % NDIM) - 1;
		case ZDIM:
			return ((i / (NDIM * NDIM)) % NDIM) - 1;
		default:
			break;
		}
		assert(false);
		return 0;
	}
	bool is_vertex() const {
		return std::abs((*this)[0]) + std::abs((*this)[1]) + std::abs((*this)[2]) == 3;
	}
	bool is_face() const {
		return std::abs((*this)[0]) + std::abs((*this)[1]) + std::abs((*this)[2]) == 1;
	}
	face to_face() const {
		if (!is_face()) {
			return face(-1);
		} else {
			for (auto& dim : dimension::full_set()) {
				if ((*this)[dim] > 0) {
					return face(dim, PLUS);
				} else if ((*this)[dim] < 0) {
					return face(dim, MINUS);
				}
			}
		}
		assert(false);
		return -1;
	}
	void set(integer x, integer y, integer z) {
		i = 0;

		if (x == 0) {
			i += 1;
		} else if (x > 0) {
			i += 2;
		}

		if (y == 0) {
			i += NDIM;
		} else if (y > 0) {
			i += 2 * NDIM;
		}

		if (z == 0) {
			i += NDIM * NDIM;
		} else if (z > 0) {
			i += 2 * NDIM * NDIM;
		}

	}
	direction flip() const {
		return direction(26 - i);
	}
	operator integer() const {
		return (i < 14) ? i : (i - 1);
	}
	static constexpr integer count() {
		return 26;
	}
	static constexpr std::array<direction, _count> full_set() {
		return { {0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26}};
	}
	static constexpr std::array<direction, _count / 2> half_set() {
		return { {0,1,2,3,4,5,6,7,8,9,10,11,12}};
	}
};

class octant: public geo_type {
	static constexpr integer _count = 8;
public:
	octant() = default;
	constexpr octant(integer);
	octant(const std::array<side, NDIM>&);
	side get_side(const dimension&) const;
	bool is_on_face(const face&) const;
	quadrant get_quadrant(const dimension&) const;
	integer operator[](const dimension& dim) const {
		return (i >> dim) & 1;
	}
	octant neighbor(const direction& dir) {
		integer ci = static_cast<integer>(ZERO);
		for (auto& d : dimension::full_set()) {
			const integer bit = integer(1) << integer(d);
			if (dir[d] == 0) {
				ci |= bit & i;
			} else if (dir[d] > 0) {
				if ((i & bit) == 0) {
					ci |= bit;
				} else {
					ci = -1;
					break;
				}
			} else if (dir[d] < 0) {
				if ((i & bit) != 0) {
					ci &= ~bit;
				} else {
					ci = -1;
					break;
				}
			}
		}
		return octant(ci);
	}
	static constexpr integer count();
	static constexpr std::array<octant, _count> full_set();
	static std::array<octant, _count / 2> face_subset(const face&);
};

class quadrant: public geo_type {
	static constexpr integer _count = 4;
public:
	quadrant() = default;
	constexpr quadrant(integer);
	quadrant(const octant&, const dimension&);
	octant get_octant_on_face(const face&) const;
	side get_side(integer) const;
	static constexpr integer count();
	static constexpr std::array<quadrant, _count> full_set();
};

inline side quadrant::get_side(integer d) const {
	return side((i >> d) & 1);
}

constexpr geo_type::geo_type(integer j) :
		i(j) {
}

constexpr side::side(integer i) :
		geo_type(i) {
}

constexpr dimension::dimension(integer i) :
		geo_type(i) {
}

constexpr face::face(integer i) :
		geo_type(i) {
}

constexpr octant::octant(integer i) :
		geo_type(i) {
}

constexpr quadrant::quadrant(integer i) :
		geo_type(i) {
}

constexpr integer side::count() {
	return _count;
}

constexpr std::array<side, side::count()> side::full_set() {
	return { {0,1}};
}

constexpr integer dimension::count() {
	return _count;
}

constexpr std::array<dimension, dimension::count()> dimension::full_set() {
	return { {0,1,2}};
}

constexpr integer face::count() {
	return _count;
}

constexpr std::array<face, face::count()> face::full_set() {
	return { {0,1,2,3,4,5}};
}

constexpr integer octant::count() {
	return _count;
}

constexpr std::array<octant, octant::count()> octant::full_set() {
	return { {0,1,2,3,4,5,6,7}};
}

constexpr integer quadrant::count() {
	return _count;
}

constexpr std::array<quadrant, 4> quadrant::full_set() {
	return { {0,1,2,3}};
}

inline geo_type::operator integer() const {
	return i;
}

inline face::face(const dimension& d, const side& s) {
	i = 2 * d.i + s.i;
}

inline dimension face::get_dimension() const {
	return dimension(i / 2);
}

inline side octant::get_side(const dimension& d) const {
	return side((i >> d.i) & 1);
}

inline bool octant::is_on_face(const face& f) const {
	return f.get_side() == get_side(f.get_dimension());
}

inline side face::get_side() const {
	return side(i & 1);
}

template<class Arc>
void geo_type::serialize(Arc& arc, const unsigned) {
	arc & i;
}

inline quadrant::quadrant(const octant& o, const dimension& d) :
		geo_type(o.get_quadrant(d)) {
}

inline octant::octant(const std::array<side, NDIM>& sides) :
		geo_type(sides[XDIM] | (sides[YDIM] << 1) | (sides[ZDIM] << 2)) {
}

inline side side::flip() const {
	return side(i ^ 1);
}

inline face face::flip() const {
	return face(i ^ 1);
}

}


constexpr integer INNER = 0;
constexpr integer OUTER = 1;

integer get_boundary_size(std::array<integer, NDIM>&, std::array<integer, NDIM>&, const geo::direction&, const geo::side&, integer inx, integer bw);

#endif /* GEOMETRY_HPP_ */
