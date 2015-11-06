/*
 * geometry.cpp
 *
 *  Created on: Oct 12, 2015
 *      Author: dmarce1
 */

#ifndef GEOMETRY_CPP_
#define GEOMETRY_CPP_

#include "geometry.hpp"

namespace geo {

direction face::to_direction() const {
	direction dir;
	switch (i) {
	case FXM:
		dir.set(-1, 0, 0);
		break;
	case FXP:
		dir.set(+1, 0, 0);
		break;
	case FYM:
		dir.set(0, -1, 0);
		break;
	case FYP:
		dir.set(0, +1, 0);
		break;
	case FZM:
		dir.set(0, 0, -1);
		break;
	case FZP:
		dir.set(0, 0, +1);
		break;
	default:
		dir = -1;
		break;
	}
	return dir;
}

std::array<face, face::count() / NDIM> face::dimension_subset(const dimension& d) {
	std::array<face, _count / NDIM> a;
	switch (d.i) {
	case XDIM:
		a = { {	0,1}};
		break;
		case YDIM:
		a = { {	2,3}};
		break;
		case ZDIM:
		a = { {	4,5}};
		break;
		default:
		a = { {}};
		assert(false);

	}
	return a;
}

quadrant octant::get_quadrant(const dimension& d) const {
	quadrant q;
	switch (d.i) {
	case XDIM:
		q.i = i >> 1;
		break;
	case YDIM:
		q.i = (i & 1) | ((i >> 1) & 2);
		break;
	case ZDIM:
		q.i = i & 3;
		break;
	default:
		q.i = -1;
		assert(false);
	}
	return q;
}

std::array<octant, octant::count() / 2> octant::face_subset(const face& f) {
	std::array<octant, octant::count() / 2> a;
	switch (f.i) {
	case FXM:
		a = { {	0,2,4,6}};
		break;
		case FXP:
		a = { {	1,3,5,7}};
		break;
		case FYM:
		a = { {	0,1,4,5}};
		break;
		case FYP:
		a = { {	2,3,6,7}};
		break;
		case FZM:
		a = { {	0,1,2,3}};
		break;
		case FZP:
		a = { {	4,5,6,7}};
		break;
		default:
		a = { {}};
		assert(false);
	}
	return a;
}

octant quadrant::get_octant_on_face(const face& f) const {
	octant o;
	const dimension d = f.get_dimension();
	const side s = f.get_side();
	switch (d.i) {
	case XDIM:
		o.i = (i << 1) | s.i;
		break;
	case YDIM:
		o.i = ((i << 1) & 4) | (s.i << 1) | (i & 1);
		break;
	case ZDIM:
		o.i = (i & 3) | (s.i << 2);
		break;
	default:
		o.i = -1;
		assert(false);
	}
	return o;
}

}
#endif /* GEOMETRY_CPP_ */
