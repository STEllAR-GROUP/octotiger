/*
 * main.cpp
 *
 *  Created on: May 24, 2016
 *      Author: dmarce1
 */

#include "sphere_points.hpp"

void sphere_point::normalize() {
	const real n = norm();
	if (n != 0.0) {
		nx /= n;
		ny /= n;
		nz /= n;
	}
}

real sphere_point::norm() const {
	return std::sqrt(nx * nx + ny * ny + nz * nz);
}

sphere_point sphere_point::operator/(real den) const {
	sphere_point p = *this;
	p.nx /= den;
	p.ny /= den;
	p.nz /= den;
	return p;
}

sphere_point operator+(const sphere_point& pt1, const sphere_point& pt2) {
	sphere_point p;
	p.nx = pt1.nx + pt2.nx;
	p.ny = pt1.ny + pt2.ny;
	p.nz = pt1.nz + pt2.nz;
	return p;
}

sphere_point operator-(const sphere_point& pt1, const sphere_point& pt2) {
	sphere_point p;
	p.nx = pt1.nx - pt2.nx;
	p.ny = pt1.ny - pt2.ny;
	p.nz = pt1.nz - pt2.nz;
	return p;
}

void triangle::normalize() {
	A.normalize();
	B.normalize();
	C.normalize();
}
real triangle::area() const {
	sphere_point v1 = A - B;
	sphere_point v2 = A - C;
	sphere_point cp;
	cp.nx = +(v1.nx * v2.ny - v1.ny * v2.nx);
	cp.ny = -(v1.nx * v2.nz - v1.nz * v2.nx);
	cp.nz = +(v1.ny * v2.nz - v1.nz * v2.ny);
	return 0.5 * cp.norm();
}
void triangle::split_triangle(std::vector<sphere_point>& sphere_points, int this_lev, int max_lev) {
	if (this_lev < max_lev) {
		triangle t1, t2, t3, t4;
		t1.A = A;
		t1.B = (A + B) / 2.0;
		t1.C = (A + C) / 2.0;

		t2.B = B;
		t2.A = (B + A) / 2.0;
		t2.C = (B + C) / 2.0;

		t3.C = C;
		t3.A = (C + A) / 2.0;
		t3.B = (C + B) / 2.0;

		t4.A = (A + B) / 2.0;
		t4.B = (B + C) / 2.0;
		t4.C = (C + A) / 2.0;

		t1.normalize();
		t2.normalize();
		t3.normalize();
		t4.normalize();

		t1.split_triangle(sphere_points, this_lev + 1, max_lev);
		t2.split_triangle(sphere_points, this_lev + 1, max_lev);
		t3.split_triangle(sphere_points, this_lev + 1, max_lev);
		t4.split_triangle(sphere_points, this_lev + 1, max_lev);
	} else {
		sphere_point p;
		p = (A + B + C) / 3.0;
		p.normalize();
		p.dA = area();
		sphere_points.push_back(p);
		//		printf("%e %e %e %e\n", p.nx, p.ny, p.nz, area());
	}
}

std::vector<sphere_point> generate_sphere_points(int nlev) {
	triangle t1, t2, t3, t4;
	std::vector<sphere_point> sphere_points;

	const real _0 = 0.0;
	const real _90 = 1.0 * M_PI / 2.0;
	const real _180 = 2.0 * M_PI / 2.0;
	const real _270 = 3.0 * M_PI / 2.0;
	const real c = _90;

	t1.A.nx = cos(_0) * sin(_0);
	t1.A.ny = sin(_0) * sin(_0);
	t1.A.nz = cos(_0);

	t1.B.nx = cos(_0) * sin(c);
	t1.B.ny = sin(_0) * sin(c);
	t1.B.nz = cos(c);

	t1.C.nx = cos(_90) * sin(c);
	t1.C.ny = sin(_90) * sin(c);
	t1.C.nz = cos(c);

	t2.A.nx = cos(_0) * sin(_0);
	t2.A.ny = sin(_0) * sin(_0);
	t2.A.nz = cos(_0);

	t2.B.nx = cos(_90) * sin(c);
	t2.B.ny = sin(_90) * sin(c);
	t2.B.nz = cos(c);

	t2.C.nx = cos(_180) * sin(c);
	t2.C.ny = sin(_180) * sin(c);
	t2.C.nz = cos(c);

	t3.A.nx = cos(_0) * sin(_0);
	t3.A.ny = sin(_0) * sin(_0);
	t3.A.nz = cos(_0);

	t3.B.nx = cos(_180) * sin(c);
	t3.B.ny = sin(_180) * sin(c);
	t3.B.nz = cos(c);

	t3.C.nx = cos(_270) * sin(c);
	t3.C.ny = sin(_270) * sin(c);
	t3.C.nz = cos(c);

	t4.A.nx = cos(_0) * sin(0);
	t4.A.ny = sin(_0) * sin(0);
	t4.A.nz = cos(0);

	t4.B.nx = cos(_270) * sin(c);
	t4.B.ny = sin(_270) * sin(c);
	t4.B.nz = cos(c);

	t4.C.nx = cos(_0) * sin(c);
	t4.C.ny = sin(_0) * sin(c);
	t4.C.nz = cos(c);

	t1.split_triangle(sphere_points, 0, nlev);
	t2.split_triangle(sphere_points, 0, nlev);
	t3.split_triangle(sphere_points, 0, nlev);
	t4.split_triangle(sphere_points, 0, nlev);

	t4.A.nz = t3.A.nz = t2.A.nz = t1.A.nz = -1.0;

	t1.split_triangle(sphere_points, 0, nlev);
	t2.split_triangle(sphere_points, 0, nlev);
	t3.split_triangle(sphere_points, 0, nlev);
	t4.split_triangle(sphere_points, 0, nlev);

	real total_area = 0.0;
	for (const auto& pt : sphere_points) {
		total_area += pt.dA;
	}
	for (auto& pt : sphere_points) {
		pt.dA *= 4.0 * M_PI / total_area;
		pt.dl = std::abs(pt.nx) + std::abs(pt.ny) + std::abs(pt.nz);
		pt.wx = std::abs(pt.nx) / pt.dl;
		pt.wy = std::abs(pt.ny) / pt.dl;
		pt.wz = std::abs(pt.nz) / pt.dl;
		printf( "%e %e %e\n", pt.nx, pt.ny, pt.nz);
	}
	printf( "Generated %i sphere points\n", int(sphere_points.size()));
	return sphere_points;
}


geo::octant sphere_point::get_octant() const {
// 	geo::octant octant;
	const integer ix = nx > 0.0 ? 1 : 0;
	const integer iy = ny > 0.0 ? 1 : 0;
	const integer iz = nz > 0.0 ? 1 : 0;
	return geo::octant(4 * iz + 2 * iy + ix);
}

