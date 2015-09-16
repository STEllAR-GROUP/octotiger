/*
 * space_vector.hpp
 *
 *  Created on: Jun 3, 2015
 *      Author: dmarce1
 */

#ifndef SPACE_VECTOR_HPP_
#define SPACE_VECTOR_HPP_

#include <cmath>

class space_vector: public std::array<real, NDIM> {
public:
	template<class Archive>
	void serialize(Archive& arc, unsigned) {
		arc & *(static_cast<std::array<real, NDIM>*>(this));
	}
	space_vector& operator=(real a) {
		for (integer i = 0; i != NDIM; ++i) {
			(*this)[i] = a;
		}
		return *this;
	}
	space_vector& operator=(const space_vector& other) {
		for (integer i = 0; i != NDIM; ++i) {
			(*this)[i] = other[i];
		}
		return *this;
	}
	space_vector& operator+=(const space_vector& other) {
		for (integer i = 0; i != NDIM; ++i) {
			(*this)[i] += other[i];
		}
		return *this;
	}
	space_vector& operator-=(const space_vector& other) {
		for (integer i = 0; i != NDIM; ++i) {
			(*this)[i] -= other[i];
		}
		return *this;
	}
	space_vector& operator*=(real a) {
		for (integer i = 0; i != NDIM; ++i) {
			(*this)[i] *= a;
		}
		return *this;
	}
	space_vector& operator/=(real a) {
		for (integer i = 0; i != NDIM; ++i) {
			(*this)[i] /= a;
		}
		return *this;
	}
	space_vector operator+(const space_vector& other) const {
		space_vector me = *this;
		me += other;
		return me;
	}
	space_vector operator-(const space_vector& other) const {
		space_vector me = *this;
		me -= other;
		return me;
	}
	space_vector operator*(real a) const {
		auto me = *this;
		me *= a;
		return me;
	}
	space_vector operator/(real a) const {
		space_vector me = *this;
		me /= a;
		return me;
	}
	space_vector operator+() const {
		space_vector me = *this;
		return me;
	}
	space_vector operator-() const {
		space_vector me = *this;
		for (integer i = 0; i != NDIM; ++i) {
			me[i] = -me[i];
		}
		return me;
	}
	real abs() const {
		real sum = ZERO;
		for (integer d = 0; d != NDIM; ++d) {
			sum += (*this)[d] * (*this)[d];
		}
		return std::sqrt(sum);
	}

};

#endif /* SPACE_VECTOR_HPP_ */
