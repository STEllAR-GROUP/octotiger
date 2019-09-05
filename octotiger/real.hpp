//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef REAL_HPP_
#define REAL_HPP_

//#define DIAGNOSTIC_MODE


using real_type = double;


#ifdef DIAGNOSTIC_MODE

#include <atomic>
#include <cmath>
#include <utility>

#define ARITHMETIC_OPERATOR( op ) \
	real& operator op##=( const real& other ) { \
		r op##= other.r; \
		++counter; \
		return *this; \
	} \
	real operator op( const real& other ) const { \
		real ret(*this); \
		ret += other; \
		++counter; \
		return ret; \
	}

#define LOGICAL_OPERATOR( op ) \
	bool operator op (const real& other ) const { \
		return r op other.r; \
	}

class real {
private:
	static std::atomic<std::size_t> counter;
	real_type r;
public:
	real() = default;
	real(const real&) = default;
	real(real&&) = default;
	real& operator=(const real&) = default;
	real& operator=(real&&) = default;
	~real() = default;

	template<class Arc>
	void serialize(Arc& arc, const unsigned) {
		arc & r;
	}

	operator real_type() const {
		return r;
	}
	real(const real_type& _r) :
			r(_r) {
	}
	real(real_type&& _r) :
			r(std::move(_r)) {
	}
	real& operator=(const real_type& other) {
		r = other;
		return *this;
	}
	real& operator=(real_type&& other) {
		r = std::move(other);
		return *this;
	}

	ARITHMETIC_OPERATOR(+)
	ARITHMETIC_OPERATOR(-)
	ARITHMETIC_OPERATOR(*)
	ARITHMETIC_OPERATOR(/)

	LOGICAL_OPERATOR(==)
	LOGICAL_OPERATOR(!=)
	LOGICAL_OPERATOR(>)
	LOGICAL_OPERATOR(<)
	LOGICAL_OPERATOR(<=)
	LOGICAL_OPERATOR(>=)

	template<class other_type>
	friend real pow(const real& a, const other_type& b);

	friend real sqrt(const real& other);

};

template<class other_type>
inline real pow(const real& a, const other_type& b) {
	real ret = std::pow(a.r, b);
	++real::counter;
	return ret;
}

inline real sqrt(const real& a) {
	real ret = std::sqrt(a.r);
	++real::counter;
	return ret;
}

#else

using real = real_type;

#endif

#endif /* REAL_HPP_ */
