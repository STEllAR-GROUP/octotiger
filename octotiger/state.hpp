//  Copyright (c) 2019 Dominic C Marcello
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef STATE_HPP_
#define STATE_HPP_

#include "octotiger/real.hpp"

#include <array>

class state {
private:
	std::array<real, NF> a;
public:
	const real& operator()(int i) const {
		return a[i];
	}
	real& operator()(int i) {
		return a[i];
	}
	template<class Archive>
	void serialize(Archive& arc, unsigned) {
		arc & a;
	}
};

HPX_IS_BITWISE_SERIALIZABLE(state);

#endif /* STATE_HPP_ */
