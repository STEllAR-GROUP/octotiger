/*
 * state.hpp
 *
 *  Created on: Oct 13, 2016
 *      Author: dmarce1
 */

#ifndef STATE_HPP_
#define STATE_HPP_

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
