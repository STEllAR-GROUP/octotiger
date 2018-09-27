/*
 * opacities.hpp
 *
 *  Created on: Sep 25, 2018
 *      Author: dmarce1
 */

#ifndef SRC_RADIATION_OPACITIES_HPP_
#define SRC_RADIATION_OPACITIES_HPP_

#include "../safe_math.hpp"
#include "../physcon.hpp"


template<class U>
U temperature(U rho, U e, U mmw) {
	constexpr U gm1 = U(2.0) / U(3.0);
	return (U(gm1) * U(mmw) * U(physcon.mh) / U(physcon.kb)) * (e * INVERSE( rho ));
}




template<class U>
U kappa_R(U rho, U e, U mmw) {

	// TODO calculate realistic Z
	const U Z = U(0.0);
	const U X = U(0.0);

	constexpr U zbar = U(2.0); //GENERALIZE
	const U T = temperature(rho, e, mmw);
	const U f1 = (T * T + U(2.7e+11) * rho);
	const U f2 = (U(1.0) + std::pow(T / U(4.5e+8), U(0.86)));
	const U k_ff_bf = U(4.0e+25) * (U(1)+X)*(Z + U(0.0001)) * rho * std::pow(SQRT(INVERSE(T)), U(7));
	const U k_T = (U(1.0)+X)*U(0.2) * T * T / (f1 * f2);
	//const U k_cond = U(2.6e-7) * zbar * (T * T) / (rho * rho) * (U(1.0) + std::pow(rho / U(2.0e+6), U(2.0) / U(3.0)));
	const U k_rad = (k_ff_bf + k_T);
	const U k_tot = k_rad;
	//	const U k_tot = k_rad * k_cond / (k_rad + k_cond);
	//const U k_tot = k_rad;;
	return rho * k_tot;
}


template<class U>
U kappa_p(U rho, U e, U mmw) {
	// TODO calculate realistic Z
	const U Z = U(0.0);

	const U T = temperature(rho, e, mmw);
	const U k = U(30.262) * U(4.0e+25) * (Z + U(0.0001)) * rho * std::pow(INVERSE(SQRT(T)), U(7));
	return rho * k;
}

template<class U>
U B_p(U rho, U e, U mmw) {
	const U T = temperature(rho, e, mmw);
	return (U(physcon.sigma) / U(M_PI)) * T * T * T * T;
}



#endif /* SRC_RADIATION_OPACITIES_HPP_ */
