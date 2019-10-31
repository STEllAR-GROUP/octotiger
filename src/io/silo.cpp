//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/io/silo.hpp"

constexpr int db_type<integer>::d;
constexpr int db_type<char>::d;
constexpr int db_type<real>::d;


double& silo_output_time() {
	static double t_;
	return t_;
}

double& silo_output_rotation_time() {
	static double t_;
	return t_;
}


int& silo_epoch() {
	static int t_ = 0;
	return t_;
}
