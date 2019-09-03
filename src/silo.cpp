//Copyright (c) 2019 Dominic C. Marcello

#include "octotiger/silo.hpp"



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
