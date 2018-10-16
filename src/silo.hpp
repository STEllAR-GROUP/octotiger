/*
 * silo.hpp
 *
 *  Created on: Oct 16, 2018
 *      Author: dmarce1
 */

#ifndef SRC_SILO_HPP_
#define SRC_SILO_HPP_

#include <silo.h>

struct silo_var_t {
	char* name_;
	void* data_;
	silo_var_t(const std::string& name);
	~silo_var_t();
	silo_var_t(silo_var_t&&);
	silo_var_t& operator=(silo_var_t&&);
	silo_var_t(const silo_var_t&) = delete;
	silo_var_t& operator=(const silo_var_t&) = delete;
	double& operator()(int i);
	double operator()(int i) const;
};

void output_all(std::string fname, int cycle);

#endif /* SRC_SILO_HPP_ */
