/*
 * silo.hpp
 *
 *  Created on: Oct 16, 2018
 *      Author: dmarce1
 */

#ifndef SRC_SILO_HPP_
#define SRC_SILO_HPP_

class node_server;
#include <silo.h>

struct silo_var_t {
private:
	std::string name_;
	std::vector<real> data_;
public:
	const char* name() const {
		return name_.c_str();
	}
	void* data() {
		return data_.data();
	}
	const void* data() const {
		return data_.data();
	}
	silo_var_t(const std::string& name, std::size_t = INX);
	~silo_var_t() = default;
	silo_var_t(silo_var_t&&) = default;
	silo_var_t& operator=(silo_var_t&&) = default;
	silo_var_t(const silo_var_t&) = delete;
	silo_var_t& operator=(const silo_var_t&) = delete;
	double& operator()(int i);
	double operator()(int i) const;
};

void output_all(std::string fname, int cycle, bool);

void load_options_from_silo(std::string fname, DBfile* = NULL);

hpx::id_type load_data_from_silo(std::string fname, node_server*, hpx::id_type);

#endif /* SRC_SILO_HPP_ */
