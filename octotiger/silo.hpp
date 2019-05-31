/*
 * silo.hpp
 *
 *  Created on: Oct 16, 2018
 *      Author: dmarce1
 */

#ifndef SRC_SILO_HPP_
#define SRC_SILO_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/real.hpp"

#include <hpx/include/naming.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <silo.h>

class node_server;

struct silo_var_t {
private:
	std::string name_;
	std::vector<real> data_;
	std::pair<real,real> range_;
public:
	void set_range(real val ) {
		range_.first = std::min(range_.first, val);
		range_.second = std::max(range_.second, val);
	}
	real min() const {
		return range_.first;
	}
	real max() const {
		return range_.second;
	}
	std::size_t size() const {
		return data_.size();
	}
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


struct  silo_load_t {
	integer nx;
	std::vector<std::pair<std::string,std::vector<real>>> vars;
	std::vector<std::pair<std::string,real>> outflows;
	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & nx;
		arc & vars;
		arc & outflows;
	}
	std::vector<silo_load_t> decompress();
};



void output_all(std::string fname, int cycle, bool);

void load_options_from_silo(std::string fname, DBfile* = NULL);

OCTOTIGER_EXPORT void load_data_from_silo(std::string fname, node_server*, hpx::id_type);

#endif /* SRC_SILO_HPP_ */
