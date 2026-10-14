//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SRC_SILO_HPP_
#define SRC_SILO_HPP_

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/real.hpp"
#include "octotiger/node_location.hpp"

#include <hpx/include/naming.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>



#include <silo.h>


#define SILO_DRIVER DB_HDF5
#define SILO_VERSION 112


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
};



void output_all(node_server* root_ptr, std::string fname, int cycle, bool);

void load_options_from_silo(std::string fname, DBfile* = nullptr, bool restore_integrator = false);

OCTOTIGER_EXPORT void load_data_from_silo(std::string fname, std::shared_ptr<node_server>, hpx::id_type);


template<class T>
struct db_type {
	static constexpr int d = DB_UNKNOWN;
};

template<>
struct db_type<integer> {
	static constexpr int d = DB_LONG_LONG;
};

template<>
struct db_type<real> {
	static constexpr int d = DB_DOUBLE;
};

template<>
struct db_type<char> {
	static constexpr int d = DB_CHAR;
};

template<>
struct db_type<std::int8_t> {
	static constexpr int d = DB_CHAR;
};


static inline std::string oct_to_str(node_location::node_id n) {
	return hpx::util::format("{:llo}", n);
}


static inline std::string outflow_name(const std::string& varname) {
	return varname + std::string("_outflow");
}



int& silo_epoch();
double& silo_output_time();
double& silo_output_rotation_time();

// Raw restart records are separate from the CGS visualization variables.
namespace silo_restart {
constexpr integer format_version = 2;

struct metadata {
    std::vector<integer> layout;
    std::vector<integer> solver_integers;
    std::vector<real> solver_reals;
    std::vector<real> globals;
};

inline void check(int status, const std::string& operation) {
    if (status != 0) {
        throw std::runtime_error("Silo checkpoint: " + operation);
    }
}

template<class T>
void write_array(DBfile* db, const std::string& name, const std::vector<T>& values) {
    if (values.empty() || values.size() > size_t(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Silo checkpoint: invalid array length for " + name);
    }
    const int count = static_cast<int>(values.size());
    check(DBWrite(db, name.c_str(), values.data(), &count, 1, db_type<T>::d),
        "unable to write " + name);
}

template<class T>
std::vector<T> read_array(DBfile* db, const std::string& name, size_t count) {
    const int type = DBGetVarType(db, name.c_str());
    // HDF5 can report a written DB_LONG_LONG as DB_LONG when both native
    // types have the same width. Preserve strict width and byte-count checks.
    const bool compatible_type = type == db_type<T>::d ||
        (db_type<T>::d == DB_LONG_LONG && type == DB_LONG && sizeof(long) == sizeof(T));
    if (count == 0 || count > size_t(std::numeric_limits<int>::max()) / sizeof(T) ||
            !compatible_type ||
            DBGetVarByteLength(db, name.c_str()) != static_cast<int>(count * sizeof(T)) ||
            DBGetVarLength(db, name.c_str()) != static_cast<int>(count)) {
        throw std::runtime_error("Silo checkpoint: invalid type/length for " + name);
    }
    std::vector<T> values(count);
    check(DBReadVar(db, name.c_str(), values.data()), "unable to read " + name);
    return values;
}

metadata snapshot();
void write_metadata(DBfile*, const metadata&);
void restore_metadata(DBfile*);
bool present(DBfile*);
std::vector<char> read_state(DBfile*, node_location::node_id);
} // namespace silo_restart


#endif /* SRC_SILO_HPP_ */
