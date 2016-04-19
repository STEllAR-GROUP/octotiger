/*
 * node_server_output.cpp
 *
 *  Created on: Jul 16, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"
#include <sys/stat.h>
#include <unistd.h>
#include "future.hpp"

inline bool file_exists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

//HPX_PLAIN_ACTION(grid::set_omega, set_omega_action2);
//HPX_PLAIN_ACTION(grid::set_pivot, set_pivot_action2);

std::size_t node_server::load_me(FILE* fp) {
	std::size_t cnt = 0;
	auto foo = std::fread;
	refinement_flag = false;
	cnt += foo(&step_num, sizeof(integer), 1, fp) * sizeof(integer);
	cnt += foo(&current_time, sizeof(real), 1, fp) * sizeof(real);
	cnt += foo(&rotational_time, sizeof(real), 1, fp) * sizeof(real);
	cnt += grid_ptr->load(fp);
	return cnt;
}

std::size_t node_server::save_me(FILE* fp) const {
	auto foo = std::fwrite;
	std::size_t cnt = 0;

	cnt += foo(&step_num, sizeof(integer), 1, fp) * sizeof(integer);
	cnt += foo(&current_time, sizeof(real), 1, fp) * sizeof(real);
	cnt += foo(&rotational_time, sizeof(real), 1, fp) * sizeof(real);
	assert(grid_ptr != nullptr);
	cnt += grid_ptr->save(fp);
	return cnt;
}

void node_server::save_to_file(const std::string& fname) const {
	save(0,fname);
//	std::string command = std::string("cp ") + fname + std::string(" restart.chk\n");
//	SYSTEM(command);
}

void node_server::load_from_file(const std::string& fname) {
	load(0, hpx::id_type(), false, fname);
}

void node_server::load_from_file_and_output(const std::string& fname, const std::string& outname) {
	load(0, hpx::id_type(), true, fname);
	std::string command = std::string("mv data.silo ") + outname + std::string("\n");
	SYSTEM(command);
}

