/*
 * node_server_output.cpp
 *
 *  Created on: Jul 16, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"
#include <sys/stat.h>
#include "future.hpp"

hpx::mutex rec_size_mutex;
integer rec_size = -1;

inline bool file_exists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

integer node_server::save(integer cnt) const {
	char flag = is_refined ? '1' : '0';
	FILE* fp = fopen("data.bin", (cnt == 0) ? "wb" : "ab");
	fwrite(&flag, sizeof(flag), 1, fp);
	++cnt;
//	printf("                                   \rSaved %li sub-grids\r", (long int) cnt);
	integer value = cnt;
	std::array<integer, NCHILD> values;
	for (auto& ci : geo::octant::full_set()) {
		if (ci != 0 && is_refined) {
			value += child_descendant_count[ci - 1];
		}
		values[ci] = value;
		fwrite(&value, sizeof(value), 1, fp);
	}
	const integer record_size = save_me(fp) + sizeof(flag) + NCHILD * sizeof(integer);
	fclose(fp);
	if (is_refined) {
		for (auto& ci : geo::octant::full_set()) {
			cnt = children[ci].save(cnt);
		}
	}

	if (my_location.level() == 0) {
		FILE* fp = fopen("data.bin", "ab");
		real omega = grid::get_omega();
		space_vector pivot = grid::get_pivot();
		fwrite(&omega, sizeof(real), 1, fp);
		for (auto& d : geo::dimension::full_set()) {
			fwrite(&(pivot[d]), sizeof(real), 1, fp);
		}
		fwrite(&record_size, sizeof(integer), 1, fp);
		fclose(fp);
		printf("Saved %li grids to checkpoint file\n", (long int) cnt);
	}

	return cnt;
}

hpx::id_type make_new_node(const node_location& loc, const hpx::id_type& _parent) {
	return GET(hpx::new_<node_server>(hpx::find_here(), loc, _parent, ZERO, ZERO));
}


HPX_PLAIN_ACTION(grid::set_omega, set_omega_action2);
HPX_PLAIN_ACTION(grid::set_pivot, set_pivot_action2);

HPX_PLAIN_ACTION(make_new_node, make_new_node_action);

grid::output_list_type node_server::load(integer cnt, const hpx::id_type& _me, bool do_output) {
	FILE* fp;
	std::size_t read_cnt = 0;
	if (rec_size == -1) {
		std::lock_guard<hpx::mutex> lock(rec_size_mutex);
		fp = fopen("data.bin", "rb");
		fseek(fp, -sizeof(integer), SEEK_END);
		read_cnt += fread(&rec_size, sizeof(integer), 1, fp);
		fseek(fp, -4 * sizeof(real) - sizeof(integer), SEEK_END);
		real omega;
		space_vector pivot;
		read_cnt += fread(&omega, sizeof(real), 1, fp);
		for (auto& d : geo::dimension::full_set()) {
			read_cnt += fread(&(pivot[d]), sizeof(real), 1, fp);
		}
		fclose(fp);
		auto localities = hpx::find_all_localities();
		std::vector<hpx::future<void>> futs;
		futs.reserve(localities.size());
		for (auto& locality : localities) {
			futs.push_back(hpx::async<set_omega_action2>(locality, omega));
			if (current_time == ZERO) {
				futs.push_back(hpx::async<set_pivot_action2>(locality, pivot));
			}
		}
		for (auto&& fut : futs) {
			GET(fut);
		}
	}
	static auto localities = hpx::find_all_localities();
	me = _me;
	fp = fopen("data.bin", "rb");
	char flag;
	fseek(fp, cnt * rec_size, SEEK_SET);
	read_cnt += fread(&flag, sizeof(char), 1, fp);
	std::vector<integer> counts(NCHILD);
	for (auto& this_cnt : counts) {
		read_cnt += fread(&this_cnt, sizeof(integer), 1, fp);
	}
	load_me(fp);
	fseek(fp, 0L, SEEK_END);
	integer total_nodes = ftell(fp) / rec_size;
	fclose(fp);
	std::list<hpx::future<grid::output_list_type>> futs;
	if (flag == '1') {
		is_refined = true;
		children.resize(NCHILD);
		for (auto& ci : geo::octant::full_set()) {
			integer loc_id = ((cnt * localities.size()) / (total_nodes + 1));
			children[ci] = hpx::async<make_new_node_action>(localities[loc_id], my_location.get_child(ci),
					me.get_gid());
			futs.push_back(children[ci].load(counts[ci], children[ci].get_gid(), do_output));
		}
	} else if (flag == '0') {
		is_refined = false;
		children.clear();
	} else {
		printf("Corrupt checkpoint file\n");
		sleep(10);
		abort();
	}
	grid::output_list_type my_list;
	for (auto&& fut : futs) {
		if (do_output) {
			grid::merge_output_lists(my_list, GET(fut));
		} else {
			GET(fut);
		}
	}
	if (!is_refined && do_output) {
		my_list = grid_ptr->get_output_list();
	//	grid_ptr = nullptr;
	}
//	hpx::async<inc_grids_loaded_action>(localities[0]).get();
	if (my_location.level() == 0) {
		if (do_output) {
			diagnostics();
			grid::output(my_list, "data.silo", current_time);
		}
		printf("Loaded checkpoint file\n");
		my_list = decltype(my_list)();

	}
	return my_list;
}

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
	save();
	std::string command = std::string("cp data.bin ") + fname + ("\n");
	SYSTEM(command);
	command = std::string("mv data.bin restart.chk\n");
	SYSTEM(command);
}

void node_server::load_from_file(const std::string& fname) {
	std::string command = std::string("ln -s ") + fname + (" data.bin\n");
	SYSTEM(command);
	load(0, hpx::id_type(), false);
	SYSTEM(std::string("rm -f data.bin\n"));
}

void node_server::load_from_file_and_output(const std::string& fname, const std::string& outname) {
	std::string command = std::string("ln -s ") + fname + (" data.bin\n");
	SYSTEM(command);
	load(0, hpx::id_type(), true);
	SYSTEM(std::string("rm -f data.bin\n"));
	command = std::string("mv data.silo ") + outname + std::string("\n");
	SYSTEM(command);
}

