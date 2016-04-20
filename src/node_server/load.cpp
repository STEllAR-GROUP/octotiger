/*
 * load.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: dmarce1
 */


#include "../diagnostics.hpp"
#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::load_action load_action_type;
HPX_REGISTER_ACTION (load_action_type);



hpx::mutex rec_size_mutex;
integer rec_size = -1;

void set_locality_data( real omega, space_vector pivot, integer record_size ) {
	grid::set_omega(omega);
	grid::set_pivot(pivot);
	rec_size = record_size;
}


hpx::id_type make_new_node(const node_location& loc, const hpx::id_type& _parent) {
	return GET(hpx::new_<node_server>(hpx::find_here(), loc, _parent, ZERO, ZERO));
}


HPX_PLAIN_ACTION(make_new_node, make_new_node_action);
HPX_PLAIN_ACTION(set_locality_data, set_locality_data_action);



hpx::future<grid::output_list_type> node_client::load(integer i, const hpx::id_type& _me, bool do_o,
		std::string s) const {
	return hpx::async<typename node_server::load_action>(get_gid(), i, _me, do_o, s);
}



grid::output_list_type node_server::load(integer cnt, const hpx::id_type& _me, bool do_output, std::string filename) {
	FILE* fp;
	std::size_t read_cnt = 0;

	if (rec_size == -1 && my_location.level() == 0) {
		fp = fopen(filename.c_str(), "rb");
		if( fp == NULL) {
			printf( "Failed to open file\n");
			abort();
		}
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
			futs.push_back(hpx::async<set_locality_data_action>(locality, omega, pivot, rec_size));
		}
		for (auto&& fut : futs) {
			GET(fut);
		}
	}


	static auto localities = hpx::find_all_localities();
	me = _me;
	fp = fopen(filename.c_str(), "rb");
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
	//printf( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n" );
	if (flag == '1') {
		is_refined = true;
		children.resize(NCHILD);
		for (auto& ci : geo::octant::full_set()) {
			integer loc_id = ((cnt * localities.size()) / (total_nodes + 1));
			children[ci] = hpx::async<make_new_node_action>(localities[loc_id], my_location.get_child(ci),
					me.get_gid());
			futs.push_back(children[ci].load(counts[ci], children[ci].get_gid(), do_output, filename));
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
	//printf( "***************************************\n" );
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
