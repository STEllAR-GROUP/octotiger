/*
 * output.cpp
 *
 *  Created on: Apr 22, 2016
 *      Author: dmarce1
 */

#include "../node_server.hpp"
#include "../node_client.hpp"

typedef node_server::output_action output_action_type;
HPX_REGISTER_ACTION (output_action_type);

hpx::future<grid::output_list_type> node_client::output(std::string fname) const {
	return hpx::async<typename node_server::output_action>(get_gid(), fname);
}

grid::output_list_type node_server::output(std::string fname) const {
	if (is_refined) {
		std::list<hpx::future<grid::output_list_type>> futs;
		for (auto i = children.begin(); i != children.end(); ++i) {
			futs.push_back(i->output(fname));
		}
		auto i = futs.begin();
		grid::output_list_type my_list = GET(*i);
		for (++i; i != futs.end(); ++i) {
			grid::merge_output_lists(my_list, GET(*i));
		}

		if (my_location.level() == 0) {
//			hpx::apply([](const grid::output_list_type& olists, const char* filename) {
			printf("Outputing...\n");
			grid::output(my_list, fname, get_time());
			printf("Done...\n");
			//		}, std::move(my_list), fname.c_str());
		}
		return my_list;

	} else {
		return grid_ptr->get_output_list();
	}

}
