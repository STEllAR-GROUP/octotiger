/*
 * node_server_output.cpp
 *
 *  Created on: Jul 16, 2015
 *      Author: dmarce1
 */

#include "node_server.hpp"
#include <sys/stat.h>
#include "future.hpp"


inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}


grid::output_list_type node_server::output(std::string fname) const {
	if( is_refined ) {
		std::list<hpx::future<grid::output_list_type>> futs;
		for( auto i = children.begin(); i != children.end(); ++i) {
			futs.push_back(i->output());
		}
		auto i = futs.begin();
		grid::output_list_type my_list = GET(*i);
		for(++i; i != futs.end(); ++i) {
			grid::output_list_type child_list = GET(*i);
			grid::merge_output_lists(my_list, child_list);
		}

		if( my_location.level() == 0 ) {
			hpx::apply([](const grid::output_list_type& olists, const char* filename) {
				printf( "Outputing...\n");
				grid::output(olists, filename);
				printf( "Done...\n");
			}, std::move(my_list), fname.c_str());
		}
		return my_list;

	} else {
		return grid_ptr->get_output_list();
	}

}


void my_system( const std::string& command) {
//	printf( "Executing system command: %s\n", command.c_str());
	if( system( command.c_str()) != EXIT_SUCCESS ) {
		assert(false);
	}
}


std::pair<std::size_t,std::size_t> node_server::save(integer loc_id, std::string fname) const {
	static hpx::lcos::local::spinlock mtx;


	std::size_t total_cnt = 0;
	std::size_t bytes_written = 0;
	integer nloc = hpx::find_all_localities().size();
	if( my_location.level() == 0 && loc_id == 0 ) {
		if( file_exists( fname )) {
			std::string command = "rm ";
			command += fname;
			if(system( command.c_str())){}
		}
		for( integer i = 0; i != nloc; ++i) {
			auto this_name = fname + std::string(".") + std::to_string(integer(hpx::get_locality_id()));
			if( file_exists( this_name)) {
				std::string command = "rm ";
				command += this_name;
				if(system( command.c_str())){}
			}
		}
	}

	std::vector<hpx::future<std::pair<std::size_t,std::size_t> >> sfuts;

	if( is_refined ) {
        sfuts.reserve(children.size());
		for( auto i = children.begin(); i != children.end(); ++i) {
			sfuts.push_back( i->save(loc_id, fname));
		}
	}

	{
		boost::lock_guard<hpx::lcos::local::spinlock> file_lock(mtx);
		FILE* fp = fopen( (fname + std::string(".") + std::to_string(integer(hpx::get_locality_id()))).c_str(), "ab");
		total_cnt++;
		bytes_written += my_location.save(fp);
		bytes_written += save_me(fp);
		fclose(fp);
	}

	if( is_refined ) {
		for( auto i = sfuts.begin(); i != sfuts.end(); ++i) {
			auto tmp = i->get();
			total_cnt += tmp.first;
			bytes_written += tmp.second;
		}
	}

	if( loc_id == 0  && my_location.level() == 0) {
// 		hpx::apply([=](std::size_t bytes_written) {
		FILE* fp = fopen("size.tmp2", "wb");
		bytes_written += 2 * fwrite( &total_cnt, sizeof(integer), 1, fp) * sizeof(integer);
		std::size_t tmp = bytes_written + 2 * sizeof(std::size_t) + sizeof(real);
		bytes_written += 2 * fwrite( &tmp, sizeof(std::size_t), 1, fp)*sizeof(std::size_t);
		fclose( fp);
		my_system( "cp size.tmp2 size.tmp1\n");
		fp = fopen("size.tmp1", "ab");
		real omega = grid::get_omega();
		bytes_written += fwrite( &omega, sizeof(real), 1, fp) * sizeof(real);
		fclose( fp);
		std::string command = "rm -r -f " + fname + "\n";
		my_system (command);
		command = "cat size.tmp1 ";
		for( integer i = 0; i != nloc; ++i ) {
				command += fname + "." + std::to_string(integer(i)) + " ";
		}
		command += "size.tmp2 > " + fname + "\n";
		my_system( command);
		command = "rm -f -r " + fname + ".*\n";
		my_system( command);
		my_system( "rm -r -f size.tmp?\n");
		printf( "Saved %i sub-grids with %lli bytes written\n", int(total_cnt), (long long)(bytes_written));
// 		}, bytes_written);
	}
	return std::make_pair(total_cnt, bytes_written);

}

std::size_t node_server::load_me( FILE* fp, integer& locality_id) {
	std::size_t cnt = 0;
	auto foo = std::fread;

//cnt += foo(&is_refined, sizeof(bool), 1, fp)*sizeof(bool);
	cnt += foo(&locality_id, sizeof(integer), 1, fp)*sizeof(integer);
	cnt += foo(&step_num, sizeof(integer), 1, fp)*sizeof(integer);
	cnt += foo(&current_time,sizeof(real), 1, fp)*sizeof(real);
	cnt += foo(&dx, sizeof(real), 1,  fp)*sizeof(real);
	cnt += foo(&(xmin[0]), sizeof(real), NDIM, fp)*sizeof(real);

	cnt += my_location.load(fp);
	cnt += grid_ptr->load(fp);
	return cnt;
}

std::size_t node_server::save_me( FILE* fp ) const {
	auto foo = std::fwrite;
	std::size_t cnt = 0;

//	cnt += foo(&is_refined, sizeof(bool), 1, fp)*sizeof(bool);
	integer locality_id = hpx::get_locality_id();
	cnt += foo(&locality_id, sizeof(integer), 1, fp)*sizeof(integer);
	cnt += foo(&step_num, sizeof(integer), 1, fp)*sizeof(integer);
	cnt += foo(&current_time,sizeof(real), 1, fp)*sizeof(real);
	cnt += foo(&dx, sizeof(real), 1, fp)*sizeof(real);
	cnt += foo(&(xmin[0]), sizeof(real), NDIM, fp)*sizeof(real);

	cnt += my_location.save(fp);
	assert( grid_ptr != nullptr );
	cnt += grid_ptr->save(fp);
	return cnt;
}
void node_server::load(const std::string& filename, node_client root) {
	FILE* fp = fopen( filename.c_str(), "rb");
	integer cnt = 0;
	std::size_t bytes_expected;
	std::size_t bytes_read = 0;
	integer total_cnt;
	real omega;
	bytes_read += fread(&total_cnt, sizeof(integer), 1, fp)*sizeof(integer);
	bytes_read += fread(&bytes_expected, sizeof(std::size_t), 1, fp)*sizeof(std::size_t);
	bytes_read += fread(&omega, sizeof(real), 1, fp)*sizeof(real);
	grid::set_omega(omega);
	printf( "Loading %lli bytes and %i subgrids...\n", (long long int)(bytes_expected), int(total_cnt));
	for( integer i = 0; i != total_cnt; ++i ) {
		auto ns = std::make_shared<node_server>();
		node_location next_loc;
		bytes_read += next_loc.load(fp);
		std::size_t file_pos = bytes_read;
		integer dummy;
		bytes_read += ns->load_me(fp, dummy);
//		printf( "Loading at %s\n", next_loc.to_str().c_str());
		root.load_node(file_pos, filename,  next_loc, root.get_gid()).get();
		++cnt;
		printf( "%4.1f %% complete \r", real(100*file_pos)/real(bytes_expected));
		fflush(stdout);
	}
	printf( "%4.1f %% complete \r", 100.0);
	fflush(stdout);
	printf( "\n");
	std::size_t bytes_check;
	integer ngrid_check;
	bytes_read += fread(&ngrid_check, sizeof(integer), 1, fp)*sizeof(integer);
	bytes_read += fread(&bytes_check, sizeof(std::size_t), 1, fp)*sizeof(std::size_t);
	fclose(fp);
	if( bytes_expected != bytes_read || bytes_expected != bytes_check || ngrid_check != cnt ) {
		printf( "Checkpoint file corrupt\n");
		printf( "Bytes read        = %lli\n", (long long) bytes_read);
		printf( "Bytes expected    = %lli\n", (long long) bytes_expected);
		printf( "Bytes end         = %lli\n", (long long) bytes_check);
		printf( "subgrids expected = %i\n", (int) total_cnt);
		printf( "subgrids end      = %i\n", (int) ngrid_check);
		abort();
	} else {
		printf( "--------Loaded: %i subgrids, %lli bytes read----------\n",int(cnt), (long long)(bytes_read));
	}
}



hpx::id_type node_server::load_node(std::size_t filepos, const std::string& fname, const node_location& loc, const hpx::id_type& _me) {
	me = _me;
	hpx::future<hpx::id_type> rc;
	if( loc == my_location) {
		integer locality_id;
		FILE* fp = fopen( fname.c_str(), "rb");
		fseek(fp, filepos, SEEK_SET);
		load_me(fp, locality_id);
		fclose(fp);
		if( locality_id != hpx::get_locality_id()) {
			const auto localities = hpx::find_all_localities();
		//	printf( "Moving %s from %i to %i\n", loc.to_str().c_str(), hpx::get_locality_id(), int(locality_id));
			rc =  me.copy_to_locality(localities[locality_id]);
		} else {
			rc = hpx::make_ready_future(me.get_gid());
		}
	} else {
		rc = hpx::make_ready_future(me.get_gid());
		if( !is_refined) {
			if( my_location.level() >= MAX_LEVEL) {
				abort();
			}
			children.resize(NCHILD);
			for (integer ci = 0; ci != NCHILD; ++ci) {
				children[ci] = hpx::new_<node_server>(hpx::find_here(), my_location.get_child(ci), me, ZERO);
			}
			is_refined = true;
			const integer flags = (my_location.level() == 0) ? GRID_IS_ROOT : 0;
			grid_ptr = std::make_shared < grid > (dx, xmin, flags);
		}
		for( integer ci = 0; ci != NCHILD; ++ci) {
			auto cloc =  my_location.get_child(ci);
			if( cloc == loc || loc.is_child_of(cloc)) {
				children[ci] = children[ci].load_node(filepos, fname, loc, children[ci].get_gid());
			}
		}
	}
	clear_family();
	return rc.get();

}

