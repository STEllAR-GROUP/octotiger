#ifndef OCTOTIGER_SET_LOCALITY_DATA_HPP
#define OCTOTIGER_SET_LOCALITY_DATA_HPP
#include <hpx/lcos/broadcast.hpp>

void set_locality_data(real omega, space_vector pivot);
HPX_DEFINE_PLAIN_ACTION(set_locality_data, set_locality_data_action);
HPX_REGISTER_ACTION_DECLARATION(set_locality_data_action, set_locality_data_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(set_locality_data_action);

#endif
