// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/util.hpp"

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

using receive_radiation_iteration_action_type = node_server::receive_radiation_iteration_action;
HPX_REGISTER_ACTION(receive_radiation_iteration_action_type);
using broadcast_radiation_iteration_action_type = node_server::broadcast_radiation_iteration_action;
HPX_REGISTER_ACTION(broadcast_radiation_iteration_action_type);

void node_server::receive_radiation_iteration(integer child, real value) {
    radiation_iteration_children[child].set_value(value);
}

void node_server::broadcast_radiation_iteration(real value) {
    radiation_iteration_global.set_value(value);
    if (is_refined) {
        for (auto const& child : children) {
            hpx::apply<broadcast_radiation_iteration_action>(child.get_unmanaged_gid(), value);
        }
    }
}

real node_server::radiation_iteration_max(real value) {
    // This ordered tree collective is entered by every node, including parents
    // without evolved cells. No OS-thread FPE guard may span an HPX suspension.
    if (!std::isfinite(value)) {
        value = std::numeric_limits<real>::max();
    }
    if (is_refined) {
        for (auto& child : radiation_iteration_children) {
            value = std::max(value, GET(child.get_future()));
        }
    }
    auto result = radiation_iteration_global.get_future();
    if (my_location.level() == 0) {
        broadcast_radiation_iteration(value);
    } else {
        hpx::apply<receive_radiation_iteration_action>(parent.get_unmanaged_gid(),
            my_location.get_child_index(), value);
    }
    return GET(result);
}

#endif
