//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/io/silo.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/util.hpp"

#include <cstring>
#include <cmath>

constexpr int db_type<integer>::d;
constexpr int db_type<char>::d;
constexpr int db_type<real>::d;


double& silo_output_time() {
	static double t_;
	return t_;
}

double& silo_output_rotation_time() {
	static double t_;
	return t_;
}


int& silo_epoch() {
	static int t_ = 0;
	return t_;
}

namespace silo_restart {

metadata snapshot() {
    metadata result;
    result.layout = {NDIM, INX, H_NX, opts().n_fields, opts().n_species,
        integer(sizeof(real)), integer(SSP_MULTISTEP)};
    result.solver_integers = {refinement_freq(), opts().min_level, opts().max_level,
        integer(opts().eos), integer(opts().problem), integer(opts().gravity),
        integer(opts().hydro), integer(opts().amr_boundary_kernel_type),
        integer(opts().hydro_host_kernel_type), integer(opts().hydro_device_kernel_type),
        integer(opts().multipole_host_kernel_type), integer(opts().multipole_device_kernel_type),
        integer(opts().monopole_host_kernel_type), integer(opts().monopole_device_kernel_type),
        integer(opts().correct_am_grav), integer(opts().correct_am_hydro),
        integer(opts().inflow_bc), integer(opts().reflect_bc), opts().ipr_nr_maxiter,
        integer(opts().cdisc_detect), opts().experiment, integer(opts().unigrid),
        integer(opts().core_refine), opts().accretor_refine, opts().donor_refine,
        integer(opts().rotating_star_amr), opts().ngrids,
        integer(opts().optimize_local_communication)};
    result.solver_reals = {opts().cfl, opts().hard_dt, opts().theta,
        opts().rho_floor, opts().tau_floor, opts().dual_energy_sw1,
        opts().dual_energy_sw2, opts().driving_rate, opts().driving_time,
        opts().entropy_driving_rate, opts().entropy_driving_time,
        opts().ipr_eint_floor, opts().ipr_nr_tol, opts().grad_rho_refine};
    const auto& p = physcon();
    result.globals = {p.A, p.G, p.B, p.kb, p.sigma, p.c, p.mh, p.h,
        grid::get_omega(), grid::get_fgamma(), grid::get_scaling_factor(),
        opts().code_to_g, opts().code_to_cm, opts().code_to_s,
        opts().xscale, opts().refinement_floor};
    return result;
}

bool present(DBfile* db) {
    const int exists = DBInqVarExists(db, "restart_version");
    check(exists < 0 ? -1 : 0, "unable to query restart_version");
    if (exists == 0 && (DBInqVarExists(db, "restart_layout") > 0 ||
            DBInqVarExists(db, "restart_groups") > 0)) {
        throw std::runtime_error("Silo checkpoint is missing its restart_version marker");
    }
    return exists != 0;
}

void write_metadata(DBfile* db, const metadata& data) {
    write_array<integer>(db, "restart_version", {format_version});
    write_array(db, "restart_layout", data.layout);
    write_array(db, "restart_solver_integers", data.solver_integers);
    write_array(db, "restart_solver_reals", data.solver_reals);
    write_array(db, "restart_globals", data.globals);
}

void restore_metadata(DBfile* db) {
    if (read_array<integer>(db, "restart_version", 1)[0] != format_version) {
        throw std::runtime_error("Unsupported exact Silo checkpoint version");
    }
    if (opts().time_integrator != SSP_MULTISTEP) {
        throw std::runtime_error("Exact SSP checkpoint requires --time_integrator=SSP_MULTISTEP");
    }
    const auto current = snapshot();
    const auto layout = read_array<integer>(db, "restart_layout", current.layout.size());
    const auto integers = read_array<integer>(db, "restart_solver_integers",
        current.solver_integers.size());
    const auto reals = read_array<real>(db, "restart_solver_reals", current.solver_reals.size());
    const auto globals = read_array<real>(db, "restart_globals", current.globals.size());
    if (layout != current.layout || integers != current.solver_integers ||
            std::memcmp(reals.data(), current.solver_reals.data(), reals.size() * sizeof(real)) != 0) {
        throw std::runtime_error("Exact Silo checkpoint has incompatible dimensions or solver configuration");
    }
    for (const real value : globals) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Exact Silo checkpoint contains invalid global state");
        }
    }
    if (!(globals[9] > 1.0) || !(globals[10] > 0.0) ||
            !(globals[11] > 0.0) || !(globals[12] > 0.0) || !(globals[13] > 0.0)) {
        throw std::runtime_error("Exact Silo checkpoint contains invalid units or geometry");
    }
    auto& p = physcon();
    p.A = globals[0]; p.G = globals[1]; p.B = globals[2]; p.kb = globals[3];
    p.sigma = globals[4]; p.c = globals[5]; p.mh = globals[6]; p.h = globals[7];
    opts().omega = globals[8];
    grid::set_omega(globals[8], false);
    grid::set_fgamma(globals[9]);
    grid::set_scaling_factor(globals[10]);
    opts().code_to_g = globals[11];
    opts().code_to_cm = globals[12];
    opts().code_to_s = globals[13];
    opts().xscale = globals[14];
    opts().refinement_floor = globals[15];
}

std::vector<char> read_state(DBfile* db, node_location::node_id node_id) {
    const std::string name = "restart/" + oct_to_str(node_id) + "/state";
    const int count = DBGetVarLength(db, name.c_str());
    // Bound allocation before parsing an external archive.
    if (count <= 0 || count > (1 << 30)) {
        throw std::runtime_error("Silo checkpoint: invalid node archive length for " + name);
    }
    return read_array<char>(db, name, size_t(count));
}

} // namespace silo_restart
