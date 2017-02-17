#include "node_server.hpp"
#include "node_client.hpp"
#include "diagnostics.hpp"
#include "future.hpp"
#include "taylor.hpp"
#include "profiler.hpp"

#include <hpx/include/lcos.hpp>
#include <hpx/runtime/serialization/list.hpp>
#include <hpx/include/run_as.hpp>

#include "options.hpp"

typedef node_server::check_for_refinement_action check_for_refinement_action_type;
HPX_REGISTER_ACTION(check_for_refinement_action_type);

hpx::future<void> node_client::check_for_refinement() const {
    return hpx::async<typename node_server::check_for_refinement_action>(get_unmanaged_gid());
}

hpx::future<void> node_server::check_for_refinement() {
    bool rc = false;
    std::array<hpx::future<void>, NCHILD+1> futs;
    integer index = 0;
    if (is_refined) {
        for (auto& child : children) {
            futs[index++] = child.check_for_refinement();
        }
    }
    if (hydro_on) {
        all_hydro_bounds();
    }
    if (!rc) {
        rc = grid_ptr->refine_me(my_location.level());
    }
    if (rc) {
        if (refinement_flag++ == 0) {
            if (!parent.empty()) {
                futs[index++] =
                    parent.force_nodes_to_exist(my_location.get_neighbors());
            }
        }
    }
    return hpx::when_all(futs);
}

typedef node_server::copy_to_locality_action copy_to_locality_action_type;
HPX_REGISTER_ACTION(copy_to_locality_action_type);

hpx::future<hpx::id_type> node_client::copy_to_locality(const hpx::id_type& id) const {
    return hpx::async<typename node_server::copy_to_locality_action>(get_gid(), id);
}

hpx::future<hpx::id_type> node_server::copy_to_locality(const hpx::id_type& id) {

    std::vector<hpx::id_type> cids;
    if (is_refined) {
        cids.resize(NCHILD);
        for (auto& ci : geo::octant::full_set()) {
            cids[ci] = children[ci].get_gid();
        }
    }
    auto rc = hpx::new_<node_server
        >(id, my_location, step_num, is_refined, current_time, rotational_time,
            child_descendant_count, std::move(*grid_ptr), cids);
    clear_family();
    return rc;
}

extern options opts;

typedef node_server::diagnostics_action diagnostics_action_type;
HPX_REGISTER_ACTION(diagnostics_action_type);

hpx::future<diagnostics_t> node_client::diagnostics(
    const std::pair<space_vector, space_vector>& axis, const std::pair<real, real>& l1,
    real c1,
    real c2) const {
    return hpx::async<typename node_server::diagnostics_action>(get_unmanaged_gid(), axis, l1, c1,
        c2);
}

typedef node_server::compare_analytic_action compare_analytic_action_type;
HPX_REGISTER_ACTION(compare_analytic_action_type);

hpx::future<analytic_t> node_client::compare_analytic() const {
    return hpx::async<typename node_server::compare_analytic_action>(get_unmanaged_gid());
}

analytic_t node_server::compare_analytic() {
    analytic_t a;
    if (!is_refined) {
        a = grid_ptr->compute_analytic(current_time);
    } else {
        std::array<hpx::future<analytic_t>, NCHILD> futs;
        integer index = 0;
        for (integer i = 0; i != NCHILD; ++i) {
            futs[index++] = children[i].compare_analytic();
        }
        for (integer i = 0; i != NCHILD; ++i) {
            a += futs[i].get();
        }
    }
    if (my_location.level() == 0) {
        printf("L1, L2\n");
        for (integer field = 0; field != NF; ++field) {
            if (a.l1a[field] > 0.0) {
                printf("%16s %e %e\n", grid::field_names[field],
                    a.l1[field] / a.l1a[field], std::sqrt(a.l2[field] / a.l2a[field]));
            }
        }
    }
    return a;
}

diagnostics_t node_server::diagnostics() const {
    auto axis = grid_ptr->find_axis();
    auto loc = line_of_centers(axis);
    real this_omega = grid::get_omega();
    std::pair<real, real> rho1, rho2, l1, l2, l3;
    real phi_1, phi_2;
    line_of_centers_analyze(loc, this_omega, rho1, rho2, l1, l2, l3, phi_1, phi_2);
    //if( rho1.first > rho2.first ) {
    //	for( integer d = 0; d != NDIM; ++d ) {
    //		//printf( "Flipping axis\n" );
    //		axis.first[d] = -axis.first[d];
    //		loc = line_of_centers(axis);
    //		line_of_centers_analyze(loc, this_omega, rho1, rho2, l1, phi_1, phi_2);
    //	}
//	}
    auto diags = diagnostics(axis, l1, rho1.first, rho2.first);

    diags.z_moment -= diags.grid_sum[rho_i]
        * (std::pow(diags.grid_com[XDIM], 2) + std::pow(diags.grid_com[YDIM], 2));
    diags.primary_z_moment -= diags.primary_sum[rho_i]
        * (std::pow(diags.primary_com[XDIM], 2) + std::pow(diags.primary_com[YDIM], 2));
    diags.secondary_z_moment -=
        diags.secondary_sum[rho_i]
            * (std::pow(diags.secondary_com[XDIM], 2)
                + std::pow(diags.secondary_com[YDIM], 2));
    if (diags.primary_sum[rho_i] < diags.secondary_sum[rho_i]) {
        std::swap(diags.primary_sum, diags.secondary_sum);
        std::swap(diags.primary_com, diags.secondary_com);
        std::swap(diags.primary_com_dot, diags.secondary_com_dot);
        std::swap(rho1, rho2);
        std::swap(phi_1, phi_2);
        std::swap(diags.primary_z_moment, diags.secondary_z_moment);
    }

    if (opts.problem != SOLID_SPHERE) {
        // run output on separate thread
        hpx::threads::run_as_os_thread([&]()
        {
            FILE* fp = fopen("diag.dat", "at");
            fprintf(fp, "%23.16e ", double(current_time));
            for (integer f = 0; f != NF; ++f) {
                fprintf(fp, "%23.16e ", double(diags.grid_sum[f] + diags.outflow_sum[f]));
                fprintf(fp, "%23.16e ", double(diags.outflow_sum[f]));
            }
            for (integer f = 0; f != NDIM; ++f) {
                fprintf(fp, "%23.16e ", double(diags.l_sum[f]));
            }
            fprintf(fp, "\n");
            fclose(fp);
        }).get();

        real a = 0.0;
        for (integer d = 0; d != NDIM; ++d) {
            a += std::pow(diags.primary_com[d] - diags.secondary_com[d], 2);
        }
        a = std::sqrt(a);
        real j1 = 0.0;
        real j2 = 0.0;
        real m1 = diags.primary_sum[rho_i];
        real m2 = diags.secondary_sum[rho_i];
        j1 -= diags.primary_com_dot[XDIM]
            * (diags.primary_com[YDIM] - diags.grid_com[YDIM]) * m1;
        j1 += diags.primary_com_dot[YDIM]
            * (diags.primary_com[XDIM] - diags.grid_com[XDIM]) * m1;
        j2 -= diags.secondary_com_dot[XDIM]
            * (diags.secondary_com[YDIM] - diags.grid_com[YDIM]) * m2;
        j2 += diags.secondary_com_dot[YDIM]
            * (diags.secondary_com[XDIM] - diags.grid_com[XDIM]) * m2;
        const real jorb = j1 + j2;
        j1 = diags.primary_sum[zz_i] - j1;
        j2 = diags.secondary_sum[zz_i] - j2;

        // run output on separate thread
        hpx::threads::run_as_os_thread([&]()
        {
            FILE* fp = fopen("binary.dat", "at");
            fprintf(fp, "%15.8e ", double(current_time));
            fprintf(fp, "%15.8e ", double(m1));
            fprintf(fp, "%15.8e ", double(m2));
            fprintf(fp, "%15.8e ", double(this_omega));
            fprintf(fp, "%15.8e ", double(a));
            fprintf(fp, "%15.8e ", double(rho1.second));
            fprintf(fp, "%15.8e ", double(rho2.second));
            fprintf(fp, "%15.8e ", double(jorb));
            fprintf(fp, "%15.8e ", double(j1));
            fprintf(fp, "%15.8e ", double(j2));
            fprintf(fp, "%15.8e ", double(diags.z_moment));
            fprintf(fp, "%15.8e ", double(diags.primary_z_moment));
            fprintf(fp, "%15.8e ", double(diags.secondary_z_moment));
            fprintf(fp, "\n");
            fclose(fp);

            fp = fopen("minmax.dat", "at");
            fprintf(fp, "%23.16e ", double(current_time));
            for (integer f = 0; f != NF; ++f) {
                fprintf(fp, "%23.16e ", double(diags.field_min[f]));
                fprintf(fp, "%23.16e ", double(diags.field_max[f]));
            }
            fprintf(fp, "\n");
            fclose(fp);

            fp = fopen("com.dat", "at");
            fprintf(fp, "%23.16e ", double(current_time));
            for (integer d = 0; d != NDIM; ++d) {
                fprintf(fp, "%23.16e ", double(diags.primary_com[d]));
            }
            for (integer d = 0; d != NDIM; ++d) {
                fprintf(fp, "%23.16e ", double(diags.secondary_com[d]));
            }
            for (integer d = 0; d != NDIM; ++d) {
                fprintf(fp, "%23.16e ", double(diags.grid_com[d]));
            }
            fprintf(fp, "\n");
            fclose(fp);
        }).get();
    } else {
        printf("L1\n");
        printf("Gravity Phi Error - %e\n", (diags.l1_error[0] / diags.l1_error[4]));
        printf("Gravity gx Error - %e\n", (diags.l1_error[1] / diags.l1_error[5]));
        printf("Gravity gy Error - %e\n", (diags.l1_error[2] / diags.l1_error[6]));
        printf("Gravity gz Error - %e\n", (diags.l1_error[3] / diags.l1_error[7]));
        printf("L2\n");
        printf("Gravity Phi Error - %e\n",
            std::sqrt(diags.l2_error[0] / diags.l2_error[4]));
        printf("Gravity gx Error - %e\n",
            std::sqrt(diags.l2_error[1] / diags.l2_error[5]));
        printf("Gravity gy Error - %e\n",
            std::sqrt(diags.l2_error[2] / diags.l2_error[6]));
        printf("Gravity gz Error - %e\n",
            std::sqrt(diags.l2_error[3] / diags.l2_error[7]));
        printf("Total Mass = %e\n", diags.grid_sum[rho_i]);
        for (integer d = 0; d != NDIM; ++d) {
            printf("%e %e\n", diags.gforce_sum[d], diags.gtorque_sum[d]);
        }
    }

    return diags;
}

diagnostics_t node_server::diagnostics(const std::pair<space_vector, space_vector>& axis,
    const std::pair<real, real>& l1, real c1, real c2) const {

    diagnostics_t sums;
    if (is_refined) {
        std::array<hpx::future<diagnostics_t>, NCHILD> futs;
        integer index = 0;
        for (integer ci = 0; ci != NCHILD; ++ci) {
            futs[index++] = children[ci].diagnostics(axis, l1, c1, c2);
        }
        auto child_sums = hpx::util::unwrapped(futs);
        sums = std::accumulate(child_sums.begin(), child_sums.end(), sums);
    } else {

        sums.primary_sum = grid_ptr->conserved_sums(sums.primary_com,
            sums.primary_com_dot, axis, l1, +1);
        sums.secondary_sum = grid_ptr->conserved_sums(sums.secondary_com,
            sums.secondary_com_dot, axis, l1, -1);
        sums.primary_z_moment = grid_ptr->z_moments(axis, l1, +1);
        sums.secondary_z_moment = grid_ptr->z_moments(axis, l1, -1);
        sums.grid_sum = grid_ptr->conserved_sums(sums.grid_com, sums.grid_com_dot, axis,
            l1, 0);
        sums.outflow_sum = grid_ptr->conserved_outflows();
        sums.l_sum = grid_ptr->l_sums();
        auto tmp = grid_ptr->field_range();
        sums.field_min = std::move(tmp.first);
        sums.field_max = std::move(tmp.second);
        sums.gforce_sum = grid_ptr->gforce_sum(false);
        sums.gtorque_sum = grid_ptr->gforce_sum(true);
        auto tmp2 = grid_ptr->diagnostic_error();
        sums.l1_error = tmp2.first;
        sums.l2_error = tmp2.second;
        auto vols = grid_ptr->frac_volumes();
        sums.roche_vol1 = grid_ptr->roche_volume(axis, l1, std::min(c1, c2), false);
        sums.roche_vol2 = grid_ptr->roche_volume(axis, l1, std::max(c1, c2), true);
        sums.primary_volume = vols[spc_ac_i - spc_i] + vols[spc_ae_i - spc_i];
        sums.secondary_volume = vols[spc_dc_i - spc_i] + vols[spc_de_i - spc_i];
        sums.z_moment = grid_ptr->z_moments(axis, l1, 0);
    }

    return sums;
}

diagnostics_t::diagnostics_t() :
    primary_sum(NF, ZERO), secondary_sum(NF, ZERO), grid_sum(NF, ZERO), outflow_sum(NF,
        ZERO), l_sum(NDIM, ZERO), field_max(NF,
        -std::numeric_limits < real > ::max()), field_min(NF,
        +std::numeric_limits < real > ::max()), gforce_sum(NDIM, ZERO), gtorque_sum(NDIM,
        ZERO) {
    for (integer d = 0; d != NDIM; ++d) {
        primary_z_moment = secondary_z_moment = z_moment = 0.0;
        roche_vol1 = roche_vol2 = primary_volume = secondary_volume = 0.0;
        primary_com[d] = secondary_com[d] = grid_com[d] = 0.0;
        primary_com_dot[d] = secondary_com_dot[d] = grid_com_dot[d] = 0.0;
    }
}

diagnostics_t& diagnostics_t::operator+=(const diagnostics_t& other) {
    primary_z_moment += other.primary_z_moment;
    secondary_z_moment += other.secondary_z_moment;
    z_moment += other.z_moment;
    for (integer d = 0; d != NDIM; ++d) {
        primary_com[d] *= primary_sum[rho_i];
        secondary_com[d] *= secondary_sum[rho_i];
        grid_com[d] *= grid_sum[rho_i];
        primary_com_dot[d] *= primary_sum[rho_i];
        secondary_com_dot[d] *= secondary_sum[rho_i];
        grid_com_dot[d] *= grid_sum[rho_i];
    }
    for (integer f = 0; f != NF; ++f) {
        grid_sum[f] += other.grid_sum[f];
        primary_sum[f] += other.primary_sum[f];
        secondary_sum[f] += other.secondary_sum[f];
        outflow_sum[f] += other.outflow_sum[f];
        field_max[f] = std::max(field_max[f], other.field_max[f]);
        field_min[f] = std::min(field_min[f], other.field_min[f]);
    }
    for (integer d = 0; d != NDIM; ++d) {
        l_sum[d] += other.l_sum[d];
        gforce_sum[d] += other.gforce_sum[d];
        gtorque_sum[d] += other.gtorque_sum[d];
    }
    if (l1_error.size() < other.l1_error.size()) {
        l1_error.resize(other.l1_error.size(), ZERO);
        l2_error.resize(other.l2_error.size(), ZERO);
    }
    for (std::size_t i = 0; i != l1_error.size(); ++i) {
        l1_error[i] += other.l1_error[i];
    }
    for (std::size_t i = 0; i != l1_error.size(); ++i) {
        l2_error[i] += other.l2_error[i];
    }
    for (integer d = 0; d != NDIM; ++d) {
        primary_com[d] += other.primary_com[d] * other.primary_sum[rho_i];
        secondary_com[d] += other.secondary_com[d] * other.secondary_sum[rho_i];
        grid_com[d] += other.grid_com[d] * other.grid_sum[rho_i];
        primary_com_dot[d] += other.primary_com_dot[d] * other.primary_sum[rho_i];
        secondary_com_dot[d] += other.secondary_com_dot[d] * other.secondary_sum[rho_i];
        grid_com_dot[d] += other.grid_com_dot[d] * other.grid_sum[rho_i];
    }
    for (integer d = 0; d != NDIM; ++d) {
        if (primary_sum[rho_i] > 0.0) {
            primary_com[d] /= primary_sum[rho_i];
            primary_com_dot[d] /= primary_sum[rho_i];
        }
        if (secondary_sum[rho_i] > 0.0) {
            secondary_com[d] /= secondary_sum[rho_i];
            secondary_com_dot[d] /= secondary_sum[rho_i];
        }
        grid_com[d] /= grid_sum[rho_i];
        grid_com_dot[d] /= grid_sum[rho_i];
    }
    roche_vol1 += other.roche_vol1;
    roche_vol2 += other.roche_vol2;
    primary_volume += other.primary_volume;
    secondary_volume += other.secondary_volume;
    return *this;
}

typedef node_server::force_nodes_to_exist_action force_nodes_to_exist_action_type;
HPX_REGISTER_ACTION(force_nodes_to_exist_action_type);

hpx::future<void> node_client::force_nodes_to_exist(
    std::vector<node_location>&& locs) const {
    return hpx::async<typename node_server::force_nodes_to_exist_action>(get_unmanaged_gid(),
        std::move(locs));
}

void node_server::force_nodes_to_exist(std::vector<node_location>&& locs) {
    std::vector<hpx::future<void>> futs;
    std::vector<node_location> parent_list;
    std::vector<std::vector<node_location>> child_lists(NCHILD);

    futs.reserve(geo::octant::count() + 2);
    parent_list.reserve(locs.size());

    integer index = 0;
    for (auto& loc : locs) {
        assert(loc != my_location);
        if (loc.is_child_of(my_location)) {
            if (refinement_flag++ == 0 && !parent.empty()) {
                futs.push_back(
                    parent.force_nodes_to_exist(my_location.get_neighbors()));
            }
            if (is_refined) {
                for (auto& ci : geo::octant::full_set()) {
                    if (loc.is_child_of(my_location.get_child(ci))) {
                        if (child_lists[ci].empty())
                        {
                            child_lists[ci].reserve(locs.size());
                        }
                        child_lists[ci].push_back(loc);
                        break;
                    }
                }
            }
        } else {
            assert(!parent.empty());
            parent_list.push_back(loc);
        }
    }
    for (auto& ci : geo::octant::full_set()) {
        if (is_refined && child_lists[ci].size()) {
            futs.push_back(children[ci].force_nodes_to_exist(std::move(child_lists[ci])));
        }
    }
    if (parent_list.size()) {
        futs.push_back(parent.force_nodes_to_exist(std::move(parent_list)));
    }

    wait_all_and_propagate_exceptions(futs);
}

typedef node_server::form_tree_action form_tree_action_type;
HPX_REGISTER_ACTION(form_tree_action_type);

hpx::future<void> node_client::form_tree(const hpx::id_type& id1, const hpx::id_type& id2,
    const std::vector<hpx::id_type>& ids) {
    return hpx::async<typename node_server::form_tree_action>(get_unmanaged_gid(), id1, id2,
        std::move(ids));
}

hpx::future<void> node_server::form_tree(const hpx::id_type& self_gid, const hpx::id_type& parent_gid,
    const std::vector<hpx::id_type>& neighbor_gids) {
    for (auto& dir : geo::direction::full_set()) {
        neighbors[dir] = neighbor_gids[dir];
    }
    me = self_gid;
    parent = parent_gid;
    if (is_refined) {
        std::array<hpx::future<void>, 2*2*2> cfuts;
        integer index = 0;

        amr_flags.resize(NCHILD);
        for (integer cx = 0; cx != 2; ++cx) {
            for (integer cy = 0; cy != 2; ++cy) {
                for (integer cz = 0; cz != 2; ++cz) {
                    std::vector<hpx::future<hpx::id_type>> child_neighbors_f(
                        geo::direction::count());
                    std::vector<hpx::id_type> child_neighbors(geo::direction::count());
                    const integer ci = cx + 2 * cy + 4 * cz;
                    for (integer dx = -1; dx != 2; ++dx) {
                        for (integer dy = -1; dy != 2; ++dy) {
                            for (integer dz = -1; dz != 2; ++dz) {
                                if (!(dx == 0 && dy == 0 && dz == 0)) {
                                    const integer x = cx + dx + 2;
                                    const integer y = cy + dy + 2;
                                    const integer z = cz + dz + 2;
                                    geo::direction i;
                                    i.set(dx, dy, dz);
                                    auto& ref = child_neighbors_f[i];
                                    auto other_child = (x % 2) + 2 * (y % 2)
                                        + 4 * (z % 2);
                                    if (x / 2 == 1 && y / 2 == 1 && z / 2 == 1) {
                                        ref = hpx::make_ready_future<hpx::id_type>(
                                            children[other_child].get_gid());
                                    } else {
                                        geo::direction dir = geo::direction(
                                            (x / 2) + NDIM * ((y / 2) + NDIM * (z / 2)));
                                        ref = neighbors[dir].get_child_client(
                                            other_child);
                                    }
                                }
                            }
                        }
                    }

                    for (auto& dir : geo::direction::full_set()) {
                        child_neighbors[dir] = child_neighbors_f[dir].get();
                        if (child_neighbors[dir] == hpx::invalid_id) {
                            amr_flags[ci][dir] = true;
                        } else {
                            amr_flags[ci][dir] = false;
                        }
                    }
                    cfuts[index++] =
                        children[ci].form_tree(children[ci].get_gid(), me.get_gid(),
                            std::move(child_neighbors));
                }
            }
        }
        return hpx::when_all(cfuts);
    } else {
        std::vector < hpx::future<std::vector<hpx::id_type>>>nfuts(NFACE);
        for (auto& f : geo::face::full_set()) {
        	const auto& neighbor = neighbors[f.to_direction()];
            if (!neighbor.empty())
            {
                nfuts[f] = neighbor.get_nieces(me.get_gid(), f ^ 1);
            }
            else
            {
                nfuts[f] = hpx::make_ready_future(std::vector<hpx::id_type>());
            }
        }
        return hpx::dataflow(
            [this](std::vector<hpx::future<std::vector<hpx::id_type>>> nfuts)
            {
                for (auto& f : geo::face::full_set()) {
                    auto ids = nfuts[f].get();
                    nieces[f].resize(ids.size());
                    for (std::size_t i = 0; i != ids.size(); ++i) {
                        nieces[f][i] = ids[i];
                    }
                }
            }
          , std::move(nfuts)
        );
    }
}

typedef node_server::get_child_client_action get_child_client_action_type;
HPX_REGISTER_ACTION(get_child_client_action_type);

hpx::future<hpx::id_type> node_client::get_child_client(const geo::octant& ci) {
    if (get_gid() != hpx::invalid_id) {
        return hpx::async<typename node_server::get_child_client_action>(get_unmanaged_gid(), ci);
    } else {
        auto tmp = hpx::invalid_id;
        return hpx::make_ready_future<hpx::id_type>(std::move(tmp));
    }
}

hpx::id_type node_server::get_child_client(const geo::octant& ci) {
    if (is_refined) {
        return children[ci].get_gid();
    } else {
        return hpx::invalid_id;
    }
}

typedef node_server::get_nieces_action get_nieces_action_type;
HPX_REGISTER_ACTION(get_nieces_action_type);

hpx::future<std::vector<hpx::id_type>> node_client::get_nieces(const hpx::id_type& aunt,
    const geo::face& f) const {
    return hpx::async<typename node_server::get_nieces_action>(get_unmanaged_gid(), aunt, f);
}

std::vector<hpx::id_type> node_server::get_nieces(const hpx::id_type& aunt,
    const geo::face& face) const {
    std::vector<hpx::id_type> nieces;
    if (is_refined) {
        std::array<hpx::future<void>, geo::octant::count()> futs;
        nieces.reserve(geo::octant::count());
        integer index = 0;
        for (auto const& ci : geo::octant::face_subset(face)) {
            nieces.push_back(children[ci].get_gid());
            futs[index++] = children[ci].set_aunt(aunt, face);
        }
        wait_all_and_propagate_exceptions(futs);
    }
    return nieces;
}

typedef node_server::get_ptr_action get_ptr_action_type;
HPX_REGISTER_ACTION(get_ptr_action_type);

std::uintptr_t node_server::get_ptr() {
    return reinterpret_cast<std::uintptr_t>(this);
}

hpx::future<node_server*> node_client::get_ptr() const {
    return hpx::async<typename node_server::get_ptr_action>(get_unmanaged_gid()).then(
        [](hpx::future<std::uintptr_t>&& fut) {
            return reinterpret_cast<node_server*>(fut.get());
        });
}
