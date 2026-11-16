// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/serialization/array.hpp>
#include <hpx/serialization/vector.hpp>

#include "octotiger/light_curve.hpp"
#include "octotiger/node_client.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"

#if !defined(HPX_COMPUTE_DEVICE_CODE)
namespace light_curve_output {
using octotiger::light_curve::Column;
using octotiger::light_curve::Hit;

namespace {
// A diagnostic runs at a completed timestep, with no concurrent regrid. Only
// leaf actions on this locality append here; the root receives scalar totals.
std::mutex collector_mutex;
std::array<std::vector<Hit>, 2> collector;

double checked(double value) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error("Light-curve output overflow or invalid physical units");
    }
    return value;
}

std::ofstream open_file(std::filesystem::path const& path) {
    std::ofstream stream;
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.open(path);
    stream << std::setprecision(17);
    return stream;
}

struct Cell {
    double temperature = 0.0;
    std::vector<std::array<double, 4>> patches;
};
} // namespace

struct Summary {
    // Code-unit T^4 area and projected area, first +z and then -z.
    std::array<double, 4> sums{};
    std::uint32_t locality = 0;

    template<class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & sums;
        archive & locality;
    }
};

void reset() {
    std::lock_guard<std::mutex> lock(collector_mutex);
    for (auto& hits : collector) {
        hits.clear();
    }
}

void record(std::vector<Hit> hits, bool from_above) {
    std::lock_guard<std::mutex> lock(collector_mutex);
    auto& destination = collector[from_above ? 0 : 1];
    destination.insert(destination.end(), std::make_move_iterator(hits.begin()),
        std::make_move_iterator(hits.end()));
}

Summary save(std::string const& directory) {
    std::array<std::vector<Hit>, 2> hits;
    {
        std::lock_guard<std::mutex> lock(collector_mutex);
        hits.swap(collector);
    }
    Summary summary;
    summary.locality = hpx::get_locality_id();
    auto output = open_file(std::filesystem::path(directory) /
        ("locality_" + std::to_string(summary.locality) + ".json"));
    output << "{\"locality\":" << summary.locality << ",\"cells\":[\n";
    double const length = opts().code_to_cm;
    bool first_cell = true;
    for (size_t side = 0; side < 2; ++side) {
        std::map<std::array<double, 4>, Cell> cells;
        for (auto const& hit : hits[side]) {
            auto& cell = cells[{hit.x, hit.y, hit.z, hit.width}];
            cell.temperature = hit.temperature;
            cell.patches.push_back({hit.x_lower, hit.x_upper, hit.y_lower, hit.y_upper});
        }
        long double area = 0.0L;
        long double temperature4_area = 0.0L;
        for (auto& entry : cells) {
            auto const& position = entry.first;
            auto& cell = entry.second;
            std::sort(cell.patches.begin(), cell.patches.end());
            long double visible_area = 0.0L;
            for (auto const& patch : cell.patches) {
                visible_area += static_cast<long double>(patch[1] - patch[0]) *
                    (patch[3] - patch[2]);
            }
            area += visible_area;
            long double const temperature2 = static_cast<long double>(cell.temperature) * cell.temperature;
            temperature4_area += temperature2 * temperature2 * visible_area;
            if (!first_cell) {
                output << ",\n";
            }
            first_cell = false;
            output << "{\"view\":\"" << (side == 0 ? "plus_z" : "minus_z")
                   << "\",\"x\":" << position[0] * length
                   << ",\"y\":" << position[1] * length
                   << ",\"z\":" << position[2] * length
                   << ",\"width\":" << checked(position[3] * length)
                   << ",\"temperature\":" << checked(cell.temperature)
                   << ",\"visible_area\":" << checked(double(visible_area * length * length))
                   << ",\"patches\":[";
            bool first_patch = true;
            for (auto const& patch : cell.patches) {
                if (!first_patch) {
                    output << ',';
                }
                first_patch = false;
                output << '[' << patch[0] * length << ',' << patch[1] * length << ','
                       << patch[2] * length << ',' << patch[3] * length << ']';
            }
            output << "]}";
        }
        summary.sums[2 * side] = checked(double(temperature4_area));
        summary.sums[2 * side + 1] = checked(double(area));
    }
    output << "\n]}\n";
    output.close();
    return summary;
}
} // namespace light_curve_output

HPX_PLAIN_ACTION(light_curve_output::reset, light_curve_reset_action);
HPX_PLAIN_ACTION(light_curve_output::save, light_curve_save_action);
using trace_light_curve_action_type = node_server::trace_light_curve_columns_action;
HPX_REGISTER_ACTION(trace_light_curve_action_type);

future<std::vector<octotiger::light_curve::Column>> node_client::trace_light_curve_columns(
    std::vector<octotiger::light_curve::Column> columns, bool from_above) const {
    return hpx::async<node_server::trace_light_curve_columns_action>(
        get_unmanaged_gid(), std::move(columns), from_above);
}

std::vector<octotiger::light_curve::Column> node_server::trace_light_curve_columns(
    std::vector<octotiger::light_curve::Column> columns, bool from_above) const {
    using octotiger::light_curve::Column;
    if (columns.empty()) {
        return {};
    }
    if (!is_refined) {
        auto result = grid_ptr->trace_light_curve_columns(std::move(columns), from_above);
        light_curve_output::record(std::move(result.hits), from_above);
        return std::move(result.columns);
    }
    double const scale = grid::get_scaling_factor();
    double const child_width = std::ldexp(scale, -my_location.level());
    double const lower_x = scale * my_location.x_location(XDIM);
    double const lower_y = scale * my_location.x_location(YDIM);
    std::array<future<std::vector<Column>>, NCHILD / 2> near;
    std::array<future<std::vector<Column>>, NCHILD / 2> far;
    for (int quadrant = 0; quadrant < NCHILD / 2; ++quadrant) {
        double const x = lower_x + (quadrant & 1) * child_width;
        double const y = lower_y + ((quadrant >> 1) & 1) * child_width;
        std::vector<Column> incoming;
        for (auto const& column : columns) {
            Column clipped;
            if (octotiger::light_curve::intersect(column, x, x + child_width,
                    y, y + child_width, clipped)) {
                incoming.push_back(clipped);
            }
        }
        if (!incoming.empty()) {
            near[quadrant] = children[quadrant + (from_above ? NCHILD / 2 : 0)]
                .trace_light_curve_columns(std::move(incoming), from_above);
        }
    }
    for (int quadrant = 0; quadrant < NCHILD / 2; ++quadrant) {
        if (near[quadrant].valid()) {
            auto pending = near[quadrant].get();
            if (!pending.empty()) {
                far[quadrant] = children[quadrant + (from_above ? 0 : NCHILD / 2)]
                    .trace_light_curve_columns(std::move(pending), from_above);
            }
        }
    }
    std::vector<Column> unfinished;
    for (auto& result : far) {
        if (result.valid()) {
            auto pending = result.get();
            unfinished.insert(unfinished.end(), std::make_move_iterator(pending.begin()),
                std::make_move_iterator(pending.end()));
        }
    }
    return unfinished;
}

void node_server::compute_light_curve() {
    using light_curve_output::checked;
    using light_curve_output::open_file;
    using octotiger::light_curve::Column;
    if (my_location.level() != 0) {
        throw std::logic_error("Only the root may start a light-curve diagnostic");
    }
    double const scale = grid::get_scaling_factor();
    double const length = opts().code_to_cm;
    double const mass = opts().code_to_g;
    double const time_unit = opts().code_to_s;
    for (double const value : {scale, length, mass, time_unit, double(physcon().sigma)}) {
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw std::runtime_error("Light curves require finite positive domain and physical units");
        }
    }
    double const area_unit = checked(length * length);
    double const luminosity_unit = checked(mass * area_unit / (time_unit * time_unit * time_unit));
    double const physical_time = checked(current_time * time_unit);
    double const half_width = checked(scale * length);
    auto const table = std::filesystem::absolute(
        std::filesystem::path(opts().data_dir) / opts().light_curve_output);
    std::filesystem::create_directories(table.parent_path());
    auto const mesh_root = std::filesystem::path(table.string() + ".mesh");
    std::filesystem::create_directories(mesh_root);
    static size_t next_sample = 0;
    size_t sample;
    std::filesystem::path directory;
    do {
        sample = next_sample++;
        directory = mesh_root / ("sample_" + std::to_string(sample));
    } while (!std::filesystem::create_directory(directory));

    std::vector<future<void>> resets;
    for (auto const& locality : options::all_localities) {
        resets.push_back(hpx::async<light_curve_reset_action>(locality));
    }
    for (auto& reset : resets) {
        reset.get();
    }
    Column camera;
    camera.x_lower = camera.y_lower = -scale;
    camera.x_upper = camera.y_upper = scale;
    auto above = me.trace_light_curve_columns({camera}, true);
    auto below = me.trace_light_curve_columns({camera}, false);
    std::array<double, 2> unresolved{};
    auto sum_unresolved = [](std::vector<Column> const& columns) {
        long double area = 0.0L;
        for (auto const& column : columns) {
            area += static_cast<long double>(column.x_upper - column.x_lower) *
                (column.y_upper - column.y_lower);
        }
        return checked(double(area));
    };
    unresolved[0] = sum_unresolved(above.get());
    unresolved[1] = sum_unresolved(below.get());
    std::vector<future<light_curve_output::Summary>> saves;
    for (auto const& locality : options::all_localities) {
        saves.push_back(hpx::async<light_curve_save_action>(locality, directory.string()));
    }
    std::array<long double, 4> sums{};
    std::vector<std::uint32_t> localities;
    for (auto& save : saves) {
        auto const part = save.get();
        localities.push_back(part.locality);
        for (size_t i = 0; i < sums.size(); ++i) {
            sums[i] += part.sums[i];
        }
    }
    std::sort(localities.begin(), localities.end());
    std::array<double, 2> area{};
    std::array<double, 2> luminosity{};
    for (size_t side = 0; side < 2; ++side) {
        double const total_area = checked(4.0 * scale * scale);
        if (std::abs(double(sums[2 * side + 1]) + unresolved[side] - total_area) > 1.0e-10 * total_area) {
            throw std::runtime_error("Light-curve traversal lost or duplicated projected area");
        }
        area[side] = checked(double(sums[2 * side + 1] * area_unit));
        luminosity[side] = checked(double(4.0L * physcon().sigma * sums[2 * side] * luminosity_unit));
        unresolved[side] = checked(unresolved[side] * area_unit);
    }
    // The manifest and table row are published only after every locality's
    // complete mesh has been written successfully. Fresh sample directories
    // retain earlier results across restarts, including repeated step numbers.
    auto manifest = open_file(directory / "manifest.json");
    manifest << "{\n\"sample\":" << sample << ",\"time\":" << physical_time
             << ",\"step\":" << step_num
             << ",\"liso_plus\":" << luminosity[0] << ",\"liso_minus\":" << luminosity[1]
             << ",\"area_plus\":" << area[0] << ",\"area_minus\":" << area[1]
             << ",\"unresolved_area_plus\":" << unresolved[0]
             << ",\"unresolved_area_minus\":" << unresolved[1]
             << ",\"domain_half_width\":" << half_width
             << ",\n\"opacity_source\":\"" << opts().light_curve_opacity_model << "\","
             << "\n\"fld_opacity_model\":\"" << opts().fld_opacity_model << "\","
             << "\n\"fld_material_model\":\"" << opts().fld_material_model << "\","
             << "\n\"fld_unit_system\":\"" << opts().fld_unit_system << "\","
             << "\n\"fld_opacity_parameters\":{\"planck_absorption\":" << opts().fld_absorption
             << ",\"scattering_mass_opacity\":" << opts().fld_scattering
             << ",\"planck_density_exponent\":" << opts().fld_density_exponent
             << ",\"planck_temperature_exponent\":" << opts().fld_temperature_exponent
             << ",\"rosseland_total\":" << opts().fld_rosseland
             << ",\"rosseland_density_exponent\":" << opts().fld_rosseland_density_exponent
             << ",\"rosseland_temperature_exponent\":" << opts().fld_rosseland_temperature_exponent
             << ",\"opacity_temperature_cutoff\":" << opts().fld_opacity_temperature_floor
             << ",\"gaunt\":" << opts().fld_gaunt << "},"
             << "\n\"opacity\":{\"absorption\":" << opts().light_curve_absorption
             << ",\"scattering\":" << opts().light_curve_scattering
             << ",\"density_exponent\":" << opts().light_curve_density_exponent
             << ",\"temperature_exponent\":" << opts().light_curve_temperature_exponent << "},"
             << "\n\"units\":{\"position\":\"cm\",\"temperature\":\"K\",\"area\":\"cm^2\","
             << "\"time\":\"s\",\"luminosity\":\"erg/s\",\"opacity_parameters\":\"code units\","
             << "\"code_to_cm\":" << length << ",\"code_to_g\":" << mass
             << ",\"code_to_s\":" << time_unit << "},"
             << "\n\"coordinates\":\"rotating grid axes; same x,y axes for both views\","
             << "\n\"emission_model\":\"blackbody at first tau_LOS=1 cell; Liso=4 sigma integral T^4 dA\","
             << "\n\"files\":[";
    for (size_t i = 0; i < localities.size(); ++i) {
        manifest << (i ? "," : "") << "\"locality_" << localities[i] << ".json\"";
    }
    manifest << "]\n}\n";
    manifest.close();
    bool const header = !std::filesystem::exists(table) || std::filesystem::file_size(table) == 0;
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(table, std::ios::app);
    output << std::setprecision(17);
    if (header) {
        output << "# time_s step liso_plus_erg_s liso_minus_erg_s area_plus_cm2 area_minus_cm2 "
                  "unresolved_area_plus_cm2 unresolved_area_minus_cm2 sample\n";
    }
    output << physical_time << ' ' << step_num << ' ' << luminosity[0] << ' ' << luminosity[1]
           << ' ' << area[0] << ' ' << area[1] << ' ' << unresolved[0] << ' ' << unresolved[1]
           << ' ' << sample << '\n';
    output.close();
    std::cout << std::setprecision(17) << "LIGHT_CURVE time=" << physical_time << " step=" << step_num
              << " liso_plus=" << luminosity[0] << " liso_minus=" << luminosity[1]
              << " area_plus=" << area[0] << " area_minus=" << area[1]
              << " unresolved_area_plus=" << unresolved[0] << " unresolved_area_minus=" << unresolved[1]
              << " sample=" << sample << '\n';
}
#endif
